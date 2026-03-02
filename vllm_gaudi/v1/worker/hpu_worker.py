# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import contextlib
import gc
import os
import time
import queue
import math
import sys
import types
import torch.nn.functional as F
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed
import torch.nn as nn
import habana_frameworks.torch as htorch
from vllm.tasks import SupportedTask
from vllm_gaudi.extension.debug import init_debug_logger
from vllm_gaudi.extension.defragmentation import OnlineDefragmenter
from vllm_gaudi.extension.profiler import (HabanaMemoryProfiler, format_bytes, setup_profiler)
from vllm_gaudi.extension.runtime import get_config

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (ensure_model_parallel_initialized, init_distributed_environment)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.utils.torch_utils import (STR_DTYPE_TO_TORCH_DTYPE, set_random_seed)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, MambaSpec)
from vllm.v1.outputs import (DraftTokenIds, AsyncModelRunnerOutput, ModelRunnerOutput)
from vllm.v1.worker.utils import bind_kv_cache
from vllm_gaudi.utils import is_fake_hpu
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import GrammarOutput, SchedulerOutput


def _patch_vllm_fla_gated_delta_rule_to_torch() -> None:
    """Replace CUDA/Triton FLA gated-delta-rule ops with a torch reference on HPU.

    Upstream FLA ops use CUDA contexts and Triton kernels (e.g. wrapper calls
    torch.cuda.device(...)), which fails on HPU builds without CUDA. :contentReference[oaicite:2]{index=2}
    Qwen3.5 (qwen3_next) uses these ops for prefill/decode in GDN attention. :contentReference[oaicite:3]{index=3}
    """
    try:
        from vllm.model_executor.layers.fla import ops as ops_mod
        from vllm.model_executor.layers.fla.ops import chunk as chunk_mod
        from vllm.model_executor.layers.fla.ops import fused_recurrent as rec_mod
    except Exception:
        return

    if getattr(ops_mod, "_VLLM_GAUDI_PATCHED_FLA_GDR", False):
        return

    def _expand_qk_to_hv(q: torch.Tensor, k: torch.Tensor, hv: int):
        # q,k: [B,T,H,K] -> [B,T,HV,K] if HV>H (GVA)
        B, T, H, K = q.shape
        if hv == H:
            return q, k
        if hv % H != 0:
            raise RuntimeError(f"HV({hv}) must be divisible by H({H})")
        r = hv // H
        return q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)

    def _l2norm(x: torch.Tensor, eps: float = 1e-6):
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

    def _gdr_torch_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        inplace_final_state: bool = False,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ):
        # Shapes (as used by qwen3_next):
        # q,k: [B,T,H,K], v: [B,T,HV,V], g,beta: [B,T,HV] (or beta: [B,T,HV,V])
        assert q.dtype == k.dtype == v.dtype, "q/k/v dtypes must match"
        B, T, H, K = q.shape
        HV = v.shape[2]
        Vd = v.shape[3]

        if scale is None:
            scale = K ** -0.5

        # expand q,k to HV heads if needed (GVA case)
        q_hv, k_hv = _expand_qk_to_hv(q, k, HV)

        # Output: [B,T,HV,V]
        out = torch.empty((B, T, HV, Vd), device=v.device, dtype=v.dtype)

        # Decide varlen view
        if cu_seqlens is None:
            # equal-length: N == B
            cu_list = None
            N = B
            def _bos_eos(n): return n * T, (n + 1) * T
        else:
            # varlen: expected B==1, tokens flattened along T
            cu_list = cu_seqlens.detach().cpu().tolist()
            N = len(cu_list) - 1
            def _bos_eos(n): return cu_list[n], cu_list[n + 1]

        # Prepare initial state access
        if initial_state is None:
            # create zero state if missing
            initial_state = torch.zeros((N, HV, Vd, K), device=v.device, dtype=v.dtype)

        # We'll compute in fp32 for stability (similar intent as kernels)
        # State tensor may be updated in place if requested.
        def _get_state_view(n: int):
            if ssm_state_indices is None:
                return initial_state[n]
            idx = int(ssm_state_indices[n].detach().cpu().item())
            return initial_state[idx]

        # helper to maybe write back
        def _write_back(n: int, state_fp32: torch.Tensor):
            if not inplace_final_state:
                return
            if ssm_state_indices is None:
                initial_state[n].copy_(state_fp32.to(initial_state.dtype))
            else:
                idx = int(ssm_state_indices[n].detach().cpu().item())
                initial_state[idx].copy_(state_fp32.to(initial_state.dtype))

        final_states = []  # fp32 states (only when output_final_state and not inplace)

        # Main loop (slow but correctness-first)
        for n in range(N):
            bos, eos = _bos_eos(n)
            if eos <= bos:
                # keep state as-is
                if output_final_state and not inplace_final_state:
                    final_states.append(_get_state_view(n).to(torch.float32))
                continue

            # state: [HV,V,K]
            state0 = _get_state_view(n)
            h = state0.to(torch.float32)

            # token iteration
            for t in range(bos, eos):
                bidx = 0 if cu_list is not None else n
                # [HV,K]
                qt = q_hv[bidx, t].to(torch.float32)
                kt = k_hv[bidx, t].to(torch.float32)
                # [HV,V]
                vt = v[bidx, t].to(torch.float32)
                # [HV] or [HV,K]
                gt = g[bidx, t].to(torch.float32)
                bt = beta[bidx, t].to(torch.float32)

                if use_qk_l2norm_in_kernel:
                    qt = _l2norm(qt)
                    kt = _l2norm(kt)
                qt = qt * float(scale)

                # decay: h *= exp(g)
                if gt.dim() == 1:
                    h = h * torch.exp(gt).view(HV, 1, 1)
                else:
                    # per-K decay (rare): gt [HV,K]
                    h = h * torch.exp(gt).view(HV, 1, K)

                # proj = h @ k  -> [HV,V]
                proj = torch.einsum("hvk,hk->hv", h, kt)
                upd = vt - proj

                # beta can be [HV] or [HV,V]
                if bt.dim() == 1:
                    upd = upd * bt.view(HV, 1)
                else:
                    upd = upd * bt

                # h += upd ⊗ k
                h = h + torch.einsum("hv,hk->hvk", upd, kt)

                # out = h @ q
                ot = torch.einsum("hvk,hk->hv", h, qt)
                out[bidx, t].copy_(ot.to(out.dtype))

            if inplace_final_state:
                _write_back(n, h)
            if output_final_state and not inplace_final_state:
                final_states.append(h)

        if output_final_state:
            if inplace_final_state:
                # mimic kernel behavior: return the (mutated) initial_state
                return out, initial_state
            # stack fp32 states then cast back to input dtype
            fs = torch.stack(final_states, dim=0).to(initial_state.dtype)
            return out, fs
        return out, None

    # Public replacements matching upstream signatures
    def chunk_gated_delta_rule_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        return _gdr_torch_impl(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            inplace_final_state=False,
            ssm_state_indices=None,
            num_accepted_tokens=None,
        )

    def fused_recurrent_gated_delta_rule_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        if beta is None:
            beta = torch.ones_like(g, dtype=g.dtype, device=g.device)
        return _gdr_torch_impl(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=True,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            inplace_final_state=inplace_final_state,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    # Patch canonical module functions
    chunk_mod.chunk_gated_delta_rule = chunk_gated_delta_rule_torch
    rec_mod.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule_torch
    ops_mod.chunk_gated_delta_rule = chunk_gated_delta_rule_torch
    ops_mod.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule_torch
    ops_mod._VLLM_GAUDI_PATCHED_FLA_GDR = True

    # Patch already-imported aliases in qwen3_next/qwen3_5 modules if present
    for modname in (
        "vllm.model_executor.models.qwen3_next",
        "vllm.model_executor.models.qwen3_5",
    ):
        try:
            m = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        if hasattr(m, "fla_chunk_gated_delta_rule"):
            setattr(m, "fla_chunk_gated_delta_rule", chunk_gated_delta_rule_torch)
        if hasattr(m, "fused_recurrent_gated_delta_rule"):
            setattr(m, "fused_recurrent_gated_delta_rule", fused_recurrent_gated_delta_rule_torch)

# Apply early in worker process import (before model execution)
_patch_vllm_fla_gated_delta_rule_to_torch()


def _patch_qwen3_next_gdn_gating_torch_fallback() -> None:
    """Patch Qwen3.5/Qwen3Next fused_gdn_gating to avoid Triton on HPU.

    qwen3_next defines fused_gdn_gating with a Triton kernel and calls
    triton.cdiv(...) (via `from vllm.triton_utils import triton`). On Gaudi/HPU
    this often fails because vllm.triton_utils may provide a placeholder module
    without cdiv, and Triton kernels aren't intended for HPU anyway.

    Replace fused_gdn_gating with a pure torch implementation:
      g = -exp(A_log) * softplus(a + dt_bias)
      beta_output = sigmoid(b)
    which matches the kernel math (using torch softplus for stability).
    """
    try:
        from vllm.model_executor.models import qwen3_next as q3n
    except Exception:
        return

    if getattr(q3n, "_VLLM_GAUDI_PATCHED_GDN_GATING", False):
        return

    def fused_gdn_gating_torch(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ):
        # Promote to fp32 for the same stability intent as the Triton kernel.
        a_f = a.to(torch.float32)
        b_f = b.to(torch.float32)
        A_f = A_log.to(torch.float32)
        dt_f = dt_bias.to(torch.float32)

        # Broadcast dt_bias / A_log to match a's shape (token_dim..., heads).
        while dt_f.dim() < a_f.dim():
            dt_f = dt_f.unsqueeze(0)
        while A_f.dim() < a_f.dim():
            A_f = A_f.unsqueeze(0)

        # softplus(a + dt_bias) with beta/threshold for numerical stability.
        softplus_x = F.softplus(a_f + dt_f, beta=beta, threshold=threshold)
        g = -torch.exp(A_f) * softplus_x  # fp32
        beta_out = torch.sigmoid(b_f).to(b.dtype)

        # Upstream expects (1, tokens, heads) for this call site.
        if g.dim() == 2:
            g = g.unsqueeze(0)
        if beta_out.dim() == 2:
            beta_out = beta_out.unsqueeze(0)
        return g, beta_out

    q3n.fused_gdn_gating = fused_gdn_gating_torch
    q3n._VLLM_GAUDI_PATCHED_GDN_GATING = True

# Apply patch early in worker process import.
_patch_qwen3_next_gdn_gating_torch_fallback()

def _patch_vllm_mamba_causal_conv1d_to_gaudi_pytorch() -> None:
    """Route vLLM upstream Mamba causal_conv1d ops to Gaudi PyTorch impl.

    Upstream uses Triton kernels via `from vllm.triton_utils import triton`,
    which is not suitable on HPU and can miss helpers like next_power_of_2/cdiv.
    vllm-gaudi provides a PyTorch reference implementation; use it on HPU.
    """
    try:
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as cv
        from vllm_gaudi.ops import causal_conv1d_pytorch as gcv
    except Exception:
        return

    # Keep originals so we can detect and patch already-imported aliases.
    _orig_fn = getattr(cv, "causal_conv1d_fn", None)
    _orig_upd = getattr(cv, "causal_conv1d_update", None)

    if getattr(cv, "_VLLM_GAUDI_PATCHED_CAUSAL_CONV1D", False):
        return

    # Wrap upstream signatures -> gaudi reference impl.
    def _hpu_causal_conv1d_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        activation: str | None = "silu",
        pad_slot_id: int = -1,  # ignored (gaudi impl uses PAD_SLOT_ID internally)
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        block_size_to_align: int = 0,
        metadata=None,
        validate_data: bool = False,
    ):
        # NOTE: gaudi reference impl will raise NotImplementedError if
        # prefix-caching metadata is provided. For now we assume these are None.
        # qwen3_next passes conv_state as (B, dim, state_len) (it does transpose(-1, -2)),
        # while gaudi pytorch ref expects (B, state_len, dim).
        conv_states_ref = conv_states
        try:
            dim = int(weight.size(0))
            state_len = int(weight.size(1) - 1)
            if conv_states_ref is not None and conv_states_ref.dim() == 3:
                # (B, dim, state_len) -> (B, state_len, dim)
                if conv_states_ref.size(-2) == dim and conv_states_ref.size(-1) == state_len:
                    conv_states_ref = conv_states_ref.transpose(-1, -2)
        except Exception:
            pass

        out = gcv.hpu_causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states_ref,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens,
            block_size_to_align=block_size_to_align,
            metadata=metadata,
            validate_data=validate_data,
            is_prompt=True,
        )

        return out

    def _hpu_causal_conv1d_update(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: bool | str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = -1,
        pad_slot_id: int = -1,  # ignored
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        validate_data: bool = False,
    ):
        # ------------------------------------------------------------------
        # Layout fix for gaudi PyTorch reference impl:
        # gcv._flatten_inputs_for_update(query_start_loc=None) expects x shaped
        # as (batch, dim) or (batch, dim, seqlen), and checks x_3d.size(1)==dim.
        # If x arrives as (dim, batch) (channel-first), transpose to token-first.
        # ------------------------------------------------------------------
        try:
            dim = int(weight.size(0))
            if query_start_loc is None:
                if x is not None and x.dim() == 2:
                    # channel-first (dim, batch) -> token-first (batch, dim)
                    if x.size(0) == dim and x.size(1) != dim:
                        x = x.transpose(0, 1).contiguous()
                elif x is not None and x.dim() == 3:
                    # handle common mis-orders:
                    # (dim, batch, seqlen) -> (batch, dim, seqlen)
                    if x.size(0) == dim and x.size(1) != dim:
                        x = x.permute(1, 0, 2).contiguous()
                    # (batch, seqlen, dim) -> (batch, dim, seqlen)
                    elif x.size(-1) == dim and x.size(1) != dim:
                        x = x.permute(0, 2, 1).contiguous()
            # gaudi update impl expects 3D: (batch, dim, cu_seqlen). For decode,
            # cu_seqlen is usually 1, but must be present.
            if x is not None and x.dim() == 2:
                x = x.unsqueeze(-1)
        except Exception:
            pass

        conv_state_ref = conv_state
        try:
            dim = int(weight.size(0))
            state_len = int(weight.size(1) - 1)
            if conv_state_ref is not None and conv_state_ref.dim() == 3:
                if conv_state_ref.size(-2) == dim and conv_state_ref.size(-1) == state_len:
                    conv_state_ref = conv_state_ref.transpose(-1, -2)
        except Exception:
            pass

        out = gcv.hpu_causal_conv1d_update(
            x,
            conv_state_ref,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            max_query_len=max_query_len,
            pad_slot_id=pad_slot_id,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            validate_data=validate_data,
        )

        return out

    cv.causal_conv1d_fn = _hpu_causal_conv1d_fn
    cv.causal_conv1d_update = _hpu_causal_conv1d_update
    cv._VLLM_GAUDI_PATCHED_CAUSAL_CONV1D = True

    # qwen3_next imports causal_conv1d_fn/update via `from ... import ...`,
    # so patch the bound symbols in that module as well. :contentReference[oaicite:2]{index=2}
    for modname in (
        "vllm.model_executor.models.qwen3_next",
        "vllm.model_executor.models.qwen3_5",
    ):
        try:
            m = __import__(modname, fromlist=["*"])
        except Exception:
            continue
        if getattr(m, "causal_conv1d_fn", None) is _orig_fn:
            setattr(m, "causal_conv1d_fn", _hpu_causal_conv1d_fn)
        if getattr(m, "causal_conv1d_update", None) is _orig_upd:
            setattr(m, "causal_conv1d_update", _hpu_causal_conv1d_update)

    # Optional: patch any other already-imported aliases.
    for _, m in list(sys.modules.items()):
        if m is None:
            continue
        if getattr(m, "causal_conv1d_fn", None) is _orig_fn:
            setattr(m, "causal_conv1d_fn", _hpu_causal_conv1d_fn)
        if getattr(m, "causal_conv1d_update", None) is _orig_upd:
            setattr(m, "causal_conv1d_update", _hpu_causal_conv1d_update)

# Apply early in worker process import
_patch_vllm_mamba_causal_conv1d_to_gaudi_pytorch()

def setup_step_profiler(steps):
    if steps is None:
        return None
    step_start, step_end = steps
    active = step_end - step_start + 1
    return setup_profiler(warmup=0, active=active)


class HPUWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.local_rank = local_rank
        self.rank = rank
        self.parallel_config.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        self.gc_track_recompiles = get_config().track_graph_compilation and not get_config().high_level_profiler_enabled
        self.step = 0
        self.profile_steps = get_config().VLLM_PROFILE_STEPS
        self.step_profiler = setup_step_profiler(self.profile_steps)
        self.step_debug = init_debug_logger('steps')

        self.model_sleeping = False
        self.kv_cache_sleeping = False
        self.kv_cache_config = None

    def init_profiler(self):
        """Initialize the profiler."""
        torch_profiler_dir = os.getenv('VLLM_TORCH_PROFILER_DIR')
        if torch_profiler_dir:
            logger.warning("VLLM_TORCH_PROFILER_DIR is deprecated!")
            torch_profiler_trace_dir = torch_profiler_dir
            logger.info("Profiling enabled. Traces will be saved to: %s", torch_profiler_trace_dir)
            if os.getenv('VLLM_PROFILER_ENABLED') == 'full':
                fn = self.model_runner.profiler.full_trace_handler
                with_stack = False
            else:
                fn = torch.profiler.tensorboard_trace_handler
                with_stack = True
            self.profiler = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.HPU,
            ],
                                                   with_stack=with_stack,
                                                   on_trace_ready=fn(torch_profiler_trace_dir, use_gzip=True))

        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        high_level_profiler = self.model_runner.profiler
        with high_level_profiler.record_event('internal', 'start_profiler'):
            # Clean up the queue
            while True:
                try:
                    high_level_profiler.profiling_trace_events.get_nowait()
                except queue.Empty:
                    break
            self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def init_device(self):
        self.device = torch.device("hpu")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        self.model_runner = HPUModelRunner(vllm_config=self.vllm_config, is_driver_worker=self.is_driver_worker)
        self.init_profiler()

    def shutdown(self):
        getattr(self.model_runner, 'shutdown_inc', lambda: None)()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        single_kv_block_size_bytes = 0
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, FullAttentionSpec):
                dtype = layer_spec.dtype
                if dtype == torch.float8_e4m3fn and os.environ.get('QUANT_CONFIG', None) is not None and \
                    os.environ.get('VLLM_DYNAMIC_KV_QUANT', None) is not None and not self.model_config.use_mla:
                    create_dynamic_scales = True
                else:
                    create_dynamic_scales = False

                # Create dummy KV cache tensors with proper shapes for profiling
                num_blocks = 1  # Use single block for profiling
                block_size = layer_spec.block_size
                num_kv_heads = layer_spec.num_kv_heads
                head_size = layer_spec.head_size

                kv_cache_shape = self.model_runner.attn_backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                                                                                   head_size)
                kv_scales_shape = kv_cache_shape[:-1] + (1, )

                hpu_k_cache = torch.zeros(kv_cache_shape, dtype=dtype, device='hpu')
                hpu_v_cache = None if self.model_config.use_mla else torch.zeros(
                    kv_cache_shape, dtype=dtype, device='hpu')

                hpu_k_scales = torch.ones(kv_scales_shape, dtype=torch.bfloat16,
                                          device='hpu') if create_dynamic_scales else None
                if create_dynamic_scales:
                    hpu_v_scales = (torch.ones(kv_scales_shape, dtype=torch.bfloat16, device='hpu'),
                                    torch.ones([num_blocks, num_kv_heads, head_size],
                                               dtype=torch.bfloat16,
                                               device='hpu'))
                else:
                    hpu_v_scales = None

                kv_caches[layer_name] = (hpu_k_cache, hpu_v_cache, hpu_k_scales, hpu_v_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes

            elif isinstance(layer_spec, MambaSpec):
                dtype0 = layer_spec.dtypes[0]
                dtype1 = layer_spec.dtypes[1]

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                hpu_ssm_cache = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_cache = torch.tensor([], dtype=dtype1, device='hpu')
                hpu_ssm_scales = torch.tensor([], dtype=dtype0, device='hpu')
                hpu_conv_scales = torch.tensor([], dtype=dtype1, device='hpu')

                kv_caches[layer_name] = (hpu_ssm_cache, hpu_conv_cache, hpu_ssm_scales, hpu_conv_scales)

                single_kv_block_size_bytes += layer_spec.page_size_bytes
            else:
                raise NotImplementedError

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, runner_kv_caches)

        if self.model_runner.unified_attn:
            # Create unified attention persistent context for profiling
            from vllm_gaudi.extension.unified_batch import UnifiedBatchPersistentContext
            self.model_runner.unified_attn_persistent_ctx = UnifiedBatchPersistentContext(
                self.model_runner.max_num_batched_tokens, 0, 0, self.model_runner.block_size, dtype,
                self.model_runner.profiler)

        if is_fake_hpu():
            fake_hpu_cache_alloc = 4 * 2**30  # take 4 GiB flat on fake hpu
            return fake_hpu_cache_alloc
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run(initialize_only=True)
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        graph_reserved_mem = (float(os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.1'))
                              if not self.model_config.enforce_eager else 0)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin
        cache_size_bytes = available_hpu_memory * graph_headroom
        graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
        dummy_block_headroom = single_kv_block_size_bytes
        msg = (f"Free device memory: {format_bytes(free_hpu_memory)}, "
               f"{format_bytes(available_hpu_memory)} usable "
               f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
               f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
               f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
               f"{format_bytes(dummy_block_headroom)} reserved for KV cache dummy "
               f"block {format_bytes(cache_size_bytes - dummy_block_headroom)} "
               "reserved for usable KV cache")

        logger.info(msg)

        # Clear the dummy KV cache to free up memory
        kv_caches = {}
        forward_context = self.vllm_config.compilation_config.static_forward_context
        for layer_name in forward_context:
            forward_context[layer_name].kv_cache = None
        runner_kv_caches = []
        gc.collect()

        return cache_size_bytes - dummy_block_headroom

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> float:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        with HabanaMemoryProfiler() as m:
            self.kv_cache_config = kv_cache_config
            self.model_runner.initialize_kv_cache(kv_cache_config)
            torch.hpu.synchronize()
        if len(self.model_runner.kv_caches) > 0:
            msg = (f"Usable num_blocks: {kv_cache_config.num_blocks}, "
                   f"actual allocated num_blocks: "
                   f"{self.model_runner.kv_caches[0][0].shape[0]} "
                   f"(_PAD_BLOCK_ID={self.model_runner._PAD_BLOCK_ID}, "
                   f"_PAD_SLOT_ID={self.model_runner._PAD_SLOT_ID})")
            logger.info(msg)
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        return self.compile_or_warm_up_model()

    def compile_or_warm_up_model(self) -> float:
        # Don't run the warmup if the model is already warmed up
        start_t = time.perf_counter()
        if not getattr(self.model_runner, 'graphed_buckets', None):
            self.model_runner.warmup_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return time.perf_counter() - start_t

    def sample_tokens(self, grammar_output: "GrammarOutput|None") -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        if self.step_debug:
            self.step_debug(f'step={self.step}')
        if self.step_profiler and self.step == self.profile_steps[0]:
            self.step_profiler.start()
        with track_graph_compile('HPUWorker.execute_model') \
                if self.gc_track_recompiles \
                else contextlib.nullcontext():
            output = self.model_runner.execute_model(scheduler_output)
        # TODO(woosuk): Send the output to the engine process.
        if self.step_profiler:
            if self.step >= self.profile_steps[0]:
                self.step_profiler.step()
            if self.step == self.profile_steps[1]:
                self.step_profiler.stop()
                self.step_profiler = None
                raise RuntimeError('Step profiling finished!')
        self.step += 1
        # NOTE(Harish): removed "if self.rank == 0 else None" for KV_connector enabling with TP>1
        # referred to Gpu Model Runner, KV connector aggregation expects valid output from all ranks
        return output

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode to reduce memory usage. Unlike GPU workers that use custom
        memory allocators, HPU workers use a simpler approach of moving model to CPU and clearing KV cache.
        Args:
            level (int): Sleep level (kept for interface compatibility, always performs level 1 operations)
        """

        if level == 2:
            logger.warning("Currently, HPU does not support level 2 sleep mode. Performing level 1 operations")
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        # Handle model - if model was loaded move it to CPU
        if self.model_sleeping:
            logger.warning("Model is already in a sleep mode, skipping moving it to CPU")
        elif not hasattr(self.model_runner, "model") or self.model_runner.model is None:
            logger.warning("Model was not loaded yet, skipping moving it to CPU")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.model.to("cpu")
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Moving model to CPU for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.model_sleeping = True

        # Handle KV cache - discard it
        if self.kv_cache_sleeping:
            logger.warning("KV cache has already been discarded by calling sleep method and it has not been "
                           "reinitialized by calling wake up method yet, skipping discarding it again")
        elif self.kv_cache_config is None:
            logger.warning("KV cache has not been initialized yet, skipping discarding it")
        else:
            with HabanaMemoryProfiler() as m:
                self.model_runner.defragmenter.cache_utils.kv_caches = None
                self.model_runner.kv_caches = []
                forward_context = self.vllm_config.compilation_config.static_forward_context
                for layer_name in forward_context:
                    forward_context[layer_name].kv_cache = None
                gc.collect()
                torch.hpu.synchronize()
            msg = f"Discarding KV cache for sleep mode took {m.get_summary_string()}"
            logger.info(msg)
            self.kv_cache_sleeping = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up the worker from sleep mode.
        It can move the model back to HPU and/or reinitialize KV cache.

        Args:
            tags: Optional list of tags (kept for interface compatibility)
        """
        assert not htorch.utils.internal.is_lazy(
        ) or self.model_config.enforce_eager, "Sleep mode is supported only for torch.compile mode"

        if tags is None:
            tags = ["weights", "kv_cache"]

        # Handle model - if model was loaded, move it back to HPU
        if "weights" in tags:
            if not self.model_sleeping:
                logger.warning("Model is not in a sleep mode, skipping moving it to HPU")
            elif not hasattr(self.model_runner, "model") or self.model_runner.model is None:
                logger.warning("Model was not loaded yet, skipping moving it to HPU")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.model.to(self.vllm_config.device_config.device)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up model, moving it back to HPU took {m.get_summary_string()}"
                logger.info(msg)
                self.model_sleeping = False

        # Handle KV cache - reinitialize it
        if "kv_cache" in tags:
            if not self.kv_cache_sleeping:
                logger.warning("KV cache is not in a sleep mode, skipping reinitializing it")
            elif self.kv_cache_config is None:
                logger.warning("KV cache config is empty, skipping reinitializing KV cache")
            else:
                with HabanaMemoryProfiler() as m:
                    self.model_runner.initialize_kv_cache(self.kv_cache_config)
                    self.model_runner.defragmenter = OnlineDefragmenter()
                    self.model_runner.defragmenter.initialize(self.model_runner.kv_caches, self.model_runner.block_size)
                    gc.collect()
                    torch.hpu.synchronize()
                msg = f"Waking up KV cache, reinitializing it took {m.get_summary_string()}"
                logger.info(msg)
                self.kv_cache_sleeping = False


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    parallel_config = vllm_config.parallel_config
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size, rank, distributed_init_method, local_rank, backend='hccl')

    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size * parallel_config.data_parallel_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size)


@contextmanager
def track_graph_compile(name: str):
    from habana_frameworks.torch.hpu.metrics import metric_localcontext
    with metric_localcontext("graph_compilation") as gc:
        yield
        htorch.hpu.synchronize()
    if gc.stats()[0][1] != 0:
        msg = f"[{name}] graph compilation detected: {gc.stats()}"
        logger.warning(msg)
