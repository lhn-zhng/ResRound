#!/usr/bin/env python3
"""Benchmark FP4 FPROP sparse correction paths.

The script intentionally measures both correction-only and full FPROP qgemm
latency.  Correction-only timings include an output reset copy; a copy-only
baseline is reported so the correction delta is visible.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from megatron.core.extensions.fp4_outlier.fast_fprop import (  # noqa: E402
    try_add_sparse_correction_inplace,
)
from megatron.core.extensions.fp4_outlier.gemm import (  # noqa: E402
    finalize_gemm_result,
    run_fprop_input_outlier_qgemm,
)
from megatron.core.extensions.fp4_outlier.sparse import (  # noqa: E402
    compute_sparse_correction,
)
from megatron.core.extensions.fp4_outlier_recipe import (  # noqa: E402
    configure_from_transformer_config,
    nvfp4_outlier_quantizer_factory,
)


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    k: int
    n: int


@dataclass
class Timing:
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def parse_shape(spec: str) -> Shape:
    if ":" in spec:
        name, dims = spec.split(":", 1)
    else:
        name, dims = "", spec
    parts = dims.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be NAME:MxKxN or MxKxN, got {spec!r}")
    m, k, n = (int(part) for part in parts)
    if not name:
        name = f"{m}x{k}x{n}"
    return Shape(name=name, m=m, k=k, n=n)


def configure_recipe(ratio: float) -> None:
    cfg = SimpleNamespace(
        fp4_outlier_ratio=ratio,
        fp4_outlier_selection_method="normal_threshold",
        fp4_outlier_adaptive_ratio=False,
        fp4_outlier_adaptive_min_ratio=0.0,
        fp4_outlier_adaptive_max_ratio=ratio,
        fp4_outlier_adaptive_reference_heaviness=15.0,
        fp4_outlier_enable_fprop=True,
        fp4_outlier_projection_ratio=ratio,
        fp4_outlier_attn_input_ratio=ratio,
        fp4_outlier_mlp_input_ratio=ratio,
        fp4_outlier_weight_ratio=ratio,
        fp4_outlier_use_double_quant=True,
        fp4_outlier_use_te_double_quant=False,
        fp4_outlier_row_chunk_size=128,
        fp4_outlier_input_store_transpose=False,
        fp4_outlier_projector_store_transpose=False,
        fp4_outlier_store_input_dense_main=False,
        fp4_outlier_fprop_store_input_dense_main=False,
        fp4_outlier_main_quantizer_rht=False,
        fp4_outlier_input_stochastic_rounding=False,
        fp4_outlier_weight_stochastic_rounding=False,
        fp4_outlier_enable_fast_fprop=True,
        fp4_outlier_fast_fprop_force_compile=False,
        fp4_outlier_fast_fprop_force_fallback=False,
        fp4_outlier_fast_fprop_store_dense_main=False,
        fp4_outlier_fast_fprop_defer_selected_sync=False,
        fp4_outlier_fast_fprop_emit_coo=False,
        fp4_outlier_fast_fprop_packed_sparse_ag=False,
    )
    configure_from_transformer_config(cfg)


def make_inputs(shape: Shape, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16, generator=gen)
    return x, w


def clear_sparse_cache(qresult) -> None:
    for attr in ("_cached_sparse_coo", "_cached_sparse_coo_t", "_cached_sparse_csr"):
        if hasattr(qresult, attr):
            delattr(qresult, attr)


def sparse_correction(qx, qw, rebuild_cache: bool) -> torch.Tensor | None:
    if rebuild_cache:
        clear_sparse_cache(qx)
    return compute_sparse_correction(qx, qw.dense_ref, transpose_dense=True)


def full_qgemm(qx, qw, *, use_direct: bool, rebuild_cache: bool) -> torch.Tensor:
    result = run_fprop_input_outlier_qgemm(
        qresult_x=qx,
        qresult_w=qw,
        out_dtype=torch.bfloat16,
        use_split_accumulator=True,
    )
    if use_direct:
        os.environ["FP4_OUTLIER_FAST_FPROP_DISABLE_DIRECT_SPARSE"] = "0"
        applied = try_add_sparse_correction_inplace(result, qresult_x=qx, qresult_w=qw)
        if not applied:
            raise RuntimeError("direct sparse correction was not applied")
        return finalize_gemm_result(result, out_dtype=torch.bfloat16, out=None, accumulate=False)

    correction = sparse_correction(qx, qw, rebuild_cache=rebuild_cache)
    return finalize_gemm_result(
        result,
        out_dtype=torch.bfloat16,
        out=None,
        accumulate=False,
        correction=correction,
    )


def time_cuda(name: str, fn: Callable[[], object], warmup: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return Timing(
        name=name,
        mean_ms=statistics.fmean(times),
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
    )


def compare_outputs(base_result: torch.Tensor, qx, qw) -> tuple[float, float]:
    os.environ["FP4_OUTLIER_FAST_FPROP_DISABLE_DIRECT_SPARSE"] = "0"
    direct = base_result.clone()
    applied = try_add_sparse_correction_inplace(direct, qresult_x=qx, qresult_w=qw)
    if not applied:
        raise RuntimeError("direct sparse correction was not applied")

    fallback = base_result.clone()
    correction = sparse_correction(qx, qw, rebuild_cache=True)
    fallback = finalize_gemm_result(
        fallback,
        out_dtype=torch.bfloat16,
        out=fallback,
        accumulate=False,
        correction=correction,
    )
    diff = (direct.float() - fallback.float()).abs()
    max_abs = diff.max().item()
    denom = fallback.float().norm().item()
    rel_l2 = (direct.float() - fallback.float()).norm().item() / max(denom, 1e-30)
    return max_abs, rel_l2


def format_timing(t: Timing) -> str:
    return (
        f"{t.name:34s} mean={t.mean_ms:9.3f} ms  "
        f"median={t.median_ms:9.3f}  min={t.min_ms:9.3f}  max={t.max_ms:9.3f}"
    )


def run_shape(shape: Shape, args: argparse.Namespace) -> None:
    print(f"\n== {shape.name}: M={shape.m} K={shape.k} N={shape.n}, ratio={args.ratio} ==")
    configure_recipe(args.ratio)
    x, w = make_inputs(shape, args.seed)

    q_input = nvfp4_outlier_quantizer_factory("linear_input")
    q_weight = nvfp4_outlier_quantizer_factory("linear_weight")

    with torch.no_grad():
        qx = q_input.quantize(x)
        qw = q_weight.quantize(w)
        base_result = run_fprop_input_outlier_qgemm(
            qresult_x=qx,
            qresult_w=qw,
            out_dtype=torch.bfloat16,
            use_split_accumulator=True,
        )
    torch.cuda.synchronize()

    selected = int(qx.outlier_valid_count())
    print(f"selected={selected} ({selected / (shape.m * shape.k):.6%} of input elements)")

    max_abs, rel_l2 = compare_outputs(base_result, qx, qw)
    print(f"direct_vs_fallback_once: max_abs={max_abs:.6g}, rel_l2={rel_l2:.6g}")

    work = torch.empty_like(base_result)

    def reset_only() -> torch.Tensor:
        work.copy_(base_result)
        return work

    def direct_correction_only() -> torch.Tensor:
        work.copy_(base_result)
        os.environ["FP4_OUTLIER_FAST_FPROP_DISABLE_DIRECT_SPARSE"] = "0"
        applied = try_add_sparse_correction_inplace(work, qresult_x=qx, qresult_w=qw)
        if not applied:
            raise RuntimeError("direct sparse correction was not applied")
        return work

    def fallback_correction_cached() -> torch.Tensor:
        work.copy_(base_result)
        correction = sparse_correction(qx, qw, rebuild_cache=False)
        return finalize_gemm_result(
            work,
            out_dtype=torch.bfloat16,
            out=work,
            accumulate=False,
            correction=correction,
        )

    def fallback_correction_rebuild() -> torch.Tensor:
        work.copy_(base_result)
        correction = sparse_correction(qx, qw, rebuild_cache=True)
        return finalize_gemm_result(
            work,
            out_dtype=torch.bfloat16,
            out=work,
            accumulate=False,
            correction=correction,
        )

    def input_quant_only():
        return q_input.quantize(x)

    def input_quant_full_direct() -> torch.Tensor:
        qx_iter = q_input.quantize(x)
        return full_qgemm(qx_iter, qw, use_direct=True, rebuild_cache=False)

    def input_quant_full_fallback() -> torch.Tensor:
        qx_iter = q_input.quantize(x)
        return full_qgemm(qx_iter, qw, use_direct=False, rebuild_cache=False)

    # Populate the fallback cache before measuring the cached variant.
    _ = sparse_correction(qx, qw, rebuild_cache=True)
    torch.cuda.synchronize()

    timings = [
        time_cuda("copy_only", reset_only, args.warmup, args.iters),
        time_cuda("direct_correction_only", direct_correction_only, args.warmup, args.iters),
        time_cuda("fallback_correction_cached", fallback_correction_cached, args.warmup, args.iters),
        time_cuda("fallback_correction_rebuild", fallback_correction_rebuild, args.warmup, args.iters),
        time_cuda(
            "full_qgemm_direct",
            lambda: full_qgemm(qx, qw, use_direct=True, rebuild_cache=False),
            args.warmup,
            args.iters,
        ),
        time_cuda(
            "full_qgemm_fallback_cached",
            lambda: full_qgemm(qx, qw, use_direct=False, rebuild_cache=False),
            args.warmup,
            args.iters,
        ),
        time_cuda(
            "full_qgemm_fallback_rebuild",
            lambda: full_qgemm(qx, qw, use_direct=False, rebuild_cache=True),
            args.warmup,
            args.iters,
        ),
        time_cuda("input_quant_only", input_quant_only, args.warmup, args.iters),
        time_cuda("input_quant_full_direct", input_quant_full_direct, args.warmup, args.iters),
        time_cuda("input_quant_full_fallback", input_quant_full_fallback, args.warmup, args.iters),
    ]

    for timing in timings:
        print(format_timing(timing))

    by_name = {timing.name: timing for timing in timings}
    copy_mean = by_name["copy_only"].mean_ms
    direct_net = by_name["direct_correction_only"].mean_ms - copy_mean
    fallback_cached_net = by_name["fallback_correction_cached"].mean_ms - copy_mean
    fallback_rebuild_net = by_name["fallback_correction_rebuild"].mean_ms - copy_mean
    full_direct = by_name["full_qgemm_direct"].mean_ms
    full_fallback_rebuild = by_name["full_qgemm_fallback_rebuild"].mean_ms
    input_full_direct = by_name["input_quant_full_direct"].mean_ms
    input_full_fallback = by_name["input_quant_full_fallback"].mean_ms

    def ratio(a: float, b: float) -> float:
        return math.inf if b == 0 else a / b

    print(
        "summary: "
        f"direct_net={direct_net:.3f} ms, "
        f"fallback_cached_net={fallback_cached_net:.3f} ms "
        f"({ratio(fallback_cached_net, direct_net):.2f}x direct), "
        f"fallback_rebuild_net={fallback_rebuild_net:.3f} ms "
        f"({ratio(fallback_rebuild_net, direct_net):.2f}x direct), "
        f"full_fallback_rebuild/full_direct={ratio(full_fallback_rebuild, full_direct):.3f}x, "
        f"input_quant_full_fallback/input_quant_full_direct="
        f"{ratio(input_full_fallback, input_full_direct):.3f}x"
    )

    del work, base_result, qx, qw, x, w
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=[
            parse_shape("qkv:32768x1024x3072"),
            parse_shape("proj:32768x1024x1024"),
            parse_shape("fc1:32768x1024x5632"),
            parse_shape("fc2:32768x2816x1024"),
        ],
    )
    parser.add_argument("--ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)

    print(f"torch={torch.__version__}, cuda_device={torch.cuda.get_device_name()}")
    print(f"warmup={args.warmup}, iters={args.iters}")
    for shape in args.shapes:
        run_shape(shape, args)


if __name__ == "__main__":
    main()
