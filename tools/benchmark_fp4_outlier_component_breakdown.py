from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from transformer_engine.pytorch.custom_recipes import quantization


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTED_PY = REPO_ROOT / "collected/nvfp4_sparse_comm_schedules/python"
DEFAULT_DATASET = REPO_ROOT / "test_tb/real_fprop_dump_llama3_8b_steps_1_500_1000"
DEFAULT_OUT_PREFIX = REPO_ROOT / "logs/fp4_outlier_component_breakdown"

if str(COLLECTED_PY) not in sys.path:
    sys.path.insert(0, str(COLLECTED_PY))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_schedules import load_group, parse_ratio, rank_dir, select_groups  # noqa: E402
from benchmark_te_module import base_row  # noqa: E402
from common import get_rank, tensor_metrics, time_cuda_op  # noqa: E402
from te_functional_baseline import make_te_input_quantizer  # noqa: E402

from megatron.core.extensions.fp4_outlier.fast_fprop import (  # noqa: E402
    _capacity_from_ratio,
    try_add_sparse_correction_inplace,
)
from megatron.core.extensions.fp4_outlier.gemm import (  # noqa: E402
    run_fprop_input_outlier_qgemm,
    run_qgemm,
)
from megatron.core.extensions.fp4_outlier_recipe import (  # noqa: E402
    configure_from_transformer_config,
    nvfp4_outlier_quantizer_factory,
    set_layer_name,
)


def maybe_init_dist() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or dist.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))


def configure_recipe(args: argparse.Namespace) -> None:
    if args.capacity_ratio_floor is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_CAPACITY_RATIO_FLOOR"] = str(
            args.capacity_ratio_floor
        )
    if args.capacity_multiplier is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_CAPACITY_MULTIPLIER"] = str(
            args.capacity_multiplier
        )
    if args.max_capacity_ratio is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_MAX_CAPACITY_RATIO"] = str(
            args.max_capacity_ratio
        )
    if args.capacity_ratio_floor_by_shape is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_CAPACITY_RATIO_FLOOR_BY_SHAPE"] = (
            args.capacity_ratio_floor_by_shape
        )
    if args.capacity_multiplier_by_shape is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_CAPACITY_MULTIPLIER_BY_SHAPE"] = (
            args.capacity_multiplier_by_shape
        )
    if args.max_capacity_ratio_by_shape is not None:
        os.environ["FP4_OUTLIER_FAST_FPROP_MAX_CAPACITY_RATIO_BY_SHAPE"] = (
            args.max_capacity_ratio_by_shape
        )

    configure_from_transformer_config(
        SimpleNamespace(
            fp4_outlier_ratio=parse_ratio(args.ratio),
            fp4_outlier_selection_method=args.selection_method,
            fp4_outlier_adaptive_ratio=False,
            fp4_outlier_adaptive_min_ratio=0.0,
            fp4_outlier_adaptive_max_ratio=0.01,
            fp4_outlier_adaptive_reference_heaviness=15.0,
            fp4_outlier_enable_fprop=True,
            fp4_outlier_enable_fast_fprop=args.fast_fprop,
            fp4_outlier_enable_dgrad=False,
            fp4_outlier_enable_wgrad=False,
            fp4_outlier_enable_nvfp4_a1_a2_all_gather=False,
            fp4_outlier_store_input_dense_main=False,
            fp4_outlier_main_quantizer_rht=False,
            fp4_outlier_input_stochastic_rounding=False,
        )
    )


def make_quantizers():
    return (
        nvfp4_outlier_quantizer_factory("linear_input"),
        nvfp4_outlier_quantizer_factory("linear_weight"),
    )


def fprop_qgemm(qx, qw) -> torch.Tensor:
    return run_qgemm(
        qresult_x=qx,
        qresult_w=qw,
        m_params=quantization.MMParams(
            out_dtype=torch.bfloat16,
            use_split_accumulator=True,
        ),
        out_dtype=torch.bfloat16,
        bias=None,
        out=None,
        accumulate=False,
        gemm_type=quantization.GEMMType.FPROP,
    )


def time_ms(
    op: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    group,
) -> float:
    return time_cuda_op(op, warmup=warmup, iters=iters, group=group)


def selected_nnz(qx) -> int:
    if getattr(qx, "fast_fprop_selected_nnz", None) is not None:
        selected = int(qx.fast_fprop_selected_nnz)
        if selected >= 0:
            return selected
    count_hint = getattr(qx, "_outlier_local_count_hint", None)
    if isinstance(count_hint, torch.Tensor) and int(count_hint.numel()) == 1:
        return int(count_hint.reshape(-1)[0].detach().cpu().item())
    row_offsets = getattr(qx, "outlier_row_offsets", None)
    if row_offsets is not None and int(row_offsets.numel()) > 0:
        return int(row_offsets[-1].detach().cpu().item())
    values = getattr(qx, "outlier_values", None)
    return 0 if values is None else int(values.numel())


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    try:
        m, k, n = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers") from exc
    if m <= 0 or k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return m, k, n


def make_synthetic_case(
    *,
    shape_text: str,
    module_suffix: str,
    index: int,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    m, k, n = parse_shape(shape_text)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed) + get_rank() * 1009 + index)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    group_row = {
        "group_index": int(index),
        "step": 0,
        "module_name": f"synthetic.{module_suffix}",
        "module_suffix": module_suffix,
    }
    return group_row, x.contiguous(), weight.contiguous()


def benchmark_group(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    group_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    group = dist.group.WORLD if dist.is_initialized() else None
    configure_recipe(args)
    linear_input, linear_weight = make_quantizers()
    module_name = str(
        group_row.get("module_name", group_row.get("module_suffix", "unknown"))
    )
    set_layer_name(linear_input, f"{module_name}.input_quantizer")
    set_layer_name(linear_weight, f"{module_name}.weight_quantizer")
    te_input_quantizer = make_te_input_quantizer()

    ref = x @ weight.t()
    qx = linear_input.quantize(x)
    qw = linear_weight.quantize(weight)
    out = fprop_qgemm(qx, qw)
    torch.cuda.synchronize()
    metrics = tensor_metrics(out.detach(), ref)

    nnz = selected_nnz(qx)
    numel = int(x.numel())
    configured_ratio = parse_ratio(args.ratio)
    actual_ratio = float(nnz) / float(max(1, numel))
    configured_capacity = _capacity_from_ratio(
        numel,
        configured_ratio,
        shape=(int(x.shape[0]), int(x.shape[1])),
    )
    requested_capacity = int(
        getattr(qx, "fast_fprop_requested_capacity", configured_capacity)
    )
    payload_capacity = int(getattr(qx, "fast_fprop_payload_capacity", nnz))
    overflow = int(getattr(qx, "fast_fprop_overflow", -1))
    configured_nnz = float(max(1.0, numel * configured_ratio))
    capacity_expanded = int(payload_capacity > requested_capacity)

    qx_for_sparse = qx
    qw_for_sparse = qw
    dense_only = run_fprop_input_outlier_qgemm(
        qresult_x=qx_for_sparse,
        qresult_w=qw_for_sparse,
        out_dtype=torch.bfloat16,
        use_split_accumulator=True,
    )
    torch.cuda.synchronize()
    sparse_scratch = torch.empty_like(dense_only)

    def quant_input():
        return linear_input.quantize(x)

    def te_quant_input_only():
        return te_input_quantizer(x.contiguous())

    def quant_weight():
        return linear_weight.quantize(weight)

    def custom_full_uncached():
        local_qx = linear_input.quantize(x)
        local_qw = linear_weight.quantize(weight)
        return fprop_qgemm(local_qx, local_qw)

    def custom_full_cached_weight():
        local_qx = linear_input.quantize(x)
        return fprop_qgemm(local_qx, qw)

    def dense_gemm_only():
        return run_fprop_input_outlier_qgemm(
            qresult_x=qx_for_sparse,
            qresult_w=qw_for_sparse,
            out_dtype=torch.bfloat16,
            use_split_accumulator=True,
        )

    def sparse_correction_only():
        sparse_scratch.zero_()
        ok = try_add_sparse_correction_inplace(
            sparse_scratch,
            qresult_x=qx_for_sparse,
            qresult_w=qw_for_sparse,
        )
        if not ok:
            raise RuntimeError("fast sparse correction path was not available")
        return sparse_scratch

    row = {
        **base_row(group_row, x, weight),
        "ratio": configured_ratio,
        "ratio_pct": configured_ratio * 100.0,
        "selection_method": args.selection_method,
        "fast_fprop": int(args.fast_fprop),
        "configured_capacity": int(configured_capacity),
        "requested_capacity": int(requested_capacity),
        "payload_capacity": int(payload_capacity),
        "configured_capacity_ratio": float(configured_capacity) / float(max(1, numel)),
        "requested_capacity_ratio": float(requested_capacity) / float(max(1, numel)),
        "payload_capacity_ratio": float(payload_capacity) / float(max(1, numel)),
        "configured_capacity_over_configured_ratio": float(configured_capacity)
        / configured_nnz,
        "requested_capacity_over_configured_ratio": float(requested_capacity)
        / configured_nnz,
        "payload_capacity_over_configured_ratio": float(payload_capacity) / configured_nnz,
        "capacity_over_configured_ratio": float(payload_capacity) / configured_nnz,
        "capacity_expanded": capacity_expanded,
        "fast_fprop_overflow": overflow,
        "selected_nnz": int(nnz),
        "actual_ratio": actual_ratio,
        "actual_ratio_pct": actual_ratio * 100.0,
        "actual_over_configured_ratio": actual_ratio / max(configured_ratio, 1.0e-12),
        "requested_capacity_over_actual_nnz": float(requested_capacity)
        / float(max(1, nnz)),
        "payload_capacity_over_actual_nnz": float(payload_capacity) / float(max(1, nnz)),
        "capacity_over_actual_nnz": float(payload_capacity) / float(max(1, nnz)),
        "bf16_gemm_ms": time_ms(
            lambda: x @ weight.t(),
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "custom_full_uncached_ms": time_ms(
            custom_full_uncached,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "custom_full_cached_weight_ms": time_ms(
            custom_full_cached_weight,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "input_select_quant_ms": time_ms(
            quant_input,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "te_input_quant_only_ms": time_ms(
            te_quant_input_only,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "weight_quant_ms": time_ms(
            quant_weight,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "dense_gemm_only_ms": time_ms(
            dense_gemm_only,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "sparse_correction_only_ms": time_ms(
            sparse_correction_only,
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "qgemm_dense_plus_sparse_ms": time_ms(
            lambda: fprop_qgemm(qx_for_sparse, qw_for_sparse),
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        ),
        "max_abs_vs_bf16_ref": metrics["max_abs"],
        "rel_l2_vs_bf16_ref": metrics["rel_l2"],
        "warmup": args.warmup,
        "iters": args.iters,
    }
    return row


def write_rows(prefix: Path, rows: list[dict[str, Any]]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rank = get_rank()
    out_prefix = prefix if rank == 0 else prefix.with_name(f"{prefix.name}_rank{rank:03d}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# FP4 Outlier Component Breakdown",
        "",
        (
            "| Rank | Module | Shape | Ratio | Actual | Payload/Ratio | "
            "Payload/Actual | Expand | BF16 | Cached | Select+quant | TE quant | Dense | Sparse |"
        ),
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["module_suffix"]),
                    f"`{row['shape']}`",
                    f"{float(row['ratio_pct']):.3f}%",
                    f"{float(row['actual_ratio_pct']):.3f}%",
                    f"{float(row['payload_capacity_over_configured_ratio']):.2f}x",
                    f"{float(row['payload_capacity_over_actual_nnz']):.2f}x",
                    str(int(row["capacity_expanded"])),
                    f"{float(row['bf16_gemm_ms']):.4f}",
                    f"{float(row['custom_full_cached_weight_ms']):.4f}",
                    f"{float(row['input_select_quant_ms']):.4f}",
                    f"{float(row['te_input_quant_only_ms']):.4f}",
                    f"{float(row['dense_gemm_only_ms']):.4f}",
                    f"{float(row['sparse_correction_only_ms']):.4f}",
                ]
            )
            + " |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rank", default=None)
    parser.add_argument("--module-suffixes", nargs="+", default=["linear_qkv", "linear_fc1"])
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--ratio", default="0.1%")
    parser.add_argument("--ratios", nargs="+", default=None)
    parser.add_argument(
        "--selection-method",
        choices=["topk", "normal_threshold"],
        default="normal_threshold",
    )
    parser.add_argument("--fast-fprop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capacity-ratio-floor", type=float, default=None)
    parser.add_argument("--capacity-multiplier", type=float, default=None)
    parser.add_argument("--max-capacity-ratio", type=float, default=None)
    parser.add_argument("--capacity-ratio-floor-by-shape", default=None)
    parser.add_argument("--capacity-multiplier-by-shape", default=None)
    parser.add_argument("--max-capacity-ratio-by-shape", default=None)
    parser.add_argument("--synthetic-shapes", nargs="*", default=None)
    parser.add_argument("--synthetic-module-suffixes", nargs="*", default=None)
    parser.add_argument("--synthetic-seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    maybe_init_dist()
    cases: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = []
    if args.synthetic_shapes:
        suffixes = args.synthetic_module_suffixes or []
        if suffixes and len(suffixes) != len(args.synthetic_shapes):
            raise RuntimeError("--synthetic-module-suffixes must match --synthetic-shapes")
        for index, shape_text in enumerate(args.synthetic_shapes):
            suffix = suffixes[index] if suffixes else f"synthetic_{index}"
            cases.append(
                make_synthetic_case(
                    shape_text=shape_text,
                    module_suffix=suffix,
                    index=index,
                    seed=args.synthetic_seed,
                )
            )
    else:
        rank_name = args.rank or f"rank{get_rank():03d}"
        rank_path = rank_dir(args.dataset_dir, rank_name)
        groups = select_groups(rank_path, args.module_suffixes, args.max_groups)
        if not groups:
            raise RuntimeError(f"No matching groups for {args.module_suffixes} in {rank_path}")
        for group_row in groups:
            group_row = {
                **group_row,
                "module_suffix": str(
                    group_row.get(
                        "module_suffix",
                        str(group_row["module_name"]).rsplit(".", 1)[-1],
                    )
                ),
            }
            x, weight = load_group(rank_path, group_row)
            cases.append((group_row, x, weight))

    rows: list[dict[str, Any]] = []
    for ratio_text in args.ratios or [args.ratio]:
        args.ratio = ratio_text
        for group_row, x, weight in cases:
            row = benchmark_group(x=x, weight=weight, group_row=group_row, args=args)
            rows.append(row)
            print(
                f"rank={row['rank']} {row['module_suffix']} shape={row['shape']} "
                f"ratio={float(row['ratio_pct']):.3f}% "
                f"actual={float(row['actual_ratio_pct']):.3f}% "
                f"payload/config={float(row['payload_capacity_over_configured_ratio']):.2f}x "
                f"payload/actual={float(row['payload_capacity_over_actual_nnz']):.2f}x "
                f"expanded={int(row['capacity_expanded'])} "
                f"bf16={float(row['bf16_gemm_ms']):.4f}ms "
                f"cached={float(row['custom_full_cached_weight_ms']):.4f}ms "
                f"input_quant={float(row['input_select_quant_ms']):.4f}ms "
                f"te_quant={float(row['te_input_quant_only_ms']):.4f}ms "
                f"dense={float(row['dense_gemm_only_ms']):.4f}ms "
                f"sparse={float(row['sparse_correction_only_ms']):.4f}ms "
                f"rel_l2={float(row['rel_l2_vs_bf16_ref']):.6g}",
                flush=True,
            )

    write_rows(args.out_prefix, rows)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
