from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist


os.environ.setdefault("NVTE_CUSTOM_NVFP4_SPARSE_AG_PACKED_FASTPATH", "1")

COLLECTED_PY = Path(
    "/workspace/Megatron-LM-312/collected/nvfp4_sparse_comm_schedules/python"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "/workspace/Megatron-LM-312/test_tb/real_fprop_dump_llama3_8b_steps_1_500_1000"
)
DEFAULT_OUT_PREFIX = REPO_ROOT / "logs" / "fp4_outlier_ag_comm_only"

if str(COLLECTED_PY) not in sys.path:
    sys.path.insert(0, str(COLLECTED_PY))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_schedules import load_group, parse_ratio, rank_dir, select_groups  # noqa: E402
from benchmark_te_module import base_row  # noqa: E402
from common import get_rank, get_world, time_cuda_op  # noqa: E402
from te_functional_baseline import make_te_input_quantizer  # noqa: E402
from transformer_engine.pytorch import distributed as te_dist  # noqa: E402

from megatron.core.extensions.fp4_outlier_recipe import (  # noqa: E402
    configure_from_transformer_config,
    nvfp4_outlier_quantizer_factory,
)


def maybe_init_dist() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or dist.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))


def selected_nnz(qx) -> int:
    selected = getattr(qx, "fast_fprop_selected_nnz", None)
    if selected is not None and int(selected) >= 0:
        return int(selected)
    count_hint = getattr(qx, "_outlier_local_count_hint", None)
    if isinstance(count_hint, torch.Tensor) and int(count_hint.numel()) == 1:
        return int(count_hint.reshape(-1)[0].detach().cpu().item())
    row_offsets = getattr(qx, "outlier_row_offsets", None)
    if row_offsets is not None and int(row_offsets.numel()) > 0:
        return int(row_offsets[-1].detach().item())
    values = getattr(qx, "outlier_values", None)
    return 0 if values is None else int(values.numel())


def configure_recipe(args: argparse.Namespace, ratio: float) -> None:
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
            fp4_outlier_ratio=ratio,
            fp4_outlier_selection_method=args.selection_method,
            fp4_outlier_adaptive_ratio=False,
            fp4_outlier_adaptive_min_ratio=0.0,
            fp4_outlier_adaptive_max_ratio=0.01,
            fp4_outlier_adaptive_reference_heaviness=15.0,
            fp4_outlier_enable_fprop=True,
            fp4_outlier_enable_fast_fprop=True,
            fp4_outlier_enable_dgrad=False,
            fp4_outlier_enable_wgrad=False,
            fp4_outlier_enable_nvfp4_a1_a2_all_gather=True,
            fp4_outlier_store_input_dense_main=False,
            fp4_outlier_main_quantizer_rht=False,
            fp4_outlier_input_stochastic_rounding=False,
        )
    )


def timed_ms(
    op: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    group,
) -> float:
    return time_cuda_op(op, warmup=warmup, iters=iters, group=group)


def gather_and_wait(qx, quantizer, group):
    gathered, handle = te_dist.gather_along_first_dim(
        qx,
        group,
        async_op=False,
        quantizer=quantizer,
    )
    if handle is not None:
        handle.wait()
    return gathered


def clone_without_sparse(qx):
    saved = {
        name: getattr(qx, name, None)
        for name in (
            "outlier_rows",
            "outlier_cols",
            "outlier_values",
            "outlier_flat_indices",
            "outlier_row_offsets",
            "_outlier_payload_full_capacity",
            "_outlier_full_capacity_flat_indices",
            "_outlier_full_capacity_values",
            "_outlier_local_count_hint",
        )
    }
    for name in saved:
        if hasattr(qx, name):
            setattr(qx, name, None)
    return saved


def restore_sparse(qx, saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        if value is not None or hasattr(qx, name):
            setattr(qx, name, value)


def benchmark_group(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    group_row: dict[str, Any],
    ratio: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    group = dist.group.WORLD if dist.is_initialized() else None
    configure_recipe(args, ratio)

    te_quantizer = make_te_input_quantizer()
    custom_quantizer = nvfp4_outlier_quantizer_factory("linear_input")

    te_qx = te_quantizer(x.contiguous())
    custom_qx = custom_quantizer.quantize(x.contiguous())
    torch.cuda.synchronize()

    selected = selected_nnz(custom_qx)
    actual_ratio = float(selected) / float(max(1, int(x.numel())))
    payload_capacity = int(getattr(custom_qx, "fast_fprop_payload_capacity", selected))
    packed_payload = bool(getattr(custom_qx, "_outlier_payload_full_capacity", False))

    te_quantizer.set_usage(rowwise=True, columnwise=False)
    custom_quantizer.set_usage(rowwise=True, columnwise=False)

    # Prime TE/NCCL allocation caches before measuring.
    gather_and_wait(te_qx, te_quantizer, group)
    gather_and_wait(custom_qx, custom_quantizer, group)
    torch.cuda.synchronize()

    timed_ms(
        lambda: gather_and_wait(te_qx, te_quantizer, group),
        warmup=max(1, min(2, args.warmup)),
        iters=max(1, min(5, args.iters)),
        group=group,
    )
    timed_ms(
        lambda: gather_and_wait(custom_qx, custom_quantizer, group),
        warmup=max(1, min(2, args.warmup)),
        iters=max(1, min(5, args.iters)),
        group=group,
    )

    te_ag_ms = timed_ms(
        lambda: gather_and_wait(te_qx, te_quantizer, group),
        warmup=args.warmup,
        iters=args.iters,
        group=group,
    )

    custom_ag_ms = timed_ms(
        lambda: gather_and_wait(custom_qx, custom_quantizer, group),
        warmup=args.warmup,
        iters=args.iters,
        group=group,
    )

    saved_sparse = clone_without_sparse(custom_qx)
    try:
        custom_dense_ag_ms = timed_ms(
            lambda: gather_and_wait(custom_qx, custom_quantizer, group),
            warmup=args.warmup,
            iters=args.iters,
            group=group,
        )
    finally:
        restore_sparse(custom_qx, saved_sparse)

    sparse_payload_ag_ms = custom_ag_ms - custom_dense_ag_ms

    return {
        **base_row(group_row, x, weight),
        "world": get_world(group),
        "ratio": ratio,
        "ratio_pct": ratio * 100.0,
        "actual_ratio": actual_ratio,
        "actual_ratio_pct": actual_ratio * 100.0,
        "selected_nnz": selected,
        "payload_capacity": payload_capacity,
        "packed_sparse_payload": int(packed_payload),
        "te_ag_comm_ms": te_ag_ms,
        "custom_dense_ag_comm_ms": custom_dense_ag_ms,
        "custom_sparse_payload_ag_comm_ms": sparse_payload_ag_ms,
        "custom_total_ag_comm_ms": custom_ag_ms,
        "custom_over_te_ag_comm": custom_ag_ms / te_ag_ms if te_ag_ms > 0 else float("nan"),
        "warmup": args.warmup,
        "iters": args.iters,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rank", default=None)
    parser.add_argument("--module-suffixes", nargs="+", default=["linear_proj"])
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--ratios", nargs="+", default=["0.1%"])
    parser.add_argument("--selection-method", default="normal_threshold")
    parser.add_argument("--capacity-ratio-floor-by-shape", default=None)
    parser.add_argument("--capacity-multiplier-by-shape", default=None)
    parser.add_argument("--max-capacity-ratio-by-shape", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    maybe_init_dist()
    rank_name = args.rank or f"rank{get_rank():03d}"
    rank_path = rank_dir(args.dataset_dir, rank_name)
    groups = select_groups(rank_path, args.module_suffixes, args.max_groups)
    if not groups:
        raise RuntimeError(f"No matching groups for {args.module_suffixes} in {rank_path}")

    rows: list[dict[str, Any]] = []
    for ratio_text in args.ratios:
        ratio = parse_ratio(ratio_text)
        for group_row in groups:
            x, weight = load_group(rank_path, group_row)
            row = benchmark_group(x=x, weight=weight, group_row=group_row, ratio=ratio, args=args)
            rows.append(row)
            if get_rank() == 0:
                print(
                    f"{row['module_suffix']} ratio={row['ratio_pct']:.1f}% "
                    f"actual={row['actual_ratio_pct']:.3f}% "
                    f"te_ag={row['te_ag_comm_ms']:.4f}ms "
                    f"custom_ag={row['custom_total_ag_comm_ms']:.4f}ms "
                    f"payload_delta={row['custom_sparse_payload_ag_comm_ms']:.4f}ms",
                    flush=True,
                )

    out_prefix = (
        args.out_prefix
        if get_rank() == 0
        else args.out_prefix.with_name(f"{args.out_prefix.name}_rank{get_rank():03d}")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
