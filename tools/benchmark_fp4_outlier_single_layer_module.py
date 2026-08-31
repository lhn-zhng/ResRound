from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import CustomRecipe, NVFP4BlockScaling


COLLECTED_PY = Path(
    "/workspace/Megatron-LM-312/collected/nvfp4_sparse_comm_schedules/python"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "/workspace/Megatron-LM-312/test_tb/real_fprop_dump_llama3_8b_steps_1_500_1000"
)
DEFAULT_OUT_PREFIX = Path(
    "/workspace/Megatron-LM-312-fprop-input/logs/single_layer_fp4_outlier_module"
)

if str(COLLECTED_PY) not in sys.path:
    sys.path.insert(0, str(COLLECTED_PY))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_schedules import (  # noqa: E402
    dense_bf16_ag_reference,
    dense_bf16_rs_reference,
    load_group,
    parse_ratio,
    rank_dir,
    select_groups,
)
from benchmark_te_module import (  # noqa: E402
    base_row,
    make_te_module,
    prime_te_module_cache,
    te_module_forward_op,
)
from common import get_rank, get_world, tensor_metrics, time_cuda_op  # noqa: E402

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


def apply_capacity_env(args: argparse.Namespace) -> None:
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


def make_recipe(args: argparse.Namespace):
    if args.recipe == "bf16":
        return None
    if args.recipe == "nvfp4":
        return NVFP4BlockScaling()

    apply_capacity_env(args)
    cfg = SimpleNamespace(
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
        fp4_outlier_enable_nvfp4_a1_a2_all_gather=args.sparse_ag,
        fp4_outlier_store_input_dense_main=False,
        fp4_outlier_main_quantizer_rht=False,
        fp4_outlier_input_stochastic_rounding=False,
    )
    configure_from_transformer_config(cfg)
    return CustomRecipe(qfactory=nvfp4_outlier_quantizer_factory)


def bf16_module_forward_op(
    module,
    x: torch.Tensor,
    *,
    train_forward: bool,
):
    inp = x.detach().requires_grad_(train_forward)

    def run() -> torch.Tensor:
        ctx = torch.enable_grad() if train_forward else torch.no_grad()
        with ctx:
            return module(inp)

    return run


def make_benchmark_module(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    group,
    train_forward: bool,
    args: argparse.Namespace,
):
    if args.module_kind == "linear":
        return make_te_module(
            x=x,
            weight=weight,
            mode=mode,
            group=group,
            train_forward=train_forward,
        )

    if mode not in {"ag", "rs"}:
        raise ValueError("mode must be 'ag' or 'rs'")
    world = get_world(group)
    _, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    if mode == "ag":
        in_features = k
        out_features = n * world
        parallel_mode = "column" if world > 1 else None
    else:
        in_features = k * world
        out_features = n
        parallel_mode = "row" if world > 1 else None

    module = te.LayerNormLinear(
        in_features,
        out_features,
        sequence_parallel=world > 1,
        tp_group=group,
        tp_size=world,
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        normalization=args.normalization,
        zero_centered_gamma=args.zero_centered_gamma,
        device=x.device,
        name=f"te_layernorm_linear_{mode}",
    )
    with torch.no_grad():
        module.weight.copy_(weight)
    module.train(mode=train_forward)
    module.weight.requires_grad_(train_forward)
    module.layer_norm_weight.requires_grad_(train_forward)
    return module


def benchmark_one(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    group_row: dict[str, Any],
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    group = dist.group.WORLD if dist.is_initialized() else None
    recipe = make_recipe(args)
    module = make_benchmark_module(
        x=x,
        weight=weight,
        mode=mode,
        group=group,
        train_forward=args.train_forward,
        args=args,
    )
    if args.cached_weight and args.recipe != "bf16":
        prime_te_module_cache(
            module,
            x,
            recipe=recipe,
            train_forward=args.train_forward,
        )
    if args.recipe == "bf16":
        op = bf16_module_forward_op(module, x, train_forward=args.train_forward)
    else:
        op = te_module_forward_op(
            module,
            x,
            recipe=recipe,
            cached_weight=args.cached_weight,
            train_forward=args.train_forward,
        )
    ref = (
        dense_bf16_ag_reference(x, weight, group)
        if mode == "ag"
        else dense_bf16_rs_reference(x, weight, group)
    )
    out = op()
    torch.cuda.synchronize()
    metrics = tensor_metrics(out.detach(), ref)
    ms = time_cuda_op(op, warmup=args.warmup, iters=args.iters, group=group)
    torch.cuda.synchronize()
    del module
    torch.cuda.empty_cache()
    return {
        **base_row(group_row, x, weight),
        "world": get_world(group),
        "recipe": args.recipe,
        "mode": mode,
        "case": f"{args.recipe}_{mode}_{'cached' if args.cached_weight else 'uncached'}",
        "module_kind": args.module_kind,
        "ratio": parse_ratio(args.ratio) if args.recipe == "custom" else 0.0,
        "ratio_pct": parse_ratio(args.ratio) * 100.0 if args.recipe == "custom" else 0.0,
        "selection_method": args.selection_method if args.recipe == "custom" else "",
        "fast_fprop": int(args.fast_fprop) if args.recipe == "custom" else 0,
        "sparse_ag": int(args.sparse_ag) if args.recipe == "custom" else 0,
        "cached_weight": int(args.cached_weight),
        "train_forward": int(args.train_forward),
        "ms": ms,
        "max_abs_vs_bf16_ref": metrics["max_abs"],
        "rel_l2_vs_bf16_ref": metrics["rel_l2"],
        "warmup": args.warmup,
        "iters": args.iters,
    }


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
        "# FP4 Outlier Single-Layer TE Module",
        "",
        "| Rank | Module | Case | Shape | ms | max_abs | rel_l2 |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    str(row["module_suffix"]),
                    f"`{row['case']}`",
                    f"`{row['shape']}`",
                    f"{float(row['ms']):.6f}",
                    f"{float(row['max_abs_vs_bf16_ref']):.6g}",
                    f"{float(row['rel_l2_vs_bf16_ref']):.6g}",
                ]
            )
            + " |"
        )
    out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rank", default=None)
    parser.add_argument("--module-suffixes", nargs="+", default=["linear_proj"])
    parser.add_argument(
        "--synthetic-shape",
        nargs=3,
        type=int,
        metavar=("M", "K", "N"),
        default=None,
        help="Use synthetic BF16 tensors with local x shape MxK and local weight shape NxK.",
    )
    parser.add_argument("--synthetic-module-suffix", default="synthetic_linear")
    parser.add_argument("--synthetic-seed", type=int, default=1234)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=["ag", "rs"], default=["ag", "rs"])
    parser.add_argument("--module-kind", choices=["linear", "layernorm_linear"], default="linear")
    parser.add_argument("--normalization", choices=["LayerNorm", "RMSNorm"], default="RMSNorm")
    parser.add_argument("--zero-centered-gamma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recipe", choices=["bf16", "nvfp4", "custom"], default="custom")
    parser.add_argument("--recipes", nargs="+", choices=["bf16", "nvfp4", "custom"], default=None)
    parser.add_argument("--ratio", default="0.1%")
    parser.add_argument("--ratios", nargs="+", default=None)
    parser.add_argument(
        "--selection-method",
        choices=["topk", "normal_threshold"],
        default="normal_threshold",
    )
    parser.add_argument("--fast-fprop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sparse-ag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capacity-ratio-floor", type=float, default=None)
    parser.add_argument("--capacity-multiplier", type=float, default=None)
    parser.add_argument("--max-capacity-ratio", type=float, default=None)
    parser.add_argument("--capacity-ratio-floor-by-shape", default=None)
    parser.add_argument("--capacity-multiplier-by-shape", default=None)
    parser.add_argument("--max-capacity-ratio-by-shape", default=None)
    parser.add_argument("--cached-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    args = parser.parse_args()

    maybe_init_dist()
    if args.synthetic_shape is None:
        rank_name = args.rank or f"rank{get_rank():03d}"
        rank_path = rank_dir(args.dataset_dir, rank_name)
        groups = select_groups(rank_path, args.module_suffixes, args.max_groups)
        if not groups:
            raise RuntimeError(f"No matching groups for {args.module_suffixes} in {rank_path}")
    else:
        m, k, n = [int(v) for v in args.synthetic_shape]
        groups = [
            {
                "group_index": 0,
                "step": 0,
                "module_name": f"synthetic.{args.synthetic_module_suffix}",
                "module_suffix": args.synthetic_module_suffix,
                "m": m,
                "k": k,
                "n": n,
            }
        ]

    rows: list[dict[str, Any]] = []
    for group_row in groups:
        group_row = {
            **group_row,
            "module_suffix": str(
                group_row.get("module_suffix", str(group_row["module_name"]).rsplit(".", 1)[-1])
            ),
        }
        if args.synthetic_shape is None:
            x, weight = load_group(rank_path, group_row)
        else:
            m, k, n = [int(v) for v in args.synthetic_shape]
            generator = torch.Generator(device="cuda")
            generator.manual_seed(int(args.synthetic_seed) + get_rank())
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
            weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator)
        for recipe in args.recipes or [args.recipe]:
            args.recipe = recipe
            ratio_values = args.ratios or [args.ratio]
            if recipe != "custom":
                ratio_values = [args.ratio]
            for ratio in ratio_values:
                args.ratio = ratio
                for mode in args.modes:
                    row = benchmark_one(
                        x=x,
                        weight=weight,
                        group_row=group_row,
                        mode=mode,
                        args=args,
                    )
                    rows.append(row)
                    print(
                        f"rank={row['rank']} {row['module_suffix']} {row['case']} "
                        f"shape={row['shape']} ratio={float(row['ratio_pct']):.3f}% "
                        f"ms={float(row['ms']):.6f} "
                        f"max_abs={float(row['max_abs_vs_bf16_ref']):.6g} "
                        f"rel_l2={float(row['rel_l2_vs_bf16_ref']):.6g}",
                        flush=True,
                    )

    write_rows(args.out_prefix, rows)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
