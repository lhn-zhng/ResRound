#!/usr/bin/env python3
"""Compare FP4 FPROP input-outlier fast path against the reference path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.custom_recipes import quantization

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from megatron.core.extensions.fp4_outlier_recipe import (
    configure_from_transformer_config,
    nvfp4_outlier_quantizer_factory,
)
from megatron.core.extensions.fp4_outlier.storage import (
    make_primary_nvfp4_storage,
)


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be MxKxN, got {text!r}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def configure(
    *,
    ratio: float,
    fast_fprop: bool,
    main_quantizer_rht: bool,
    store_dense_main: bool,
) -> None:
    configure_from_transformer_config(
        SimpleNamespace(
            fp4_outlier_ratio=ratio,
            fp4_outlier_selection_method="normal_threshold",
            fp4_outlier_adaptive_ratio=False,
            fp4_outlier_adaptive_min_ratio=0.0,
            fp4_outlier_adaptive_max_ratio=ratio,
            fp4_outlier_adaptive_reference_heaviness=15.0,
            fp4_outlier_enable_fprop=True,
            fp4_outlier_enable_fast_fprop=fast_fprop,
            fp4_outlier_enable_dgrad=False,
            fp4_outlier_enable_wgrad=False,
            fp4_outlier_enable_nvfp4_a1_a2_all_gather=False,
            fp4_outlier_store_input_dense_main=store_dense_main,
            fp4_outlier_main_quantizer_rht=main_quantizer_rht,
            fp4_outlier_input_stochastic_rounding=False,
        )
    )


def make_data(
    m: int,
    k: int,
    n: int,
    *,
    seed: int,
    spike_frac: float,
    spike_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    dy = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, generator=generator)

    if spike_frac > 0.0 and spike_scale != 0.0:
        count = max(1, int(x.numel() * spike_frac))
        flat = torch.randperm(x.numel(), device="cuda", generator=generator)[:count]
        signs = torch.randint(0, 2, (count,), device="cuda", generator=generator, dtype=torch.int32)
        signs = (signs.to(torch.float32) * 2.0 - 1.0).to(torch.bfloat16)
        x.reshape(-1)[flat] = signs * spike_scale
    return x, w, dy


def stats(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> str:
    if actual is None or expected is None:
        return f"{name}: missing actual={actual is not None} expected={expected is not None}"
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    denom_l2 = expected_f.norm().clamp_min(1.0e-12)
    denom_mae = expected_f.abs().mean().clamp_min(1.0e-12)
    return (
        f"{name}: max_abs={diff.abs().max().item():.6e}, "
        f"mean_abs={diff.abs().mean().item():.6e}, "
        f"rel_l2={(diff.norm() / denom_l2).item():.6e}, "
        f"rel_mae={(diff.abs().mean() / denom_mae).item():.6e}"
    )


def valid_flat_and_values(qresult) -> tuple[torch.Tensor, torch.Tensor]:
    count = qresult.outlier_valid_count()
    flat = qresult.outlier_flat_indices.narrow(0, 0, count).to(torch.int64)
    values = qresult.outlier_values.narrow(0, 0, count)
    return flat, values


def payload_report(ref, fast) -> list[str]:
    ref_flat, ref_values = valid_flat_and_values(ref)
    fast_flat, fast_values = valid_flat_and_values(fast)
    lines = [
        (
            "payload: "
            f"ref_nnz={ref_flat.numel()}, fast_nnz={fast_flat.numel()}, "
            f"fast_backend={getattr(fast, 'fast_fprop_select_quant_backend', None)}, "
            f"fast_actual_ratio={getattr(fast, 'fast_fprop_actual_ratio', None)}"
        )
    ]
    if ref_flat.numel() != fast_flat.numel():
        lines.append("payload_flat: nnz mismatch")
        return lines

    ref_sorted, ref_order = torch.sort(ref_flat)
    fast_sorted, fast_order = torch.sort(fast_flat)
    flat_equal = bool(torch.equal(ref_sorted, fast_sorted))
    lines.append(f"payload_flat: sorted_equal={flat_equal}")
    if flat_equal:
        ref_values_sorted = ref_values[ref_order]
        fast_values_sorted = fast_values[fast_order]
        lines.append(stats("payload_values", fast_values_sorted, ref_values_sorted))
    else:
        mismatch = int((ref_sorted != fast_sorted).sum().item())
        lines.append(f"payload_flat: sorted_mismatch_count={mismatch}")

    ref_offsets = ref.outlier_row_offsets
    fast_offsets = fast.outlier_row_offsets
    if ref_offsets is not None and fast_offsets is not None:
        lines.append(f"row_offsets_equal={bool(torch.equal(ref_offsets, fast_offsets))}")
    return lines


def dequant_primary(qresult) -> torch.Tensor:
    storage = make_primary_nvfp4_storage(qresult, use_rowwise=True, use_columnwise=False)
    out = storage.dequantize(dtype=torch.bfloat16)
    return out.reshape(qresult.logical_2d_shape())


def default_columnwise_report(ref, fast) -> list[str]:
    return [
        stats(
            "default_data_t_raw",
            fast.default_data_t.float() if fast.default_data_t is not None else None,
            ref.default_data_t.float() if ref.default_data_t is not None else None,
        ),
        stats(
            "default_scale_t_raw",
            fast.default_scale_t.float() if fast.default_scale_t is not None else None,
            ref.default_scale_t.float() if ref.default_scale_t is not None else None,
        ),
        stats("default_amax_col", fast.default_global_amax_col, ref.default_global_amax_col),
    ]


def qgemm_fprop(linear_input, qx, qw, *, ratio: float, main_rht: bool, fast_sparse: bool) -> torch.Tensor:
    configure(
        ratio=ratio,
        fast_fprop=fast_sparse,
        main_quantizer_rht=main_rht,
        store_dense_main=True,
    )
    return linear_input.qgemm(
        qx.data,
        qw.data,
        quantization.MMParams(out_dtype=torch.bfloat16, use_split_accumulator=True),
        torch.bfloat16,
        qx.scale,
        qw.scale,
        gemm_type=quantization.GEMMType.FPROP,
        qresult_x=qx,
        qresult_w=qw,
    )


def qgemm_wgrad(linear_grad_output, qdy, qx) -> torch.Tensor:
    return linear_grad_output.qgemm(
        qdy.data_t,
        qx.data_t,
        quantization.MMParams(out_dtype=torch.bfloat16, use_split_accumulator=True),
        torch.bfloat16,
        qdy.scale_t,
        qx.scale_t,
        gemm_type=quantization.GEMMType.WGRAD,
        qresult_x=qdy,
        qresult_w=qx,
    )


def make_recipe(*, ratio: float, fast: bool, main_rht: bool) -> CustomRecipe:
    configure(
        ratio=ratio,
        fast_fprop=fast,
        main_quantizer_rht=main_rht,
        store_dense_main=False,
    )
    return CustomRecipe(qfactory=nvfp4_outlier_quantizer_factory)


def te_linear_run(
    x: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor,
    *,
    ratio: float,
    fast: bool,
    main_rht: bool,
    seed: int,
    module_kind: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recipe = make_recipe(ratio=ratio, fast=fast, main_rht=main_rht)
    if module_kind == "linear":
        module = te.Linear(
            int(x.shape[1]),
            int(w.shape[0]),
            bias=False,
            params_dtype=torch.bfloat16,
            device=x.device,
            name=f"debug_te_linear_fast_{int(fast)}",
        )
    elif module_kind == "layernorm_linear":
        module = te.LayerNormLinear(
            int(x.shape[1]),
            int(w.shape[0]),
            bias=False,
            params_dtype=torch.bfloat16,
            device=x.device,
            normalization="RMSNorm",
            zero_centered_gamma=True,
            name=f"debug_te_layernorm_linear_fast_{int(fast)}",
        )
    else:
        raise ValueError(f"Unknown TE module kind {module_kind!r}.")
    with torch.no_grad():
        module.weight.copy_(w)
        if hasattr(module, "layer_norm_weight"):
            module.layer_norm_weight.fill_(1.0)
    module.train(True)
    module.weight.requires_grad_(True)
    if hasattr(module, "layer_norm_weight"):
        module.layer_norm_weight.requires_grad_(True)
    inp = x.detach().clone().requires_grad_(True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        out = module(inp, is_first_microbatch=None)
    torch.cuda.synchronize()

    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 1)
    out.backward(dy)
    torch.cuda.synchronize()

    out_detached = out.detach()
    grad_input = inp.grad.detach().clone()
    grad_weight = module.weight.grad.detach().clone()
    grad_ln_weight = None
    if hasattr(module, "layer_norm_weight") and module.layer_norm_weight.grad is not None:
        grad_ln_weight = module.layer_norm_weight.grad.detach().clone()
    del module, inp, out
    torch.cuda.empty_cache()
    return out_detached, grad_input, grad_weight, grad_ln_weight


def te_linear_backward_report(
    x: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor,
    *,
    ratio: float,
    main_rht: bool,
    seed: int,
    module_kind: str,
) -> list[str]:
    ref_out, ref_dx, ref_dw, ref_dln = te_linear_run(
        x,
        w,
        dy,
        ratio=ratio,
        fast=False,
        main_rht=main_rht,
        seed=seed,
        module_kind=module_kind,
    )
    fast_out, fast_dx, fast_dw, fast_dln = te_linear_run(
        x,
        w,
        dy,
        ratio=ratio,
        fast=True,
        main_rht=main_rht,
        seed=seed,
        module_kind=module_kind,
    )
    prefix = "te_linear" if module_kind == "linear" else "te_layernorm_linear"
    lines = [
        stats(f"{prefix}_forward_fast_vs_ref", fast_out, ref_out),
        stats(f"{prefix}_dgrad_fast_vs_ref", fast_dx, ref_dx),
        stats(f"{prefix}_wgrad_fast_vs_ref", fast_dw, ref_dw),
    ]
    if ref_dln is not None or fast_dln is not None:
        lines.append(stats(f"{prefix}_ln_weight_grad_fast_vs_ref", fast_dln, ref_dln))
    return lines


def quantize_xw(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    ratio: float,
    fast: bool,
    main_rht: bool,
    columnwise_source: str,
):
    old_source = os.environ.get("FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE")
    os.environ["FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE"] = columnwise_source
    try:
        configure(
            ratio=ratio,
            fast_fprop=fast,
            main_quantizer_rht=main_rht,
            store_dense_main=True,
        )
        linear_input = nvfp4_outlier_quantizer_factory("linear_input")
        linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")
        qx = linear_input.quantize(x)
        qw = linear_weight.quantize(w)
        torch.cuda.synchronize()
        return linear_input, qx, qw
    finally:
        if old_source is None:
            os.environ.pop("FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE", None)
        else:
            os.environ["FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE"] = old_source


def run_one(args: argparse.Namespace, shape: tuple[int, int, int]) -> None:
    m, k, n = shape
    print(f"\n=== shape MxKxN={m}x{k}x{n}, ratio={args.ratio}, main_rht={args.main_rht} ===")
    x, w, dy = make_data(
        m,
        k,
        n,
        seed=args.seed,
        spike_frac=args.spike_frac,
        spike_scale=args.spike_scale,
    )

    ref_linear_input, qx_ref, qw_ref = quantize_xw(
        x,
        w,
        ratio=args.ratio,
        fast=False,
        main_rht=args.main_rht,
        columnwise_source=args.columnwise_source,
    )
    fast_linear_input, qx_fast, qw_fast = quantize_xw(
        x,
        w,
        ratio=args.ratio,
        fast=True,
        main_rht=args.main_rht,
        columnwise_source=args.columnwise_source,
    )

    for line in payload_report(qx_ref, qx_fast):
        print(line)
    print(stats("dense_main", qx_fast.dense_main, qx_ref.dense_main))
    print(stats("primary_dequant_A1", dequant_primary(qx_fast), dequant_primary(qx_ref)))
    for line in default_columnwise_report(qx_ref, qx_fast):
        print(line)

    y_ref = qgemm_fprop(
        ref_linear_input,
        qx_ref,
        qw_ref,
        ratio=args.ratio,
        main_rht=args.main_rht,
        fast_sparse=False,
    )
    y_fast_fallback_corr = qgemm_fprop(
        fast_linear_input,
        qx_fast,
        qw_fast,
        ratio=args.ratio,
        main_rht=args.main_rht,
        fast_sparse=False,
    )
    y_fast_direct_corr = qgemm_fprop(
        fast_linear_input,
        qx_fast,
        qw_fast,
        ratio=args.ratio,
        main_rht=args.main_rht,
        fast_sparse=True,
    )
    torch.cuda.synchronize()

    print(stats("fprop_fast_payload_plus_fallback_corr_vs_ref", y_fast_fallback_corr, y_ref))
    print(stats("fprop_fast_direct_corr_vs_ref", y_fast_direct_corr, y_ref))
    print(stats("fprop_direct_corr_vs_fallback_corr", y_fast_direct_corr, y_fast_fallback_corr))

    configure(
        ratio=args.ratio,
        fast_fprop=False,
        main_quantizer_rht=args.main_rht,
        store_dense_main=True,
    )
    linear_grad_output = nvfp4_outlier_quantizer_factory("linear_grad_output")
    qdy = linear_grad_output.quantize(dy)
    dw_ref = qgemm_wgrad(linear_grad_output, qdy, qx_ref)
    dw_fast = qgemm_wgrad(linear_grad_output, qdy, qx_fast)
    torch.cuda.synchronize()
    print(stats("wgrad_fast_x_default_vs_ref", dw_fast, dw_ref))

    qx_ref.update_usage(rowwise_usage=False, columnwise_usage=True)
    qx_fast.update_usage(rowwise_usage=False, columnwise_usage=True)
    dw_ref_saved_style = qgemm_wgrad(linear_grad_output, qdy, qx_ref)
    dw_fast_saved_style = qgemm_wgrad(linear_grad_output, qdy, qx_fast)
    torch.cuda.synchronize()
    print(stats("wgrad_after_update_usage_fast_vs_ref", dw_fast_saved_style, dw_ref_saved_style))

    if args.te_linear_check:
        for line in te_linear_backward_report(
            x,
            w,
            dy,
            ratio=args.ratio,
            main_rht=args.main_rht,
            seed=args.seed,
            module_kind="linear",
        ):
            print(line)

    if args.te_layernorm_linear_check:
        for line in te_linear_backward_report(
            x,
            w,
            dy,
            ratio=args.ratio,
            main_rht=args.main_rht,
            seed=args.seed,
            module_kind="layernorm_linear",
        ):
            print(line)

    del (
        x,
        w,
        dy,
        qx_ref,
        qw_ref,
        qx_fast,
        qw_fast,
        y_ref,
        y_fast_fallback_corr,
        y_fast_direct_corr,
        qdy,
        dw_ref,
        dw_fast,
        dw_ref_saved_style,
        dw_fast_saved_style,
    )
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(512, 1024, 1024), (4096, 1024, 1024)],
        help="One or more MxKxN shapes.",
    )
    parser.add_argument("--ratio", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--main-rht", action="store_true")
    parser.add_argument("--columnwise-source", default="direct", choices=["direct", "te_default_direct", "outlier_reuse"])
    parser.add_argument("--spike-frac", type=float, default=0.0)
    parser.add_argument("--spike-scale", type=float, default=0.0)
    parser.add_argument("--te-linear-check", action="store_true")
    parser.add_argument("--te-layernorm-linear-check", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError("NVFP4 requires Blackwell or newer GPU.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    for shape in args.shapes:
        run_one(args, shape)


if __name__ == "__main__":
    main()
