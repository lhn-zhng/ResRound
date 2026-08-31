# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from transformer_engine.pytorch.custom_recipes import quantization

from megatron.core.extensions.fp4_outlier_recipe import (
    configure_from_transformer_config,
    get_config,
    nvfp4_outlier_quantizer_factory,
)
from megatron.core.extensions.fp4_outlier.split import resolve_outlier_ratio, split_outliers
from megatron.core.extensions.fp4_outlier.storage import make_primary_nvfp4_storage


def _configure(**kwargs):
    values = dict(
        fp4_outlier_ratio=0.01,
        fp4_outlier_selection_method="topk",
        fp4_outlier_adaptive_ratio=False,
        fp4_outlier_adaptive_min_ratio=0.0,
        fp4_outlier_adaptive_max_ratio=0.01,
        fp4_outlier_adaptive_reference_heaviness=15.0,
        fp4_outlier_enable_fprop=True,
        fp4_outlier_enable_fast_fprop=False,
        fp4_outlier_enable_dgrad=False,
        fp4_outlier_enable_wgrad=False,
        fp4_outlier_enable_nvfp4_a1_a2_all_gather=False,
    )
    values.update(kwargs)
    configure_from_transformer_config(SimpleNamespace(**values))


def test_factory_is_fprop_input_only():
    _configure(fp4_outlier_ratio=0.25)

    cfg = get_config()
    assert cfg.enable_fprop is True
    assert cfg.enable_fast_fprop is False
    assert cfg.enable_dgrad is False
    assert cfg.enable_wgrad is False

    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")
    linear_grad_output = nvfp4_outlier_quantizer_factory("linear_grad_output")

    assert linear_input.apply_outlier_split is True
    assert linear_input.outlier_ratio == 0.25
    assert linear_weight.apply_outlier_split is False
    assert linear_weight.store_dense_ref is True
    assert linear_grad_output.apply_outlier_split is False


def test_adaptive_ratio_resolves_from_tensor_heaviness():
    _configure(
        fp4_outlier_ratio=0.003,
        fp4_outlier_adaptive_ratio=True,
        fp4_outlier_adaptive_min_ratio=0.0,
        fp4_outlier_adaptive_max_ratio=0.01,
        fp4_outlier_adaptive_reference_heaviness=15.0,
    )
    cfg = get_config()

    light = torch.ones(8, 8, dtype=torch.bfloat16)
    light_ratio = resolve_outlier_ratio(
        light,
        base_ratio=cfg.outlier_ratio,
        adaptive_enabled=cfg.adaptive_outlier_ratio,
        adaptive_min_ratio=cfg.adaptive_outlier_min_ratio,
        adaptive_max_ratio=cfg.adaptive_outlier_max_ratio,
        adaptive_reference_heaviness=cfg.adaptive_outlier_reference_heaviness,
    )
    assert light_ratio.heaviness == pytest.approx(1.0)
    assert light_ratio.effective == 0.0

    heavy = torch.ones(8, 8, dtype=torch.bfloat16)
    heavy[0, 0] = 1000
    heavy_ratio = resolve_outlier_ratio(
        heavy,
        base_ratio=cfg.outlier_ratio,
        adaptive_enabled=cfg.adaptive_outlier_ratio,
        adaptive_min_ratio=cfg.adaptive_outlier_min_ratio,
        adaptive_max_ratio=cfg.adaptive_outlier_max_ratio,
        adaptive_reference_heaviness=cfg.adaptive_outlier_reference_heaviness,
    )
    assert heavy_ratio.heaviness > cfg.adaptive_outlier_reference_heaviness
    assert cfg.outlier_ratio < heavy_ratio.effective <= cfg.adaptive_outlier_max_ratio


def test_split_uses_adaptive_effective_ratio_for_selection():
    x = torch.arange(100, dtype=torch.bfloat16).reshape(10, 10)
    _, _, _, values, _, _ = split_outliers(
        x,
        outlier_ratio=0.5,
        effective_outlier_ratio=0.0,
        selection_method="topk",
    )
    assert values is None

    _, _, _, values, _, _ = split_outliers(
        x,
        outlier_ratio=0.5,
        effective_outlier_ratio=0.1,
        selection_method="topk",
    )
    assert values is not None
    assert int(values.numel()) == 10


def test_factory_threads_sparse_all_gather_flag_to_input_only():
    _configure(fp4_outlier_enable_nvfp4_a1_a2_all_gather=True)

    cfg = get_config()
    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")
    linear_grad_output = nvfp4_outlier_quantizer_factory("linear_grad_output")

    assert cfg.enable_nvfp4_a1_a2_all_gather is True
    assert linear_input.enable_sparse_all_gather is True
    assert linear_input.needs_runtime_main_scale is False
    assert linear_input._fast_nvfp4_quantizer is linear_input._main_nvfp4_quantizer
    assert linear_weight.enable_sparse_all_gather is False
    assert linear_grad_output.enable_sparse_all_gather is False


def test_config_threads_fast_fprop_flag():
    _configure(fp4_outlier_enable_fast_fprop=True)

    cfg = get_config()
    assert cfg.enable_fast_fprop is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_input_quantize_records_adaptive_ratio_metadata():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure(
        fp4_outlier_ratio=0.003,
        fp4_outlier_selection_method="normal_threshold",
        fp4_outlier_adaptive_ratio=True,
        fp4_outlier_adaptive_min_ratio=0.0,
        fp4_outlier_adaptive_max_ratio=0.01,
        fp4_outlier_adaptive_reference_heaviness=15.0,
    )
    x = torch.ones(32, 64, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1000

    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    qx = linear_input.quantize(x)

    assert qx.configured_outlier_ratio == pytest.approx(0.003)
    assert qx.outlier_heaviness is not None
    assert qx.outlier_heaviness > 1.0
    assert qx.effective_outlier_ratio is not None
    assert 0.0 <= qx.effective_outlier_ratio <= 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_weight_workspace_generation_invalidates_transpose_cache():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure()
    weight = torch.nn.Parameter(
        torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    )
    linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")
    qweight = linear_weight.quantize(weight)
    first_generation = qweight.weight_update_generation
    setattr(qweight, "_fast_fprop_weight_t_state", ("stale", weight.t().contiguous()))

    original_version = weight._version
    weight.data.add_(1)
    assert weight._version == original_version

    linear_weight.update_quantized(weight, qweight)
    assert qweight.weight_update_generation == first_generation + 1
    assert getattr(qweight, "_fast_fprop_weight_t_state", None) is None
    torch.testing.assert_close(qweight.dense_ref, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_input_quantize_exposes_sparse_all_gather_payload():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure(fp4_outlier_ratio=0.25, fp4_outlier_enable_nvfp4_a1_a2_all_gather=True)
    torch.manual_seed(321)
    torch.cuda.manual_seed(321)

    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    qx = linear_input.quantize(x)

    assert qx.data is not None
    assert qx.scale is not None
    assert qx.outlier_rows is not None
    assert qx.outlier_cols is not None
    assert qx.outlier_values is not None
    assert qx.outlier_flat_indices is not None
    assert qx.outlier_row_offsets is not None
    assert qx.outlier_row_offsets.numel() == x.shape[0] + 1
    assert int(qx.outlier_row_offsets[-1].item()) == int(qx.outlier_values.numel())
    assert qx.default_data is None
    assert qx.default_data_t is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_primary_storage_prefers_gathered_quantizer():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure(fp4_outlier_ratio=0.25, fp4_outlier_enable_nvfp4_a1_a2_all_gather=True)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    qx = linear_input.quantize(x)

    gathered_quantizer = qx._primary_quantizer.copy()
    gathered_quantizer.with_rht = not qx._primary_quantizer.with_rht
    qx._gathered_primary_quantizer = gathered_quantizer

    storage = make_primary_nvfp4_storage(qx, use_rowwise=True, use_columnwise=False)
    assert storage._quantizer.with_rht == gathered_quantizer.with_rht


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_all_input_outliers_match_bf16_fprop_reference():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure(fp4_outlier_ratio=1.0)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")

    qx = linear_input.quantize(x)
    qw = linear_weight.quantize(w)
    y = linear_input.qgemm(
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

    ref = (x.float() @ w.float().t()).to(torch.bfloat16)
    assert tuple(y.shape) == (128, 128)
    assert int(qx.outlier_values.numel()) == x.numel()
    torch.testing.assert_close(y, ref, rtol=0.0, atol=4.0e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 requires CUDA")
def test_backward_gemms_route_to_default_nvfp4():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer GPU")

    _configure(fp4_outlier_ratio=0.01)
    torch.manual_seed(456)
    torch.cuda.manual_seed(456)

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    linear_input = nvfp4_outlier_quantizer_factory("linear_input")
    linear_weight = nvfp4_outlier_quantizer_factory("linear_weight")
    linear_grad_output = nvfp4_outlier_quantizer_factory("linear_grad_output")

    qx = linear_input.quantize(x)
    qw = linear_weight.quantize(w)
    qdy = linear_grad_output.quantize(dy)
    params = quantization.MMParams(out_dtype=torch.bfloat16, use_split_accumulator=True)

    dx = linear_grad_output.qgemm(
        qdy.data,
        qw.data,
        params,
        torch.bfloat16,
        qdy.scale,
        qw.scale,
        gemm_type=quantization.GEMMType.DGRAD,
        qresult_x=qdy,
        qresult_w=qw,
    )
    dw = linear_grad_output.qgemm(
        qdy.data_t,
        qx.data_t,
        params,
        torch.bfloat16,
        qdy.scale_t,
        qx.scale_t,
        gemm_type=quantization.GEMMType.WGRAD,
        qresult_x=qdy,
        qresult_w=qx,
    )

    assert tuple(dx.shape) == (128, 128)
    assert tuple(dw.shape) == (128, 128)
    assert torch.isfinite(dx.float()).all()
    assert torch.isfinite(dw.float()).all()
