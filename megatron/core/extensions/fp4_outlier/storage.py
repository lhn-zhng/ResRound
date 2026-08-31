"""Packing helpers for TE NVFP4 tensors."""

from __future__ import annotations

from typing import Optional

import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage

from .tensor import OutlierAwareNVFP4TensorRef


def pack_quantized(
    qtensor,
    *,
    original_shape: torch.Size,
    owner,
    dtype: torch.dtype,
    outlier_rows: Optional[torch.Tensor],
    outlier_cols: Optional[torch.Tensor],
    outlier_values: Optional[torch.Tensor],
    outlier_flat_indices: Optional[torch.Tensor],
    outlier_row_offsets: Optional[torch.Tensor],
    dense_ref: Optional[torch.Tensor],
    dense_main: Optional[torch.Tensor],
) -> OutlierAwareNVFP4TensorRef:
    device = getattr(qtensor, "device", None)
    if device is None:
        for attr in ("_rowwise_data", "_columnwise_data"):
            data = getattr(qtensor, attr, None)
            if data is not None:
                device = data.device
                break
    if device is None:
        raise ValueError("Failed to infer device from NVFP4 quantized tensor.")

    qresult = OutlierAwareNVFP4TensorRef(
        data=qtensor._rowwise_data,
        scale=qtensor._rowwise_scale_inv,
        data_t=qtensor._columnwise_data,
        scale_t=qtensor._columnwise_scale_inv,
        global_amax_row=qtensor._amax_rowwise,
        global_amax_col=qtensor._amax_columnwise,
        dtype=dtype,
        device=device,
        quant_dtype=qtensor._fp4_dtype,
        original_shape=original_shape,
        _quantizer=owner,
        outlier_rows=outlier_rows,
        outlier_cols=outlier_cols,
        outlier_values=outlier_values,
        outlier_flat_indices=outlier_flat_indices,
        outlier_row_offsets=outlier_row_offsets,
        dense_ref=dense_ref,
        dense_main=dense_main,
    )
    qresult._primary_quantizer = getattr(qtensor, "_quantizer", None)
    return qresult


def store_default_variant(qresult: OutlierAwareNVFP4TensorRef, qtensor) -> None:
    qresult.default_data = qtensor._rowwise_data
    qresult.default_scale = qtensor._rowwise_scale_inv
    qresult.default_data_t = qtensor._columnwise_data
    qresult.default_scale_t = qtensor._columnwise_scale_inv
    qresult.default_global_amax_row = qtensor._amax_rowwise
    qresult.default_global_amax_col = qtensor._amax_columnwise
    qresult._default_quantizer = getattr(qtensor, "_quantizer", None)


def _make_variant_nvfp4_storage(
    qresult: OutlierAwareNVFP4TensorRef,
    *,
    variant: str,
    quantizer: NVFP4Quantizer,
    use_rowwise: bool,
    use_columnwise: bool,
    rowwise_data_override: Optional[torch.Tensor] = None,
    columnwise_data_override: Optional[torch.Tensor] = None,
    columnwise_scale_override: Optional[torch.Tensor] = None,
    columnwise_amax_override: Optional[torch.Tensor] = None,
    quantizer_override: Optional[NVFP4Quantizer] = None,
) -> NVFP4TensorStorage:
    prefix = "" if variant == "primary" else f"{variant}_"

    def _get(name: str):
        return getattr(qresult, f"{prefix}{name}")

    rowwise_scale_inv = _get("scale") if use_rowwise else None
    columnwise_scale_inv = (
        columnwise_scale_override
        if use_columnwise and columnwise_scale_override is not None
        else (_get("scale_t") if use_columnwise else None)
    )
    if rowwise_scale_inv is None and columnwise_scale_inv is None:
        raise ValueError(f"Missing {variant} NVFP4 variant on qresult.")

    gathered_quantizer_attr = (
        "_gathered_primary_quantizer" if variant == "primary" else f"_gathered_{variant}_quantizer"
    )
    local_quantizer_attr = "_primary_quantizer" if variant == "primary" else f"_{variant}_quantizer"
    storage_quantizer = quantizer_override
    if storage_quantizer is None:
        storage_quantizer = getattr(qresult, gathered_quantizer_attr, None)
    if storage_quantizer is None:
        storage_quantizer = getattr(qresult, local_quantizer_attr, None)
    if storage_quantizer is None:
        storage_quantizer = quantizer
    storage_quantizer = storage_quantizer.copy()
    storage_quantizer.set_usage(rowwise=use_rowwise, columnwise=use_columnwise)

    return NVFP4TensorStorage(
        rowwise_data=(
            rowwise_data_override
            if use_rowwise and rowwise_data_override is not None
            else (_get("data") if use_rowwise else None)
        ),
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=(
            columnwise_data_override
            if use_columnwise and columnwise_data_override is not None
            else (_get("data_t") if use_columnwise else None)
        ),
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=_get("global_amax_row") if use_rowwise else None,
        amax_columnwise=(
            columnwise_amax_override
            if use_columnwise and columnwise_amax_override is not None
            else (_get("global_amax_col") if use_columnwise else None)
        ),
        fp4_dtype=storage_quantizer.dtype,
        quantizer=storage_quantizer,
    )


def make_primary_nvfp4_storage(
    qresult: OutlierAwareNVFP4TensorRef,
    *,
    use_rowwise: bool,
    use_columnwise: bool,
    rowwise_data_override: Optional[torch.Tensor] = None,
) -> NVFP4TensorStorage:
    quantizer = getattr(qresult._quantizer, "_main_nvfp4_quantizer", None)
    if quantizer is None:
        raise ValueError("Missing main NVFP4 quantizer on qresult.")
    return _make_variant_nvfp4_storage(
        qresult,
        variant="primary",
        quantizer=quantizer,
        use_rowwise=use_rowwise,
        use_columnwise=use_columnwise,
        rowwise_data_override=rowwise_data_override,
    )


def make_default_nvfp4_storage(
    qresult: OutlierAwareNVFP4TensorRef,
    *,
    use_rowwise: bool,
    use_columnwise: bool,
    columnwise_data_override: Optional[torch.Tensor] = None,
    columnwise_scale_override: Optional[torch.Tensor] = None,
    columnwise_amax_override: Optional[torch.Tensor] = None,
    quantizer_override: Optional[NVFP4Quantizer] = None,
) -> NVFP4TensorStorage:
    quantizer = getattr(qresult._quantizer, "_default_nvfp4_quantizer", None)
    if quantizer is None:
        raise ValueError("Missing default NVFP4 quantizer on qresult.")
    return _make_variant_nvfp4_storage(
        qresult,
        variant="default",
        quantizer=quantizer,
        use_rowwise=use_rowwise,
        use_columnwise=use_columnwise,
        columnwise_data_override=columnwise_data_override,
        columnwise_scale_override=columnwise_scale_override,
        columnwise_amax_override=columnwise_amax_override,
        quantizer_override=quantizer_override,
    )
