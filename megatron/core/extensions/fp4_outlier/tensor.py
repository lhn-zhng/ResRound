"""Tensor wrapper carrying NVFP4 payload plus sparse input outliers."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import torch
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4TensorRef


@dataclasses.dataclass
class OutlierAwareNVFP4TensorRef(NVFP4TensorRef):
    """NVFP4 reference tensor with optional sparse outlier and dense reference payloads."""

    outlier_rows: Optional[torch.Tensor] = None
    outlier_cols: Optional[torch.Tensor] = None
    outlier_values: Optional[torch.Tensor] = None
    outlier_flat_indices: Optional[torch.Tensor] = None
    outlier_row_offsets: Optional[torch.Tensor] = None
    configured_outlier_ratio: Optional[float] = None
    effective_outlier_ratio: Optional[float] = None
    outlier_heaviness: Optional[float] = None
    default_data: Optional[torch.Tensor] = None
    default_scale: Optional[torch.Tensor] = None
    default_data_t: Optional[torch.Tensor] = None
    default_scale_t: Optional[torch.Tensor] = None
    default_global_amax_row: Optional[torch.Tensor] = None
    default_global_amax_col: Optional[torch.Tensor] = None
    dense_ref: Optional[torch.Tensor] = None
    dense_main: Optional[torch.Tensor] = None
    weight_rounding_data_t: Optional[torch.Tensor] = None
    weight_rounding_scale_t: Optional[torch.Tensor] = None
    weight_rounding_global_amax_col: Optional[torch.Tensor] = None
    weight_update_generation: Optional[int] = None
    source_requires_grad: bool = False
    forward_grad_enabled: Optional[bool] = None

    def logical_2d_shape(self) -> tuple[int, int]:
        if self.original_shape is None:
            raise ValueError("Missing original_shape on outlier tensor.")
        shape = tuple(int(dim) for dim in self.original_shape)
        if len(shape) > 2:
            return (math.prod(shape[:-1]), shape[-1])
        if len(shape) == 2:
            return shape
        if len(shape) == 1:
            return (1, shape[0])
        raise ValueError(f"Unsupported original_shape for outlier tensor: {shape}.")

    def outlier_valid_count(self) -> int:
        if self.outlier_values is None:
            return 0
        selected = getattr(self, "fast_fprop_selected_nnz", None)
        if selected is not None and int(selected) >= 0:
            count = int(selected)
        else:
            count_hint = getattr(self, "_outlier_local_count_hint", None)
            if (
                isinstance(count_hint, torch.Tensor)
                and int(count_hint.numel()) == 1
                and count_hint.device == self.outlier_values.device
            ):
                count = int(count_hint.reshape(-1)[0].detach().cpu().item())
            elif self.outlier_row_offsets is not None and int(self.outlier_row_offsets.numel()) > 0:
                count = int(self.outlier_row_offsets[-1].detach().cpu().item())
            else:
                count = int(self.outlier_values.numel())
        if count < 0 or count > int(self.outlier_values.numel()):
            raise ValueError(
                "Invalid outlier payload count: "
                f"count={count}, values_numel={int(self.outlier_values.numel())}."
            )
        return count

    def valid_outlier_values(self) -> Optional[torch.Tensor]:
        if self.outlier_values is None:
            return None
        return self.outlier_values.narrow(0, 0, self.outlier_valid_count())

    def ensure_outlier_coo(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.outlier_rows is not None and self.outlier_cols is not None:
            return self.outlier_rows, self.outlier_cols
        if self.outlier_values is None or int(self.outlier_values.numel()) == 0:
            return self.outlier_rows, self.outlier_cols
        if self.outlier_flat_indices is None:
            return self.outlier_rows, self.outlier_cols

        _, cols = self.logical_2d_shape()
        count = self.outlier_valid_count()
        flat = self.outlier_flat_indices.narrow(0, 0, count).to(torch.int64)
        self.outlier_rows = torch.div(flat, cols, rounding_mode="floor").to(torch.int32).contiguous()
        self.outlier_cols = (flat % cols).to(torch.int32).contiguous()
        return self.outlier_rows, self.outlier_cols

    def _overlay_outliers(self, dense: torch.Tensor) -> torch.Tensor:
        values = self.valid_outlier_values()
        if values is None or int(values.numel()) == 0:
            return dense
        rows, cols = self.ensure_outlier_coo()
        if rows is None or cols is None:
            raise ValueError("Outlier values are present but COO indices are unavailable.")
        dense = dense.clone()
        dense[rows.long(), cols.long()] = values.to(dtype=dense.dtype)
        return dense

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "OutlierAwareNVFP4TensorRef"]:
        saved_default_data_t = (
            self.weight_rounding_data_t
            if self.weight_rounding_data_t is not None
            else self.default_data_t
        )
        saved_default_scale_t = (
            self.weight_rounding_scale_t
            if self.weight_rounding_scale_t is not None
            else self.default_scale_t
        )
        saved_default_amax_col = (
            self.weight_rounding_global_amax_col
            if self.weight_rounding_global_amax_col is not None
            else self.default_global_amax_col
        )
        tensors = [
            self.data,
            self.data_t,
            self.scale,
            self.scale_t,
            self.global_amax_row,
            self.global_amax_col,
            self.default_data,
            self.default_scale,
            saved_default_data_t,
            saved_default_scale_t,
            self.default_global_amax_row,
            saved_default_amax_col,
            self.outlier_rows,
            self.outlier_cols,
            self.outlier_values,
            self.outlier_flat_indices,
            self.outlier_row_offsets,
            self.dense_ref,
            self.dense_main,
            self.weight_rounding_data_t,
            self.weight_rounding_scale_t,
            self.weight_rounding_global_amax_col,
        ]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        self.global_amax_row = None
        self.global_amax_col = None
        self.default_data = None
        self.default_scale = None
        self.default_data_t = None
        self.default_scale_t = None
        self.default_global_amax_row = None
        self.default_global_amax_col = None
        self.outlier_rows = None
        self.outlier_cols = None
        self.outlier_values = None
        self.outlier_flat_indices = None
        self.outlier_row_offsets = None
        self.dense_ref = None
        self.dense_main = None
        self.weight_rounding_data_t = None
        self.weight_rounding_scale_t = None
        self.weight_rounding_global_amax_col = None
        return tensors, self

    def restore_from_saved(
        self,
        tensors: list[Optional[torch.Tensor]],
    ) -> list[Optional[torch.Tensor]]:
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        self.global_amax_row = tensors[4]
        self.global_amax_col = tensors[5]
        self.default_data = tensors[6]
        self.default_scale = tensors[7]
        self.default_data_t = tensors[8]
        self.default_scale_t = tensors[9]
        self.default_global_amax_row = tensors[10]
        self.default_global_amax_col = tensors[11]
        self.outlier_rows = tensors[12]
        self.outlier_cols = tensors[13]
        self.outlier_values = tensors[14]
        self.outlier_flat_indices = tensors[15]
        self.outlier_row_offsets = tensors[16]
        self.dense_ref = tensors[17]
        self.dense_main = tensors[18]
        self.weight_rounding_data_t = tensors[19]
        self.weight_rounding_scale_t = tensors[20]
        self.weight_rounding_global_amax_col = tensors[21]
        return tensors[22:]

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype if dtype is not None else self.dtype
        if target_dtype is None:
            target_dtype = torch.bfloat16

        dense = self.dense_ref
        if dense is None and self.dense_main is not None:
            dense = self.dense_main
            dense = self._overlay_outliers(dense)

        if dense is None:
            from .storage import make_primary_nvfp4_storage

            dense = make_primary_nvfp4_storage(
                self,
                use_rowwise=True,
                use_columnwise=False,
            ).dequantize(dtype=target_dtype)
            if dense.ndim != 2:
                dense = dense.reshape(-1, dense.shape[-1])
            dense = self._overlay_outliers(dense)

        if dense.dtype != target_dtype:
            dense = dense.to(dtype=target_dtype)
        if self.original_shape is not None and tuple(dense.shape) != tuple(self.original_shape):
            shape = tuple(int(dim) for dim in self.original_shape)
            if int(dense.numel()) == int(math.prod(shape)):
                dense = dense.reshape(shape)
        return dense
