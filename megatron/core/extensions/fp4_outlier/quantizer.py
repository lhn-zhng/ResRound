"""Custom NVFP4 quantizer for FPROP input outlier split."""

from __future__ import annotations

from typing import Optional

import torch
from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.custom_recipes import utils as te_utils
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from .config import get_config
from .fast_fprop import try_fast_select_quantize
from .gemm import run_qgemm
from .runtime import log_rank0_once
from .split import has_values, reshape_to_2d, resolve_outlier_ratio, split_outliers
from .storage import pack_quantized, store_default_variant
from .tensor import OutlierAwareNVFP4TensorRef


_QRESULT_STATE_ATTRS = (
    "data",
    "scale",
    "data_t",
    "scale_t",
    "global_amax_row",
    "global_amax_col",
    "default_data",
    "default_scale",
    "default_data_t",
    "default_scale_t",
    "default_global_amax_row",
    "default_global_amax_col",
    "dtype",
    "quant_dtype",
    "original_shape",
    "outlier_rows",
    "outlier_cols",
    "outlier_values",
    "outlier_flat_indices",
    "outlier_row_offsets",
    "dense_ref",
    "dense_main",
    "weight_rounding_data_t",
    "weight_rounding_scale_t",
    "weight_rounding_global_amax_col",
    "weight_update_generation",
    "source_requires_grad",
    "forward_grad_enabled",
    "configured_outlier_ratio",
    "effective_outlier_ratio",
    "outlier_heaviness",
)


class OutlierAwareNVFP4QuantizerRef(NVFP4QuantizerRef):
    """NVFP4 custom quantizer that only changes FPROP linear_input."""

    def __init__(
        self,
        *,
        apply_outlier_split: bool = False,
        store_dense_ref: bool = False,
        store_dense_main: bool = False,
        outlier_ratio: float = 0.01,
        outlier_selection_method: str = "topk",
        enable_sparse_all_gather: bool = False,
        role: str = "unknown",
        **kwargs,
    ):
        super().__init__(**kwargs)
        cfg = get_config()
        self.apply_outlier_split = apply_outlier_split
        self.store_dense_ref = store_dense_ref
        self.store_dense_main = store_dense_main
        self.outlier_ratio = outlier_ratio
        self.outlier_selection_method = outlier_selection_method
        self.enable_sparse_all_gather = enable_sparse_all_gather
        self.needs_runtime_main_scale = False
        self.role = role
        self._weight_update_generation = 0

        self._main_nvfp4_quantizer = NVFP4Quantizer(
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            with_rht=cfg.main_quantizer_rht,
            with_post_rht_amax=cfg.main_quantizer_rht,
            with_2d_quantization=(role == "linear_weight"),
            stochastic_rounding=(cfg.input_stochastic_rounding and role == "linear_input"),
            with_random_sign_mask=True,
        )
        self._fast_nvfp4_quantizer = self._main_nvfp4_quantizer
        self._default_nvfp4_quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=True,
            with_rht=(role != "linear_weight"),
            with_post_rht_amax=(role != "linear_weight"),
            with_2d_quantization=(role == "linear_weight"),
            stochastic_rounding=(role == "linear_grad_output"),
            with_random_sign_mask=True,
        )
        self._sync_internal_quantizers()

    def _sync_internal_quantizers(self) -> None:
        for quantizer in (self._main_nvfp4_quantizer, self._default_nvfp4_quantizer):
            quantizer.internal = self.internal
            quantizer.with_amax_reduction = getattr(self, "with_amax_reduction", False)
            quantizer.amax_reduction_group = getattr(self, "amax_reduction_group", None)

    def set_usage(
        self,
        *,
        rowwise: Optional[bool] = None,
        columnwise: Optional[bool] = None,
    ) -> None:
        super().set_usage(rowwise=rowwise, columnwise=columnwise)
        self._sync_internal_quantizers()

    def _quantize_default(self, tensor: torch.Tensor):
        quantizer = self._default_nvfp4_quantizer.copy()
        rowwise, columnwise = self._default_variant_usage()
        quantizer.set_usage(rowwise=rowwise, columnwise=columnwise)
        quantizer.internal = self.internal
        quantizer.with_amax_reduction = getattr(self, "with_amax_reduction", False)
        quantizer.amax_reduction_group = getattr(self, "amax_reduction_group", None)
        return quantizer(tensor)

    def _default_variant_usage(self) -> tuple[bool, bool]:
        cfg = get_config()
        if self.role == "linear_input":
            return not cfg.enable_fprop, True
        return True, True

    def _quantize_main(self, tensor: torch.Tensor):
        quantizer = self._main_nvfp4_quantizer.copy()
        if self.role == "linear_input" and self.apply_outlier_split:
            quantizer.set_usage(rowwise=True, columnwise=False)
        else:
            quantizer.set_usage(rowwise=self.rowwise_usage, columnwise=self.columnwise_usage)
        quantizer.internal = self.internal
        quantizer.with_amax_reduction = getattr(self, "with_amax_reduction", False)
        quantizer.amax_reduction_group = getattr(self, "amax_reduction_group", None)
        return quantizer(tensor)

    def _can_reuse_fast_columnwise_as_default(self, fast_qtensor) -> bool:
        if fast_qtensor is None or self.role != "linear_input":
            return False
        if getattr(fast_qtensor, "columnwise_source", None) not in {
            "direct",
            "te_default_direct",
        }:
            return False
        return (
            getattr(fast_qtensor, "_columnwise_data", None) is not None
            and getattr(fast_qtensor, "_columnwise_scale_inv", None) is not None
            and getattr(fast_qtensor, "_amax_columnwise", None) is not None
        )

    def _store_fast_columnwise_as_default(
        self,
        qresult: OutlierAwareNVFP4TensorRef,
        fast_qtensor,
    ) -> None:
        qresult.default_data = None
        qresult.default_scale = None
        qresult.default_global_amax_row = None
        qresult.default_data_t = fast_qtensor._columnwise_data
        qresult.default_scale_t = fast_qtensor._columnwise_scale_inv
        qresult.default_global_amax_col = fast_qtensor._amax_columnwise

        default_quantizer = self._default_nvfp4_quantizer.copy()
        default_quantizer.set_usage(rowwise=False, columnwise=True)
        default_quantizer.internal = self.internal
        default_quantizer.with_amax_reduction = getattr(self, "with_amax_reduction", False)
        default_quantizer.amax_reduction_group = getattr(self, "amax_reduction_group", None)
        qresult._default_quantizer = default_quantizer

    def quantize(self, tensor: torch.Tensor, **kwargs) -> OutlierAwareNVFP4TensorRef:
        assert tensor.dtype in te_utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."
        del kwargs

        original_shape = tensor.shape
        tensor_2d = reshape_to_2d(tensor)
        cfg = get_config()
        self._sync_internal_quantizers()
        ratio = resolve_outlier_ratio(
            tensor_2d,
            base_ratio=self.outlier_ratio,
            adaptive_enabled=cfg.adaptive_outlier_ratio and self.apply_outlier_split,
            adaptive_min_ratio=cfg.adaptive_outlier_min_ratio,
            adaptive_max_ratio=cfg.adaptive_outlier_max_ratio,
            adaptive_reference_heaviness=cfg.adaptive_outlier_reference_heaviness,
        )

        fast_qtensor = None
        if self.apply_outlier_split and cfg.enable_fprop and cfg.enable_fast_fprop:
            fast_qtensor = try_fast_select_quantize(
                tensor_2d,
                owner=self,
                ratio=ratio,
                selection_method=self.outlier_selection_method,
                store_dense_main=self.store_dense_main,
            )
        reuse_fast_columnwise_as_default = self._can_reuse_fast_columnwise_as_default(
            fast_qtensor
        )
        default_qtensor = None if reuse_fast_columnwise_as_default else self._quantize_default(
            tensor_2d
        )

        if fast_qtensor is not None:
            main = fast_qtensor.dense_main if self.store_dense_main else None
            outlier_rows = fast_qtensor.outlier_rows
            outlier_cols = fast_qtensor.outlier_cols
            outlier_values = fast_qtensor.outlier_values
            outlier_flat_indices = fast_qtensor.outlier_flat_indices
            outlier_row_offsets = fast_qtensor.outlier_row_offsets
            main_qtensor = fast_qtensor
        elif self.apply_outlier_split:
            (
                main,
                outlier_rows,
                outlier_cols,
                outlier_values,
                outlier_flat_indices,
                outlier_row_offsets,
            ) = split_outliers(
                tensor_2d,
                outlier_ratio=self.outlier_ratio,
                selection_method=self.outlier_selection_method,
                effective_outlier_ratio=ratio.effective,
            )
        else:
            main = tensor_2d
            outlier_rows = None
            outlier_cols = None
            outlier_values = None
            outlier_flat_indices = None
            outlier_row_offsets = None

        if fast_qtensor is None:
            main_qtensor = self._quantize_main(main) if self.apply_outlier_split else default_qtensor
        qresult = pack_quantized(
            main_qtensor,
            original_shape=original_shape,
            owner=self,
            dtype=tensor_2d.dtype,
            outlier_rows=outlier_rows,
            outlier_cols=outlier_cols,
            outlier_values=outlier_values,
            outlier_flat_indices=outlier_flat_indices,
            outlier_row_offsets=outlier_row_offsets,
            dense_ref=tensor_2d if self.store_dense_ref else None,
            dense_main=main if self.store_dense_main else None,
        )
        qresult.source_requires_grad = bool(tensor.requires_grad)
        forward_grad_enabled = getattr(
            self,
            "_fp4_forward_grad_enabled",
            None,
        )
        qresult.forward_grad_enabled = (
            bool(forward_grad_enabled)
            if forward_grad_enabled is not None
            else None
        )
        if self.role == "linear_weight":
            self._weight_update_generation += 1
            qresult.weight_update_generation = self._weight_update_generation
        qresult.configured_outlier_ratio = ratio.configured if self.apply_outlier_split else None
        qresult.effective_outlier_ratio = (
            fast_qtensor.effective_ratio
            if fast_qtensor is not None
            else (ratio.effective if self.apply_outlier_split else None)
        )
        qresult.outlier_heaviness = (
            fast_qtensor.heaviness
            if fast_qtensor is not None
            else (ratio.heaviness if self.apply_outlier_split else None)
        )
        if fast_qtensor is not None:
            qresult.fast_fprop_select_quant_backend = fast_qtensor.backend
            qresult.fast_fprop_selected_nnz = fast_qtensor.selected_nnz
            qresult.fast_fprop_actual_ratio = fast_qtensor.actual_ratio
            qresult.fast_fprop_overflow = fast_qtensor.overflow
            qresult.fast_fprop_requested_capacity = fast_qtensor.requested_capacity
            qresult.fast_fprop_payload_capacity = fast_qtensor.payload_capacity
            if getattr(fast_qtensor, "sparse_policy_hint", None) is not None:
                qresult._fast_fprop_sparse_policy_hint = fast_qtensor.sparse_policy_hint
            if getattr(fast_qtensor, "active_rows", None) is not None:
                qresult._fast_fprop_active_rows = fast_qtensor.active_rows
                qresult._fast_fprop_active_rows_num_rows = fast_qtensor.active_rows_num_rows
                qresult._fast_fprop_active_row_count = fast_qtensor.active_row_count
            if getattr(fast_qtensor, "full_capacity_flat_indices", None) is not None:
                qresult._outlier_payload_full_capacity = True
                qresult._outlier_full_capacity_flat_indices = (
                    fast_qtensor.full_capacity_flat_indices
                )
                qresult._outlier_full_capacity_values = fast_qtensor.full_capacity_values
                qresult._outlier_local_count_hint = fast_qtensor.local_count_hint
            if getattr(fast_qtensor, "full_capacity_cols", None) is not None:
                qresult._fast_fprop_full_capacity_cols = fast_qtensor.full_capacity_cols
        if default_qtensor is not None:
            store_default_variant(qresult, default_qtensor)
        else:
            self._store_fast_columnwise_as_default(qresult, fast_qtensor)
            log_rank0_once(
                "fast_fprop:default:reuse_columnwise",
                (
                    "FP4 fast FPROP reusing r207 direct columnwise as default: "
                    "role=%s, source=%s, shape=%s"
                ),
                self.role,
                getattr(fast_qtensor, "columnwise_source", None),
                tuple(tensor_2d.shape),
            )

        if (
            self.role == "linear_input"
            and qresult.data_t is None
            and qresult.default_data_t is not None
        ):
            qresult.data_t = qresult.default_data_t
            qresult.scale_t = qresult.default_scale_t
            qresult.global_amax_col = qresult.default_global_amax_col

        if fast_qtensor is not None and int(fast_qtensor.selected_nnz) >= 0:
            outlier_actual_ratio = float(fast_qtensor.selected_nnz) / max(
                1,
                int(tensor_2d.numel()),
            )
        elif fast_qtensor is not None:
            outlier_actual_ratio = float("nan")
        else:
            outlier_count = 0 if not has_values(outlier_values) else int(outlier_values.numel())
            outlier_actual_ratio = float(outlier_count) / max(1, int(tensor_2d.numel()))
        log_rank0_once(
            f"quantize:{self.role}",
            (
                "FP4 FPROP input-outlier quantizer active: role=%s, split=%s, "
                "method=%s, adaptive=%s, configured_ratio=%.6f, "
                "effective_ratio=%.6f, actual_ratio=%.6f, shape=%s"
            ),
            self.role,
            self.apply_outlier_split,
            self.outlier_selection_method,
            ratio.adaptive and self.apply_outlier_split,
            self.outlier_ratio,
            ratio.effective if self.apply_outlier_split else self.outlier_ratio,
            outlier_actual_ratio,
            tuple(tensor_2d.shape),
        )
        return qresult

    def update_quantized(
        self,
        src: torch.Tensor,
        dst,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ):
        if noop_flag is not None and noop_flag.item() != 0:
            return dst
        src_qresult = self.quantize(src)
        for attr in _QRESULT_STATE_ATTRS:
            setattr(dst, attr, getattr(src_qresult, attr))
        if self.role == "linear_weight":
            setattr(dst, "_fast_fprop_weight_t_state", None)
            setattr(dst, "_weight_rounding_static_state", None)
            setattr(dst, "_weight_rounding_payload_state", None)
            setattr(dst, "_weight_rounding_column_quantizer", None)
            setattr(self, "_fast_fprop_weight_t_state", None)
        return dst

    @torch.compiler.disable()
    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: OutlierAwareNVFP4TensorRef | None = None,
        qresult_w: OutlierAwareNVFP4TensorRef | None = None,
    ) -> torch.Tensor:
        if qresult_x is None or qresult_w is None:
            return super().qgemm(
                qx=qx,
                qw=qw,
                m_params=m_params,
                out_dtype=out_dtype,
                sx=sx,
                sw=sw,
                bias=bias,
                out=out,
                accumulate=accumulate,
                gemm_type=gemm_type,
                qresult_x=qresult_x,
                qresult_w=qresult_w,
            )

        cfg = get_config()
        if gemm_type == quantization.GEMMType.FPROP and cfg.enable_fprop:
            log_rank0_once(
                f"qgemm:{gemm_type.value}:custom",
                "FP4 FPROP input-outlier qgemm active: gemm_type=%s",
                gemm_type.value,
            )
        else:
            log_rank0_once(
                f"qgemm:{gemm_type.value}:default",
                "FP4 FPROP input-outlier recipe using default NVFP4 qgemm: gemm_type=%s",
                gemm_type.value,
            )
        return run_qgemm(
            qresult_x=qresult_x,
            qresult_w=qresult_w,
            m_params=m_params,
            out_dtype=out_dtype,
            bias=bias,
            out=out,
            accumulate=accumulate,
            gemm_type=gemm_type,
        )
