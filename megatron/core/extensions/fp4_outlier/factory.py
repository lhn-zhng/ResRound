"""Transformer Engine CustomRecipe factory for FP4 FPROP input outliers."""

from __future__ import annotations

import os

from transformer_engine.pytorch.custom_recipes import utils as te_utils
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from .config import get_config
from .quantizer import OutlierAwareNVFP4QuantizerRef


_ORIGINAL_NVFP4_QUANTIZER_CALL = NVFP4Quantizer.__call__
_ALIGNMENT_GRAD_QUANTIZER_ATTR = "_fp4_weight_rounding_grad_alignment_capture"


def _alignment_nvfp4_quantizer_call(self, tensor, *args, **kwargs):
    if getattr(self, _ALIGNMENT_GRAD_QUANTIZER_ATTR, False):
        from .gemm import consume_weight_rounding_grad_alignment

        consume_weight_rounding_grad_alignment(tensor)
    return _ORIGINAL_NVFP4_QUANTIZER_CALL(self, tensor, *args, **kwargs)


def _enable_grad_alignment_capture(quantizer: NVFP4Quantizer) -> None:
    if NVFP4Quantizer.__call__ is not _alignment_nvfp4_quantizer_call:
        NVFP4Quantizer.__call__ = _alignment_nvfp4_quantizer_call
    # Keep the marker on the live object. A process-global set of Python
    # ``id`` values is unsafe because evaluation can destroy temporary
    # quantizers and a later weight quantizer may reuse the same integer ID.
    setattr(quantizer, _ALIGNMENT_GRAD_QUANTIZER_ATTR, True)


def _grad_alignment_capture_enabled() -> bool:
    return (
        os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT", "0") == "1"
        and (
            os.environ.get(
                "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_CAPTURE_GRAD",
                "0",
            )
            == "1"
            or os.environ.get(
                "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_PREDICT",
                "0",
            )
            == "1"
        )
    )


def nvfp4_outlier_quantizer_factory(role: str):
    """Create role-specific quantizers for the FP4 FPROP input-outlier recipe."""
    cfg = get_config()
    if role == "linear_grad_output":
        quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            stochastic_rounding=True,
            with_random_sign_mask=True,
        )
        if _grad_alignment_capture_enabled():
            _enable_grad_alignment_capture(quantizer)
        return quantizer

    common_kwargs = dict(
        dtype=te_utils.Fp4Formats.E2M1,
        pow_2_scales=False,
        with_random_sign_mask=True,
    )
    role_kwargs = {
        "linear_input": dict(
            rowwise=True,
            columnwise=True,
            quant_tile_shape=(1, 16),
            with_rht=False,
            apply_outlier_split=cfg.enable_fprop,
            store_dense_ref=False,
            store_dense_main=(
                cfg.store_input_dense_main
                or cfg.weight_rounding_combined_audit
                or cfg.weight_rounding_joint_objective
            ),
            outlier_ratio=cfg.outlier_ratio if cfg.enable_fprop else 0.0,
            outlier_selection_method=cfg.outlier_selection_method,
            enable_sparse_all_gather=cfg.enable_nvfp4_a1_a2_all_gather,
        ),
        "linear_weight": dict(
            rowwise=True,
            columnwise=True,
            quant_tile_shape=(16, 16),
            with_rht=False,
            apply_outlier_split=False,
            store_dense_ref=cfg.enable_fprop,
            store_dense_main=False,
            outlier_ratio=0.0,
            outlier_selection_method=cfg.outlier_selection_method,
        ),
    }
    kwargs = role_kwargs.get(role)
    if kwargs is None:
        return None
    return OutlierAwareNVFP4QuantizerRef(role=role, **common_kwargs, **kwargs)
