"""Runtime config for the FP4 FPROP input-outlier recipe."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class FP4FpropInputOutlierConfig:
    """Small config surface for input-side sparse outlier FPROP."""

    outlier_ratio: float = 0.01
    outlier_selection_method: str = "topk"
    adaptive_outlier_ratio: bool = False
    adaptive_outlier_min_ratio: float = 0.0
    adaptive_outlier_max_ratio: float = 0.01
    adaptive_outlier_reference_heaviness: float = 15.0
    enable_fprop: bool = True
    enable_fast_fprop: bool = False
    enable_dgrad: bool = False
    enable_wgrad: bool = False
    enable_nvfp4_a1_a2_all_gather: bool = False
    enable_weight_rounding: bool = False
    weight_rounding_group_size: int = 64
    weight_rounding_rounds_per_group: int = 1
    weight_rounding_selection_tokens: int = 2048
    weight_rounding_stratified_sampling: bool = False
    weight_rounding_stratified_batch_size: int = 0
    weight_rounding_crossfit_audit: bool = False
    weight_rounding_combined_audit: bool = False
    weight_rounding_joint_objective: bool = False
    weight_rounding_audit_tokens: int = 512
    weight_rounding_audit_max_regression_fraction: float = 0.5
    weight_rounding_audit_min_relative_gain: float = 0.0
    weight_rounding_offdiag_shrink: float = 1.0
    weight_rounding_expansion_only: bool = True
    weight_rounding_min_expansion_ratio: float = 1.0
    weight_rounding_max_expansion_ratio: float = 0.0
    weight_rounding_layer_start: int = 0
    weight_rounding_layer_end: int = -1
    weight_rounding_qkv_layer_end: int = -1
    weight_rounding_proj_layer_end: int = -1
    weight_rounding_fc1_layer_end: int = -1
    weight_rounding_reuse_generation_payload: bool = True
    weight_rounding_dgrad_consistency: bool = False
    store_input_dense_main: bool = False
    main_quantizer_rht: bool = False
    input_stochastic_rounding: bool = False


_CONFIG = FP4FpropInputOutlierConfig()


def get_config() -> FP4FpropInputOutlierConfig:
    return _CONFIG


def configure_from_transformer_config(config) -> None:
    """Hook called by Megatron before creating the TE CustomRecipe."""
    global _CONFIG

    ratio = float(getattr(config, "fp4_outlier_ratio", 0.01))
    method = str(getattr(config, "fp4_outlier_selection_method", "topk")).lower()
    _CONFIG = FP4FpropInputOutlierConfig(
        outlier_ratio=ratio,
        outlier_selection_method=method,
        adaptive_outlier_ratio=bool(getattr(config, "fp4_outlier_adaptive_ratio", False)),
        adaptive_outlier_min_ratio=float(
            getattr(config, "fp4_outlier_adaptive_min_ratio", 0.0)
        ),
        adaptive_outlier_max_ratio=float(
            getattr(config, "fp4_outlier_adaptive_max_ratio", 0.01)
        ),
        adaptive_outlier_reference_heaviness=float(
            getattr(config, "fp4_outlier_adaptive_reference_heaviness", 15.0)
        ),
        enable_fprop=bool(getattr(config, "fp4_outlier_enable_fprop", True)),
        enable_fast_fprop=bool(getattr(config, "fp4_outlier_enable_fast_fprop", False)),
        enable_dgrad=bool(getattr(config, "fp4_outlier_enable_dgrad", False)),
        enable_wgrad=bool(getattr(config, "fp4_outlier_enable_wgrad", False)),
        enable_nvfp4_a1_a2_all_gather=bool(
            getattr(config, "fp4_outlier_enable_nvfp4_a1_a2_all_gather", False)
        ),
        enable_weight_rounding=bool(
            getattr(config, "fp4_outlier_enable_weight_rounding", False)
        ),
        weight_rounding_group_size=int(
            getattr(config, "fp4_outlier_weight_rounding_group_size", 64)
        ),
        weight_rounding_rounds_per_group=int(
            getattr(config, "fp4_outlier_weight_rounding_rounds_per_group", 1)
        ),
        weight_rounding_selection_tokens=int(
            getattr(config, "fp4_outlier_weight_rounding_selection_tokens", 2048)
        ),
        weight_rounding_stratified_sampling=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_stratified_sampling",
                False,
            )
        ),
        weight_rounding_stratified_batch_size=int(
            getattr(
                config,
                "fp4_outlier_weight_rounding_stratified_batch_size",
                0,
            )
        ),
        weight_rounding_crossfit_audit=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_crossfit_audit",
                False,
            )
        ),
        weight_rounding_combined_audit=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_combined_audit",
                False,
            )
        ),
        weight_rounding_joint_objective=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_joint_objective",
                False,
            )
        ),
        weight_rounding_audit_tokens=int(
            getattr(config, "fp4_outlier_weight_rounding_audit_tokens", 512)
        ),
        weight_rounding_audit_max_regression_fraction=float(
            getattr(
                config,
                "fp4_outlier_weight_rounding_audit_max_regression_fraction",
                0.5,
            )
        ),
        weight_rounding_audit_min_relative_gain=float(
            getattr(
                config,
                "fp4_outlier_weight_rounding_audit_min_relative_gain",
                0.0,
            )
        ),
        weight_rounding_offdiag_shrink=float(
            getattr(config, "fp4_outlier_weight_rounding_offdiag_shrink", 1.0)
        ),
        weight_rounding_expansion_only=bool(
            getattr(config, "fp4_outlier_weight_rounding_expansion_only", True)
        ),
        weight_rounding_min_expansion_ratio=float(
            getattr(
                config,
                "fp4_outlier_weight_rounding_min_expansion_ratio",
                1.0,
            )
        ),
        weight_rounding_max_expansion_ratio=float(
            getattr(
                config,
                "fp4_outlier_weight_rounding_max_expansion_ratio",
                0.0,
            )
        ),
        weight_rounding_layer_start=int(
            getattr(config, "fp4_outlier_weight_rounding_layer_start", 0)
        ),
        weight_rounding_layer_end=int(
            getattr(config, "fp4_outlier_weight_rounding_layer_end", -1)
        ),
        weight_rounding_qkv_layer_end=int(
            getattr(config, "fp4_outlier_weight_rounding_qkv_layer_end", -1)
        ),
        weight_rounding_proj_layer_end=int(
            getattr(config, "fp4_outlier_weight_rounding_proj_layer_end", -1)
        ),
        weight_rounding_fc1_layer_end=int(
            getattr(config, "fp4_outlier_weight_rounding_fc1_layer_end", -1)
        ),
        weight_rounding_reuse_generation_payload=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_reuse_generation_payload",
                True,
            )
        ),
        weight_rounding_dgrad_consistency=bool(
            getattr(
                config,
                "fp4_outlier_weight_rounding_dgrad_consistency",
                False,
            )
        ),
        store_input_dense_main=bool(getattr(config, "fp4_outlier_store_input_dense_main", False)),
        main_quantizer_rht=bool(getattr(config, "fp4_outlier_main_quantizer_rht", False)),
        input_stochastic_rounding=bool(
            getattr(config, "fp4_outlier_input_stochastic_rounding", False)
        ),
    )
