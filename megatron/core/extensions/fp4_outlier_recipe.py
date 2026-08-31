"""Compatibility entrypoint for the modular FP4 FPROP input-outlier recipe."""

from megatron.core.extensions.fp4_outlier import (
    FP4FpropInputOutlierConfig,
    configure_from_transformer_config,
    get_config,
    nvfp4_outlier_quantizer_factory,
    select_te_default_fprop_ag_quantizers,
    set_layer_name,
)

__all__ = [
    "FP4FpropInputOutlierConfig",
    "configure_from_transformer_config",
    "get_config",
    "nvfp4_outlier_quantizer_factory",
    "select_te_default_fprop_ag_quantizers",
    "set_layer_name",
]
