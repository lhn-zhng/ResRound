"""Small runtime helpers shared by the FP4 outlier recipe and TE wrappers."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_LOGGED_MESSAGES: set[str] = set()
_QUANTIZER_LAYER_NAMES: dict[int, str] = {}


def log_rank0_once(key: str, message: str, *args) -> None:
    if key in _LOGGED_MESSAGES:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    elif int(os.getenv("RANK", "0")) != 0:
        return
    _LOGGED_MESSAGES.add(key)
    logger.warning(message, *args)


def set_layer_name(quantizer: Any, name: str) -> None:
    """Attach a human-readable module path to a quantizer when TE exposes one."""
    setattr(quantizer, "layer_name", name)
    _QUANTIZER_LAYER_NAMES[id(quantizer)] = name


def select_te_default_fprop_ag_quantizers(
    *,
    input_quantizer,
    weight_quantizer,
    output_quantizer,
    grad_output_quantizer,
    parallel_mode,
    sequence_parallel,
):
    """Compatibility hook for older wrappers.

    The clean FPROP-input branch does not replace quantizers for custom all-gather modes, so this
    intentionally returns the original quantizers.
    """
    del parallel_mode, sequence_parallel
    return input_quantizer, weight_quantizer, output_quantizer, grad_output_quantizer
