"""Sparse outlier selection for FP4 FPROP inputs."""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

ADAPTIVE_OUTLIER_ZERO_RATIO_THRESHOLD = 1.0e-5


@dataclasses.dataclass(frozen=True)
class OutlierRatio:
    """Resolved outlier ratio metadata for adaptive selection."""

    configured: float
    effective: float
    heaviness: Optional[float]
    adaptive: bool


SparseSplit = tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


def reshape_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor


def has_values(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and int(tensor.numel()) > 0


def compute_heaviness(tensor: torch.Tensor) -> Optional[float]:
    """Return max(abs(x)) / mean(abs(x)) for adaptive ratio selection."""
    abs_tensor = tensor.abs()
    mean_abs = abs_tensor.mean()
    if not bool(torch.isfinite(mean_abs).item()) or float(mean_abs.item()) <= 1.0e-12:
        return None
    return float((abs_tensor.max() / mean_abs).item())


def clamp_adaptive_outlier_ratio(
    ratio: float,
    *,
    min_ratio: float,
    max_ratio: float,
) -> float:
    if ratio < ADAPTIVE_OUTLIER_ZERO_RATIO_THRESHOLD:
        return 0.0
    return max(float(min_ratio), min(float(max_ratio), float(ratio)))


def adaptive_outlier_ratio_from_heaviness(
    *,
    base_ratio: float,
    heaviness: Optional[float],
    enabled: bool,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float,
) -> float:
    if not enabled:
        return float(base_ratio)
    if heaviness is None:
        return 0.0

    log_heaviness = math.log(max(float(heaviness), 1.0))
    log_ref = math.log(max(float(reference_heaviness), 1.0))
    if log_ref <= 0.0:
        return clamp_adaptive_outlier_ratio(
            float(base_ratio),
            min_ratio=min_ratio,
            max_ratio=max_ratio,
        )
    return clamp_adaptive_outlier_ratio(
        float(base_ratio) * (log_heaviness / log_ref),
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )


def resolve_outlier_ratio(
    tensor: torch.Tensor,
    *,
    base_ratio: float,
    adaptive_enabled: bool,
    adaptive_min_ratio: float,
    adaptive_max_ratio: float,
    adaptive_reference_heaviness: float,
) -> OutlierRatio:
    if base_ratio <= 0.0:
        return OutlierRatio(
            configured=float(base_ratio),
            effective=float(base_ratio),
            heaviness=None,
            adaptive=bool(adaptive_enabled),
        )

    heaviness = compute_heaviness(tensor) if adaptive_enabled else None
    effective_ratio = adaptive_outlier_ratio_from_heaviness(
        base_ratio=float(base_ratio),
        heaviness=heaviness,
        enabled=adaptive_enabled,
        min_ratio=float(adaptive_min_ratio),
        max_ratio=float(adaptive_max_ratio),
        reference_heaviness=float(adaptive_reference_heaviness),
    )
    return OutlierRatio(
        configured=float(base_ratio),
        effective=float(effective_ratio),
        heaviness=heaviness,
        adaptive=bool(adaptive_enabled),
    )


def select_topk_outlier_flat_indices(
    tensor: torch.Tensor,
    *,
    outlier_ratio: float,
) -> Optional[torch.Tensor]:
    if outlier_ratio <= 0.0:
        return None
    numel = int(tensor.numel())
    nnz = min(numel, max(1, int(numel * outlier_ratio)))
    _, flat_indices = torch.topk(tensor.reshape(-1).abs(), k=nnz, sorted=False)
    return flat_indices


def select_normal_threshold_outlier_flat_indices(
    tensor: torch.Tensor,
    *,
    outlier_ratio: float,
) -> Optional[torch.Tensor]:
    if outlier_ratio <= 0.0:
        return None
    if outlier_ratio >= 1.0:
        return torch.arange(tensor.numel(), device=tensor.device)

    centered = tensor.to(torch.float32)
    mean = centered.mean()
    std = centered.std(unbiased=False)
    if not torch.isfinite(std) or float(std.detach().item()) == 0.0:
        return None

    tail_probability = min(max(outlier_ratio / 2.0, 1.0e-12), 0.5 - 1.0e-12)
    quantile = 1.0 - tail_probability
    z_score = math.sqrt(2.0) * torch.erfinv(centered.new_tensor(2.0 * quantile - 1.0))
    threshold = z_score * std
    mask = (centered - mean).abs() >= threshold
    return mask.reshape(-1).nonzero(as_tuple=False).flatten()


def row_offsets_from_rows(rows: torch.Tensor, *, num_rows: int) -> torch.Tensor:
    row_counts = torch.bincount(rows.to(torch.int64), minlength=int(num_rows))
    row_offsets = torch.empty(int(num_rows) + 1, device=rows.device, dtype=torch.int32)
    row_offsets[0] = 0
    row_offsets[1:] = row_counts.cumsum(0).to(torch.int32)
    return row_offsets


def split_outliers_from_flat_indices(
    tensor: torch.Tensor,
    flat_indices: torch.Tensor,
) -> SparseSplit:
    if int(flat_indices.numel()) == 0:
        return tensor, None, None, None, None, None

    cols = int(tensor.shape[1])
    flat_indices = flat_indices.reshape(-1)
    outlier_rows = torch.div(flat_indices, cols, rounding_mode="floor").to(torch.int32)
    outlier_cols = (flat_indices % cols).to(torch.int32)
    outlier_values = tensor.reshape(-1)[flat_indices]
    outlier_flat_indices = flat_indices.to(torch.int32).contiguous()
    outlier_row_offsets = row_offsets_from_rows(outlier_rows, num_rows=int(tensor.shape[0]))

    main = tensor.clone()
    main.reshape(-1)[flat_indices] = 0
    return (
        main,
        outlier_rows,
        outlier_cols,
        outlier_values,
        outlier_flat_indices,
        outlier_row_offsets,
    )


def split_outliers(
    tensor: torch.Tensor,
    *,
    outlier_ratio: float,
    selection_method: str,
    effective_outlier_ratio: Optional[float] = None,
) -> SparseSplit:
    ratio = float(outlier_ratio if effective_outlier_ratio is None else effective_outlier_ratio)
    if selection_method == "topk":
        flat_indices = select_topk_outlier_flat_indices(tensor, outlier_ratio=ratio)
    elif selection_method == "normal_threshold":
        flat_indices = select_normal_threshold_outlier_flat_indices(
            tensor,
            outlier_ratio=ratio,
        )
    else:
        raise ValueError(
            f"Unsupported fp4 outlier selection method '{selection_method}'. "
            "Expected 'topk' or 'normal_threshold'."
        )
    if flat_indices is None:
        return tensor, None, None, None, None, None
    return split_outliers_from_flat_indices(tensor, flat_indices)
