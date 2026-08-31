"""Ephemeral local-covariance rounding for FPROP NVFP4 weights.

The routine keeps the native NVFP4 scales and payload shape.  For every output
row and local input-channel group, a configured small number of E2M1 codes may
move from their round-to-nearest endpoint to the other adjacent endpoint.  The
FPROP rowwise payload may be reused for the rest of the current quantized-weight
generation.  An optional dual-view mode mirrors the selected code displacement
into a cached columnwise NVFP4 view for DGRAD; the master weight and WGRAD path
remain untouched.
A final row-wise guard evaluates all proposed group changes jointly against the
full input width, so accepted payload rows strictly decrease the sampled
weight-induced FPROP error even when cross-group covariance is nonzero.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import torch

from .config import get_config
from .runtime import log_rank0_once


_E2M1_LEVEL_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_LEVEL_CACHE: dict[tuple[str, Optional[int]], torch.Tensor] = {}
_EXACT_ROW_GUARD_CHUNK_ROWS = 1024
_EXACT_ROW_GUARD_EPS_MULTIPLIER = 64.0
_OFFLINE_SWEEP_ELIGIBLE_CALL_INDEX = 0


def _e2m1_levels(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    levels = _LEVEL_CACHE.get(key)
    if levels is None:
        levels = torch.tensor(
            _E2M1_LEVEL_VALUES,
            device=device,
            dtype=torch.float32,
        )
        _LEVEL_CACHE[key] = levels
    return levels


def unpack_fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack TE's low-nibble-first FP4x2 byte layout."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError(
            "Expected a 2D uint8 rowwise NVFP4 payload, "
            f"got dtype={packed.dtype}, shape={tuple(packed.shape)}."
        )
    rows, packed_cols = packed.shape
    codes = torch.empty(
        (rows, packed_cols * 2),
        device=packed.device,
        dtype=torch.uint8,
    )
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibbles into TE's rowwise FP4x2 byte layout."""
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError(
            "Expected an even-width 2D uint8 E2M1 code tensor, "
            f"got dtype={codes.dtype}, shape={tuple(codes.shape)}."
        )
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def sample_weight_rounding_inputs(
    quantized_input: torch.Tensor,
    selection_tokens: int,
    *,
    stratified: bool,
    stratified_batch_size: int = 0,
) -> torch.Tensor:
    """Select calibration rows while preserving sequence and batch coverage.

    Megatron linear inputs normally have shape ``[sequence, batch, hidden]``.
    A flat fixed-stride sample can alias with the batch dimension (for example,
    stride 32 with microbatch size 32 selects batch lane zero only). The
    stratified path cycles through evenly spaced batch lanes and sequence
    positions without changing the requested sample count. Non-3D inputs and
    the disabled path retain the legacy flattened-stride behavior.
    """
    if selection_tokens <= 0:
        raise ValueError("selection_tokens must be positive.")

    if stratified and quantized_input.ndim == 2:
        if stratified_batch_size <= 0:
            raise ValueError(
                "stratified_batch_size must be positive for flattened inputs."
            )
        if quantized_input.shape[0] % stratified_batch_size:
            raise ValueError(
                "Flattened token count must be divisible by stratified_batch_size: "
                f"tokens={quantized_input.shape[0]}, batch={stratified_batch_size}."
            )
        quantized_input = quantized_input.reshape(
            quantized_input.shape[0] // stratified_batch_size,
            stratified_batch_size,
            quantized_input.shape[1],
        )

    if stratified and quantized_input.ndim == 3:
        sequence, batch, hidden = quantized_input.shape
        total_tokens = int(sequence) * int(batch)
        sample_count = min(int(selection_tokens), total_tokens)
        if sample_count == total_tokens:
            return quantized_input.reshape(total_tokens, hidden)

        batch_slots = min(int(batch), sample_count)
        rounds = (sample_count + batch_slots - 1) // batch_slots
        sample_ids = torch.arange(
            sample_count,
            device=quantized_input.device,
            dtype=torch.int64,
        )
        batch_slot = torch.remainder(sample_ids, batch_slots)
        batch_index = torch.div(
            (2 * batch_slot + 1) * int(batch),
            2 * batch_slots,
            rounding_mode="floor",
        )
        round_index = torch.div(sample_ids, batch_slots, rounding_mode="floor")
        sequence_index = torch.div(
            (2 * round_index + 1) * int(sequence),
            2 * rounds,
            rounding_mode="floor",
        )
        return quantized_input[sequence_index, batch_index]

    flattened = quantized_input.reshape(-1, quantized_input.shape[-1])
    stride = max(
        1,
        (flattened.shape[0] + selection_tokens - 1) // selection_tokens,
    )
    return flattened[::stride][:selection_tokens]


def split_weight_rounding_crossfit_inputs(
    sampled_input: torch.Tensor,
    audit_tokens: int,
    *,
    stratified_batch_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sampled rows into disjoint proposal and held-out audit sets.

    When the sample was constructed in sequence-round-major, batch-lane-minor
    order, whole sequence rounds are alternated between the two sets.  Thus
    both halves keep all batch lanes instead of an even/odd flattened split
    aliasing with the batch dimension.
    """
    if sampled_input.ndim != 2:
        raise ValueError(
            "Cross-fit input must be a 2D [tokens, hidden] tensor, got "
            f"shape={tuple(sampled_input.shape)}."
        )
    total_tokens = int(sampled_input.shape[0])
    if not 0 < audit_tokens < total_tokens:
        raise ValueError(
            "audit_tokens must be between zero and the sampled token count: "
            f"audit_tokens={audit_tokens}, sampled_tokens={total_tokens}."
        )

    proposal_tokens = total_tokens - int(audit_tokens)
    batch = int(stratified_batch_size)
    if (
        batch > 0
        and total_tokens % batch == 0
        and audit_tokens % batch == 0
        and proposal_tokens % batch == 0
    ):
        rounds = total_tokens // batch
        audit_rounds = audit_tokens // batch
        sampled_rounds = sampled_input.reshape(rounds, batch, sampled_input.shape[1])
        audit_round_ids = torch.div(
            (2 * torch.arange(audit_rounds, device=sampled_input.device) + 1)
            * rounds,
            2 * audit_rounds,
            rounding_mode="floor",
        )
        audit_round_mask = torch.zeros(
            rounds,
            device=sampled_input.device,
            dtype=torch.bool,
        )
        audit_round_mask[audit_round_ids] = True
        proposal_input = sampled_rounds[~audit_round_mask].reshape(
            proposal_tokens,
            sampled_input.shape[1],
        )
        audit_input = sampled_rounds[audit_round_mask].reshape(
            audit_tokens,
            sampled_input.shape[1],
        )
        return proposal_input, audit_input

    return sampled_input[:proposal_tokens], sampled_input[proposal_tokens:]


def adjacent_codes_and_values(
    weight: torch.Tensor,
    nearest: torch.Tensor,
    nearest_codes: torch.Tensor,
    *,
    scale_tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover each weight's other adjacent E2M1 endpoint at fixed scales."""
    if weight.shape != nearest.shape or weight.shape != nearest_codes.shape:
        raise ValueError(
            "Weight, dequantized weight, and code shapes must match: "
            f"{tuple(weight.shape)}, {tuple(nearest.shape)}, "
            f"{tuple(nearest_codes.shape)}."
        )
    rows, cols = weight.shape
    if rows % scale_tile_size or cols % scale_tile_size:
        raise ValueError(
            "NVFP4 2D weight dimensions must be divisible by the scale tile "
            f"{scale_tile_size}, got {rows}x{cols}."
        )

    levels = _e2m1_levels(weight.device)
    nearest_index = (nearest_codes & 0x07).to(torch.long)
    nearest_level = levels[nearest_index]

    row_tiles = rows // scale_tile_size
    col_tiles = cols // scale_tile_size
    scale_samples = torch.where(
        nearest_level > 0.0,
        nearest.abs() / nearest_level.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(nearest),
    )
    tile_scale = (
        scale_samples.reshape(
            row_tiles,
            scale_tile_size,
            col_tiles,
            scale_tile_size,
        )
        .permute(0, 2, 1, 3)
        .amax(dim=(-1, -2))
    )
    effective_scale = (
        tile_scale[:, :, None, None]
        .expand(row_tiles, col_tiles, scale_tile_size, scale_tile_size)
        .permute(0, 2, 1, 3)
        .reshape(rows, cols)
    )

    scaled_abs = weight.abs() / effective_scale.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    floor_index = (
        (scaled_abs[..., None] >= levels)
        .sum(dim=-1)
        .sub(1)
        .clamp_(0, len(_E2M1_LEVEL_VALUES) - 1)
    )
    ceil_index = (
        (scaled_abs[..., None] > levels)
        .sum(dim=-1)
        .clamp_(0, len(_E2M1_LEVEL_VALUES) - 1)
    )
    floor_level = levels[floor_index]
    ceil_level = levels[ceil_index]
    nearest_is_floor = (nearest_level - floor_level).abs() <= (
        nearest_level - ceil_level
    ).abs()
    adjacent_index = torch.where(nearest_is_floor, ceil_index, floor_index)
    sign_code = (weight < 0.0).to(torch.uint8) << 3
    adjacent_codes = sign_code | adjacent_index.to(torch.uint8)
    adjacent_values = (
        torch.where(weight < 0.0, -1.0, 1.0)
        * levels[adjacent_index]
        * effective_scale
    )
    return adjacent_codes, adjacent_values


def select_local_covariance_codes(
    quantized_input: torch.Tensor,
    weight: torch.Tensor,
    nearest: torch.Tensor,
    nearest_codes: torch.Tensor,
    adjacent: torch.Tensor,
    adjacent_codes: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select improving adjacent codes per row and input group."""
    error = nearest.float() - weight.float()
    delta = adjacent.float() - nearest.float()
    return select_local_covariance_codes_from_error(
        quantized_input,
        error,
        delta,
        nearest_codes,
        adjacent_codes,
        group_size=group_size,
    )


def _enforce_exact_row_descent(
    quantized_input: torch.Tensor,
    error: torch.Tensor,
    delta: torch.Tensor,
    selected: torch.Tensor,
    *,
    group_size: int,
    max_updates_per_group: int,
    dense_main: Optional[torch.Tensor] = None,
    dense_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reject joint row updates that do not decrease the full sampled error.

    The block-local selector ignores covariance between different input
    groups.  For a row error ``e`` and the joint sparse update ``d``, this
    guard evaluates the equivalent exact condition

        ||X (e + d)||_2^2 - ||X e||_2^2
        = 2 d^T X^T X e + d^T X^T X d < 0.

    Computing the correction from the selected coordinates keeps its work
    sparse.  The baseline residual still sees the complete row error, which is
    required for a genuine guarantee in the presence of cross-group terms.
    When ``dense_main`` and ``dense_weight`` are provided, the guarded
    baseline is the complete input-plus-weight residual

        X_q @ e.T + (X_q - X_dense) @ W.T.
    """
    tokens, width = quantized_input.shape
    rows = error.shape[0]
    joint_objective = dense_main is not None or dense_weight is not None
    if (
        error.shape != delta.shape
        or error.shape != selected.shape
        or width != error.shape[1]
        or width % group_size
        or max_updates_per_group <= 0
        or max_updates_per_group > group_size
    ):
        raise ValueError(
            "Incompatible exact-row-guard shapes: "
            f"input={tuple(quantized_input.shape)}, error={tuple(error.shape)}, "
            f"delta={tuple(delta.shape)}, selected={tuple(selected.shape)}, "
            f"group_size={group_size}, "
            f"max_updates_per_group={max_updates_per_group}."
        )
    if joint_objective and (
        dense_main is None
        or dense_weight is None
        or dense_main.shape != quantized_input.shape
        or dense_weight.shape != error.shape
    ):
        raise ValueError(
            "Joint exact-row guard requires matching dense main and weight: "
            f"input={tuple(quantized_input.shape)}, "
            f"dense_main={None if dense_main is None else tuple(dense_main.shape)}, "
            f"error={tuple(error.shape)}, "
            f"dense_weight={None if dense_weight is None else tuple(dense_weight.shape)}."
        )
    if tokens == 0 or rows == 0:
        return selected

    blocks = width // group_size
    selected_blocks = selected.reshape(rows, blocks, group_size)
    delta_blocks = delta.float().reshape(rows, blocks, group_size)
    block_has_update = selected_blocks.any(dim=2)
    active_rows = block_has_update.any(dim=1)

    selected_index = selected_blocks.to(torch.uint8).topk(
        k=max_updates_per_group,
        dim=2,
    ).indices
    selected_mask = selected_blocks.gather(2, selected_index)
    selected_delta = delta_blocks.gather(
        2,
        selected_index,
    )
    selected_delta = torch.where(
        selected_mask,
        selected_delta,
        torch.zeros_like(selected_delta),
    )
    block_offsets = (
        torch.arange(blocks, device=selected.device, dtype=torch.int64)
        * group_size
    )
    selected_column = selected_index + block_offsets[None, :, None]

    x = quantized_input.float()
    input_quantization_error = (
        x - dense_main.float() if joint_objective else None
    )
    error_fp32 = error.float()
    dense_weight_fp32 = dense_weight.float() if joint_objective else None
    accepted_rows = torch.zeros(rows, device=selected.device, dtype=torch.bool)
    eps = torch.finfo(torch.float32).eps
    for row_start in range(0, rows, _EXACT_ROW_GUARD_CHUNK_ROWS):
        row_end = min(rows, row_start + _EXACT_ROW_GUARD_CHUNK_ROWS)
        chunk_rows = row_end - row_start
        columns = selected_column[row_start:row_end]
        update = selected_delta[row_start:row_end]

        selected_x = x[:, columns.reshape(-1)].reshape(
            tokens,
            chunk_rows,
            blocks,
            max_updates_per_group,
        )
        correction = (selected_x * update[None]).sum(dim=(2, 3))
        baseline = x @ error_fp32[row_start:row_end].t()
        if input_quantization_error is not None:
            baseline = baseline + (
                input_quantization_error
                @ dense_weight_fp32[row_start:row_end].t()
            )
        baseline_energy = baseline.square().sum(dim=0)
        updated_energy = (baseline + correction).square().sum(dim=0)

        # Reject numerically marginal changes as well as true regressions.  The
        # tolerance makes the strict descent claim robust to FP32 reduction
        # roundoff without changing any accepted payload value.
        energy_scale = torch.maximum(
            baseline_energy.abs(),
            updated_energy.abs(),
        ).clamp_min(1.0)
        tolerance = _EXACT_ROW_GUARD_EPS_MULTIPLIER * eps * energy_scale
        accepted_rows[row_start:row_end] = (
            updated_energy < baseline_energy - tolerance
        ) & active_rows[row_start:row_end]

    guarded = selected_blocks & accepted_rows[:, None, None]
    return guarded.reshape_as(selected)


def enforce_crossfit_audit(
    audit_input: torch.Tensor,
    error: torch.Tensor,
    delta: torch.Tensor,
    selected: torch.Tensor,
    *,
    group_size: int,
    max_updates_per_group: int,
    max_regression_fraction: float,
    min_relative_gain: float = 0.0,
    dense_main: Optional[torch.Tensor] = None,
    dense_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reject proposal-selected rows that fail an independent token audit.

    For held-out token ``t`` and output row ``j``, let ``r[t,j]`` be the
    original output error and ``c[t,j]`` the correction from the proposed code
    moves. By default ``r`` is the TE weight-induced error. When ``dense_main``
    and ``dense_weight`` are supplied, ``r`` is instead the complete dense-main
    FPROP residual

      audit_input @ (dense_weight + error).T
      - dense_main @ dense_weight.T.

    A row is retained only when

      sum_t ((r[t,j] + c[t,j])**2 - r[t,j]**2) < 0

    and the fraction of tokens with a numerically measurable scalar-error
    regression does not exceed ``max_regression_fraction``.  Rejected rows
    return to their original TE codes, so the emitted payload remains a single
    shared weight tensor and does not require token-dependent GEMMs.
    """
    tokens, width = audit_input.shape
    rows = error.shape[0]
    combined_audit = dense_main is not None or dense_weight is not None
    if (
        tokens <= 0
        or error.shape != delta.shape
        or error.shape != selected.shape
        or width != error.shape[1]
        or width % group_size
        or max_updates_per_group <= 0
        or max_updates_per_group > group_size
        or not 0.0 <= max_regression_fraction <= 1.0
        or not 0.0 <= min_relative_gain <= 1.0
    ):
        raise ValueError(
            "Incompatible cross-fit audit inputs: "
            f"audit_input={tuple(audit_input.shape)}, error={tuple(error.shape)}, "
            f"delta={tuple(delta.shape)}, selected={tuple(selected.shape)}, "
            f"group_size={group_size}, "
            f"max_updates_per_group={max_updates_per_group}, "
            f"max_regression_fraction={max_regression_fraction}, "
            f"min_relative_gain={min_relative_gain}."
        )
    if combined_audit and (
        dense_main is None
        or dense_weight is None
        or dense_main.shape != audit_input.shape
        or dense_weight.shape != error.shape
    ):
        raise ValueError(
            "Combined cross-fit audit requires matching dense main and weight: "
            f"audit_input={tuple(audit_input.shape)}, "
            f"dense_main={None if dense_main is None else tuple(dense_main.shape)}, "
            f"error={tuple(error.shape)}, "
            f"dense_weight={None if dense_weight is None else tuple(dense_weight.shape)}."
        )
    if rows == 0:
        return selected

    blocks = width // group_size
    selected_blocks = selected.reshape(rows, blocks, group_size)
    delta_blocks = delta.float().reshape(rows, blocks, group_size)
    active_rows = selected_blocks.any(dim=2).any(dim=1)
    selected_index = selected_blocks.to(torch.uint8).topk(
        k=max_updates_per_group,
        dim=2,
    ).indices
    selected_mask = selected_blocks.gather(2, selected_index)
    selected_delta = delta_blocks.gather(2, selected_index)
    selected_delta = torch.where(
        selected_mask,
        selected_delta,
        torch.zeros_like(selected_delta),
    )
    block_offsets = (
        torch.arange(blocks, device=selected.device, dtype=torch.int64)
        * group_size
    )
    selected_column = selected_index + block_offsets[None, :, None]

    x = audit_input.float()
    input_quantization_error = (
        x - dense_main.float() if combined_audit else None
    )
    error_fp32 = error.float()
    dense_weight_fp32 = dense_weight.float() if combined_audit else None
    accepted_rows = torch.zeros(rows, device=selected.device, dtype=torch.bool)
    eps = torch.finfo(torch.float32).eps
    for row_start in range(0, rows, _EXACT_ROW_GUARD_CHUNK_ROWS):
        row_end = min(rows, row_start + _EXACT_ROW_GUARD_CHUNK_ROWS)
        chunk_rows = row_end - row_start
        columns = selected_column[row_start:row_end]
        update = selected_delta[row_start:row_end]

        selected_x = x[:, columns.reshape(-1)].reshape(
            tokens,
            chunk_rows,
            blocks,
            max_updates_per_group,
        )
        correction = (selected_x * update[None]).sum(dim=(2, 3))
        baseline = x @ error_fp32[row_start:row_end].t()
        if input_quantization_error is not None:
            baseline = baseline + (
                input_quantization_error
                @ dense_weight_fp32[row_start:row_end].t()
            )
        baseline_square = baseline.square()
        updated_square = (baseline + correction).square()

        baseline_energy = baseline_square.sum(dim=0)
        updated_energy = updated_square.sum(dim=0)
        energy_scale = torch.maximum(
            baseline_energy.abs(),
            updated_energy.abs(),
        ).clamp_min(1.0)
        aggregate_tolerance = (
            _EXACT_ROW_GUARD_EPS_MULTIPLIER * eps * energy_scale
        )
        required_descent = torch.maximum(
            aggregate_tolerance,
            baseline_energy * min_relative_gain,
        )
        aggregate_descent = (
            updated_energy < baseline_energy - required_descent
        )

        token_scale = torch.maximum(
            baseline_square.abs(),
            updated_square.abs(),
        ).clamp_min(1.0)
        token_tolerance = _EXACT_ROW_GUARD_EPS_MULTIPLIER * eps * token_scale
        regression_fraction = (
            updated_square > baseline_square + token_tolerance
        ).float().mean(dim=0)
        accepted_rows[row_start:row_end] = (
            aggregate_descent
            & (regression_fraction <= max_regression_fraction)
            & active_rows[row_start:row_end]
        )

    audited = selected_blocks & accepted_rows[:, None, None]
    return audited.reshape_as(selected)


def select_local_covariance_codes_from_error(
    quantized_input: torch.Tensor,
    error: torch.Tensor,
    delta: torch.Tensor,
    nearest_codes: torch.Tensor,
    adjacent_codes: torch.Tensor,
    *,
    group_size: int,
    rounds_per_group: int = 1,
    offdiag_shrink: float = 1.0,
    dense_main: Optional[torch.Tensor] = None,
    dense_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily select local adjacent codes from output-residual changes.

    The default objective is the sampled weight-induced error.  Supplying
    ``dense_main`` and ``dense_weight`` adds the local input-quantization
    cross term, so proposals approximate descent of

        ||X_q @ (W_q + d).T - X_dense @ W.T||_F^2.
    """
    tokens, width = quantized_input.shape
    rows = error.shape[0]
    joint_objective = dense_main is not None or dense_weight is not None
    if (
        error.shape != delta.shape
        or error.shape != nearest_codes.shape
        or error.shape != adjacent_codes.shape
        or width != error.shape[1]
        or width % group_size
        or rounds_per_group <= 0
        or rounds_per_group > group_size
        or not 0.0 <= offdiag_shrink <= 1.0
    ):
        raise ValueError(
            "Incompatible input/error/delta/code/group shapes: "
            f"input={tuple(quantized_input.shape)}, error={tuple(error.shape)}, "
            f"delta={tuple(delta.shape)}, nearest_codes={tuple(nearest_codes.shape)}, "
            f"adjacent_codes={tuple(adjacent_codes.shape)}, group_size={group_size}, "
            f"rounds_per_group={rounds_per_group}, "
            f"offdiag_shrink={offdiag_shrink}."
        )
    if joint_objective and (
        dense_main is None
        or dense_weight is None
        or dense_main.shape != quantized_input.shape
        or dense_weight.shape != error.shape
    ):
        raise ValueError(
            "Joint local-covariance selection requires matching dense main and weight: "
            f"input={tuple(quantized_input.shape)}, "
            f"dense_main={None if dense_main is None else tuple(dense_main.shape)}, "
            f"error={tuple(error.shape)}, "
            f"dense_weight={None if dense_weight is None else tuple(dense_weight.shape)}."
        )
    blocks = width // group_size
    x_blocks = quantized_input.float().reshape(tokens, blocks, group_size)
    covariance = torch.einsum("tbi,tbj->bij", x_blocks, x_blocks)
    joint_cross_covariance = None
    if joint_objective:
        input_error_blocks = (
            quantized_input.float() - dense_main.float()
        ).reshape(tokens, blocks, group_size)
        joint_cross_covariance = torch.einsum(
            "tbi,tbj->bij",
            x_blocks,
            input_error_blocks,
        )
    if offdiag_shrink < 1.0:
        covariance_diagonal = torch.diag_embed(
            covariance.diagonal(dim1=-2, dim2=-1)
        )
        covariance = covariance_diagonal + offdiag_shrink * (
            covariance - covariance_diagonal
        )
        if joint_cross_covariance is not None:
            joint_cross_diagonal = torch.diag_embed(
                joint_cross_covariance.diagonal(dim1=-2, dim2=-1)
            )
            joint_cross_covariance = joint_cross_diagonal + offdiag_shrink * (
                joint_cross_covariance - joint_cross_diagonal
            )

    error_blocks = error.float().reshape(rows, blocks, group_size)
    delta_blocks = delta.float().reshape(rows, blocks, group_size)
    diagonal = covariance.diagonal(dim1=-2, dim2=-1)[None]
    available = adjacent_codes.reshape_as(nearest_codes) != nearest_codes
    available = available.reshape(rows, blocks, group_size)
    selected = torch.zeros_like(available)
    cov_error = torch.einsum("bij,nbj->nbi", covariance, error_blocks)
    if joint_cross_covariance is not None:
        dense_weight_blocks = dense_weight.float().reshape(
            rows,
            blocks,
            group_size,
        )
        cov_error = cov_error + torch.einsum(
            "bij,nbj->nbi",
            joint_cross_covariance,
            dense_weight_blocks,
        )
    covariance_for_rows = (
        covariance[None].expand(rows, -1, -1, -1)
        if rounds_per_group > 1
        else None
    )
    for round_index in range(rounds_per_group):
        change = (
            2.0 * delta_blocks * cov_error
            + delta_blocks.square() * diagonal
        )
        change = torch.where(
            available & (change < 0.0),
            change,
            torch.full_like(change, float("inf")),
        )
        best_change, best_index = change.min(dim=2)
        take = torch.isfinite(best_change)
        round_selected = torch.zeros_like(available)
        round_selected.scatter_(2, best_index[..., None], take[..., None])
        selected |= round_selected
        available &= ~round_selected

        if round_index + 1 == rounds_per_group:
            continue

        # Update G @ e exactly from the selected covariance column instead of
        # recomputing the dense block matvec for every greedy round:
        # G @ (e + delta_k e_k) = G @ e + delta_k G[:, k].
        chosen_delta = delta_blocks.gather(
            2, best_index[..., None]
        ).squeeze(2)
        chosen_delta = torch.where(take, chosen_delta, 0.0)
        chosen_covariance_column = torch.gather(
            covariance_for_rows,
            3,
            best_index[:, :, None, None].expand(
                -1, -1, group_size, 1
            ),
        ).squeeze(3)
        cov_error = cov_error + (
            chosen_delta[:, :, None] * chosen_covariance_column
        )
    selected = _enforce_exact_row_descent(
        quantized_input,
        error,
        delta,
        selected.reshape_as(nearest_codes),
        group_size=group_size,
        max_updates_per_group=rounds_per_group,
        dense_main=dense_main,
        dense_weight=dense_weight,
    ).reshape_as(available)

    rounded_codes = torch.where(
        selected.reshape_as(nearest_codes),
        adjacent_codes,
        nearest_codes,
    )
    return rounded_codes, selected.reshape_as(nearest_codes)


def _static_rounding_cache_key(
    cache_owner,
    dense_weight: torch.Tensor,
    packed_weight: torch.Tensor,
) -> Optional[tuple]:
    """Build a generation-safe key for the weight-only rounding state."""
    if cache_owner is None:
        return None
    generation = getattr(cache_owner, "weight_update_generation", None)
    if generation is None:
        return None
    return (
        int(generation),
        int(dense_weight.untyped_storage().data_ptr()),
        int(dense_weight.storage_offset()),
        int(getattr(dense_weight, "_version", -1)),
        tuple(dense_weight.shape),
        tuple(dense_weight.stride()),
        int(packed_weight.untyped_storage().data_ptr()),
        int(packed_weight.storage_offset()),
        int(getattr(packed_weight, "_version", -1)),
        tuple(packed_weight.shape),
        int(dense_weight.device.index or torch.cuda.current_device()),
    )


@torch.no_grad()
def _maybe_run_offline_sample_sweep(
    quantized_input: torch.Tensor,
    dense_main: Optional[torch.Tensor],
    dense_weight: torch.Tensor,
    error: torch.Tensor,
    delta: torch.Tensor,
    nearest_codes: torch.Tensor,
    adjacent_codes: torch.Tensor,
    *,
    group_size: int,
    rounds_per_group: int,
    offdiag_shrink: float,
    stratified_batch_size: int,
) -> None:
    """One-shot, environment-gated sweep on real model activations.

    This diagnostic is disabled by default. Eligible calls are QKV/FC1 pairs
    in layer order when expansion-only rounding is active.
    """
    if os.environ.get("FP4_WEIGHT_ROUNDING_OFFLINE_SWEEP", "0") != "1":
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    global _OFFLINE_SWEEP_ELIGIBLE_CALL_INDEX
    call_index = _OFFLINE_SWEEP_ELIGIBLE_CALL_INDEX
    _OFFLINE_SWEEP_ELIGIBLE_CALL_INDEX += 1
    targets = {
        int(value)
        for value in os.environ.get(
            "FP4_WEIGHT_ROUNDING_OFFLINE_SWEEP_CALLS",
            "0,1,2,30,31,32,63,64,65",
        ).split(",")
        if value.strip()
    }
    if call_index not in targets:
        return

    if quantized_input.ndim == 2:
        batch = int(stratified_batch_size)
        if batch <= 0 or quantized_input.shape[0] % batch:
            raise ValueError(
                "Offline sweep requires a valid stratified batch size for "
                f"2D input, got shape={tuple(quantized_input.shape)}, batch={batch}."
            )
        input_3d = quantized_input.reshape(
            quantized_input.shape[0] // batch,
            batch,
            quantized_input.shape[1],
        )
        dense_main_3d = (
            dense_main.reshape(
                quantized_input.shape[0] // batch,
                batch,
                quantized_input.shape[1],
            )
            if dense_main is not None
            else None
        )
    elif quantized_input.ndim == 3:
        input_3d = quantized_input
        dense_main_3d = dense_main
        batch = int(input_3d.shape[1])
    else:
        raise ValueError(
            "Offline sweep expects [tokens, hidden] or [sequence, batch, hidden], "
            f"got {tuple(quantized_input.shape)}."
        )

    # Even and odd sequence positions are disjoint but cover every batch lane.
    calibration_source = input_3d[0::2]
    test_source = input_3d[1::2]
    calibration_dense_source = (
        dense_main_3d[0::2] if dense_main_3d is not None else None
    )
    test_dense_source = (
        dense_main_3d[1::2] if dense_main_3d is not None else None
    )
    test_tokens = min(
        int(os.environ.get("FP4_WEIGHT_ROUNDING_OFFLINE_SWEEP_TEST_TOKENS", "2048")),
        int(test_source.shape[0] * test_source.shape[1]),
    )
    test_input = sample_weight_rounding_inputs(
        test_source,
        test_tokens,
        stratified=True,
        stratified_batch_size=batch,
    )
    test_dense_main = (
        sample_weight_rounding_inputs(
            test_dense_source,
            test_tokens,
            stratified=True,
            stratified_batch_size=batch,
        )
        if test_dense_source is not None
        else None
    )

    def energy(weight_error: torch.Tensor) -> torch.Tensor:
        residual = test_input.float() @ weight_error.float().t()
        if test_dense_main is not None:
            residual = residual + (
                (test_input.float() - test_dense_main.float())
                @ dense_weight.float().t()
            )
        return residual.square().sum(dim=0, dtype=torch.float64)

    baseline = energy(error)
    configs = (
        ("old_weight_audit_p2048_a1024_r0p55", 2048, 1024, 0.55, False),
        ("combined_p512_a512_r0p5", 512, 512, 0.5, True),
        ("combined_p1024_a512_r0p5", 1024, 512, 0.5, True),
        ("combined_p1024_a1024_r0p5", 1024, 1024, 0.5, True),
        ("combined_p2048_a512_r0p4", 2048, 512, 0.4, True),
        ("combined_p2048_a512_r0p5", 2048, 512, 0.5, True),
        ("combined_p2048_a1024_r0p4", 2048, 1024, 0.4, True),
        ("combined_p2048_a1024_r0p5", 2048, 1024, 0.5, True),
        ("combined_p2048_a2048_r0p4", 2048, 2048, 0.4, True),
        ("combined_p2048_a2048_r0p5", 2048, 2048, 0.5, True),
    )
    layer = call_index // 3
    module = ("qkv", "proj", "fc1")[call_index % 3]
    for (
        name,
        proposal_tokens,
        audit_tokens,
        max_fraction,
        combined_audit,
    ) in configs:
        sampled = sample_weight_rounding_inputs(
            calibration_source,
            proposal_tokens + audit_tokens,
            stratified=True,
            stratified_batch_size=batch,
        )
        proposal, audit = split_weight_rounding_crossfit_inputs(
            sampled,
            audit_tokens,
            stratified_batch_size=batch,
        )
        audit_dense_main = None
        if combined_audit:
            if calibration_dense_source is None:
                raise ValueError(
                    "Offline combined sweep requires the unquantized dense A1."
                )
            sampled_dense_main = sample_weight_rounding_inputs(
                calibration_dense_source,
                proposal_tokens + audit_tokens,
                stratified=True,
                stratified_batch_size=batch,
            )
            _, audit_dense_main = split_weight_rounding_crossfit_inputs(
                sampled_dense_main,
                audit_tokens,
                stratified_batch_size=batch,
            )
        if error.is_cuda:
            torch.cuda.synchronize(error.device)
        started = time.perf_counter()
        _, proposed = select_local_covariance_codes_from_error(
            proposal,
            error,
            delta,
            nearest_codes,
            adjacent_codes,
            group_size=group_size,
            rounds_per_group=rounds_per_group,
            offdiag_shrink=offdiag_shrink,
        )
        selected = enforce_crossfit_audit(
            audit,
            error,
            delta,
            proposed,
            group_size=group_size,
            max_updates_per_group=rounds_per_group,
            max_regression_fraction=max_fraction,
            dense_main=audit_dense_main,
            dense_weight=dense_weight if combined_audit else None,
        )
        if error.is_cuda:
            torch.cuda.synchronize(error.device)
        selection_ms = (time.perf_counter() - started) * 1000.0
        candidate = energy(
            error + torch.where(selected, delta, torch.zeros_like(delta))
        )
        tolerance = (
            _EXACT_ROW_GUARD_EPS_MULTIPLIER
            * torch.finfo(torch.float32).eps
            * torch.maximum(candidate.abs(), baseline.abs()).clamp_min(1.0)
        )
        record = {
            "call_index": call_index,
            "layer": layer,
            "module": module,
            "config": name,
            "proposal_tokens": proposal_tokens,
            "audit_tokens": audit_tokens,
            "max_regression_fraction": max_fraction,
            "combined_audit": combined_audit,
            "test_delta_percent": 100.0
            * (float(candidate.sum().item()) / float(baseline.sum().item()) - 1.0),
            "regressed_rows": int((candidate > baseline + tolerance).sum().item()),
            "rows": int(error.shape[0]),
            "proposed_updates": int(proposed.sum().item()),
            "retained_updates": int(selected.sum().item()),
            "selection_ms": selection_ms,
        }
        print("FP4_WEIGHT_ROUNDING_OFFLINE_SWEEP " + json.dumps(record), flush=True)


def maybe_round_fprop_weight_data(
    input_storage,
    weight_storage,
    dense_weight: Optional[torch.Tensor],
    *,
    dense_main: Optional[torch.Tensor] = None,
    cache_owner=None,
) -> Optional[torch.Tensor]:
    """Return a rounded rowwise payload, or ``None`` when the feature is off."""
    cfg = get_config()
    if not cfg.enable_weight_rounding:
        return None
    if dense_weight is None:
        raise ValueError(
            "Ephemeral FPROP weight rounding requires the saved dense master weight."
        )

    dense_weight_2d = dense_weight.detach()
    if dense_weight_2d.ndim != 2:
        dense_weight_2d = dense_weight_2d.reshape(
            -1,
            dense_weight_2d.shape[-1],
        )
    rows, width = dense_weight_2d.shape
    if cfg.weight_rounding_expansion_only:
        expansion_ratio = rows / width
        if (
            rows < width
            or expansion_ratio < cfg.weight_rounding_min_expansion_ratio
            or (
                cfg.weight_rounding_max_expansion_ratio > 0.0
                and expansion_ratio > cfg.weight_rounding_max_expansion_ratio
            )
        ):
            return None
    if width % cfg.weight_rounding_group_size:
        raise ValueError(
            f"Weight width {width} is not divisible by configured local covariance "
            f"group size {cfg.weight_rounding_group_size}."
        )

    packed = weight_storage._rowwise_data
    cache_key = _static_rounding_cache_key(
        cache_owner,
        dense_weight_2d,
        packed,
    )
    cached = (
        getattr(cache_owner, "_weight_rounding_static_state", None)
        if cache_key is not None
        else None
    )
    can_cache = not (
        dense_weight_2d.is_cuda
        and torch.cuda.is_current_stream_capturing()
    )
    payload_cache_key = (
        (
            "generation_payload_v5_joint_objective",
            cache_key,
            int(cfg.weight_rounding_group_size),
            int(cfg.weight_rounding_rounds_per_group),
            int(cfg.weight_rounding_selection_tokens),
            bool(cfg.weight_rounding_stratified_sampling),
            int(cfg.weight_rounding_stratified_batch_size),
            bool(cfg.weight_rounding_crossfit_audit),
            bool(cfg.weight_rounding_combined_audit),
            bool(cfg.weight_rounding_joint_objective),
            int(cfg.weight_rounding_audit_tokens),
            float(cfg.weight_rounding_audit_max_regression_fraction),
            float(cfg.weight_rounding_audit_min_relative_gain),
            float(cfg.weight_rounding_offdiag_shrink),
            float(cfg.weight_rounding_min_expansion_ratio),
            float(cfg.weight_rounding_max_expansion_ratio),
            bool(cfg.weight_rounding_dgrad_consistency),
        )
        if cache_key is not None
        else None
    )
    cached_payload = (
        getattr(cache_owner, "_weight_rounding_payload_state", None)
        if (
            cfg.weight_rounding_reuse_generation_payload
            and payload_cache_key is not None
        )
        else None
    )
    if (
        cached_payload is not None
        and cached_payload[0] == payload_cache_key
        and can_cache
    ):
        log_rank0_once(
            "fprop_weight_rounding:payload_cache_hit",
            "FP4 ephemeral weight-rounding generation payload cache hit.",
        )
        return cached_payload[1]

    if cached is not None and cached[0] == cache_key and can_cache:
        error, delta, nearest_codes, adjacent_codes = cached[1:]
        log_rank0_once(
            "fprop_weight_rounding:cache_hit",
            "FP4 ephemeral weight-rounding static-state cache hit.",
        )
    else:
        weight = dense_weight_2d.float()
        nearest = weight_storage.dequantize(dtype=torch.float32).reshape_as(weight)
        nearest_codes = unpack_fp4_codes(packed)
        adjacent_codes, adjacent = adjacent_codes_and_values(
            weight,
            nearest,
            nearest_codes,
        )
        error = nearest - weight
        delta = adjacent - nearest
        if (
            cache_key is not None
            and can_cache
            and not cfg.weight_rounding_reuse_generation_payload
        ):
            setattr(
                cache_owner,
                "_weight_rounding_static_state",
                (
                    cache_key,
                    error,
                    delta,
                    nearest_codes,
                    adjacent_codes,
                ),
            )

    quantized_input = input_storage.dequantize(dtype=torch.float32)
    dense_main_2d = None
    if (
        cfg.weight_rounding_combined_audit
        or cfg.weight_rounding_joint_objective
    ):
        if dense_main is None:
            raise ValueError(
                "Combined weight-rounding audit requires the unquantized dense A1."
            )
        dense_main_2d = dense_main.detach()
        if dense_main_2d.ndim != 2:
            dense_main_2d = dense_main_2d.reshape(
                -1,
                dense_main_2d.shape[-1],
            )
        if dense_main_2d.shape != quantized_input.shape:
            raise ValueError(
                "Combined weight-rounding audit input shape mismatch: "
                f"quantized={tuple(quantized_input.shape)}, "
                f"dense_main={tuple(dense_main_2d.shape)}."
            )
    _maybe_run_offline_sample_sweep(
        quantized_input,
        dense_main_2d,
        dense_weight_2d,
        error,
        delta,
        nearest_codes,
        adjacent_codes,
        group_size=cfg.weight_rounding_group_size,
        rounds_per_group=cfg.weight_rounding_rounds_per_group,
        offdiag_shrink=cfg.weight_rounding_offdiag_shrink,
        stratified_batch_size=cfg.weight_rounding_stratified_batch_size,
    )
    selection_input = sample_weight_rounding_inputs(
        quantized_input,
        cfg.weight_rounding_selection_tokens,
        stratified=cfg.weight_rounding_stratified_sampling,
        stratified_batch_size=cfg.weight_rounding_stratified_batch_size,
    )
    selection_dense_main = (
        sample_weight_rounding_inputs(
            dense_main_2d,
            cfg.weight_rounding_selection_tokens,
            stratified=cfg.weight_rounding_stratified_sampling,
            stratified_batch_size=cfg.weight_rounding_stratified_batch_size,
        )
        if dense_main_2d is not None
        else None
    )
    audit_input = None
    audit_dense_main = None
    if cfg.weight_rounding_crossfit_audit:
        selection_input, audit_input = split_weight_rounding_crossfit_inputs(
            selection_input,
            cfg.weight_rounding_audit_tokens,
            stratified_batch_size=(
                cfg.weight_rounding_stratified_batch_size
                if cfg.weight_rounding_stratified_sampling
                else 0
            ),
        )
        if selection_dense_main is not None:
            selection_dense_main, audit_dense_main = (
                split_weight_rounding_crossfit_inputs(
                    selection_dense_main,
                    cfg.weight_rounding_audit_tokens,
                    stratified_batch_size=(
                        cfg.weight_rounding_stratified_batch_size
                        if cfg.weight_rounding_stratified_sampling
                        else 0
                    ),
                )
            )

    rounded_codes, selected = select_local_covariance_codes_from_error(
        selection_input,
        error,
        delta,
        nearest_codes,
        adjacent_codes,
        group_size=cfg.weight_rounding_group_size,
        rounds_per_group=cfg.weight_rounding_rounds_per_group,
        offdiag_shrink=cfg.weight_rounding_offdiag_shrink,
        dense_main=(
            selection_dense_main
            if cfg.weight_rounding_joint_objective
            else None
        ),
        dense_weight=(
            dense_weight_2d
            if cfg.weight_rounding_joint_objective
            else None
        ),
    )
    if audit_input is not None:
        selected = enforce_crossfit_audit(
            audit_input,
            error,
            delta,
            selected,
            group_size=cfg.weight_rounding_group_size,
            max_updates_per_group=cfg.weight_rounding_rounds_per_group,
            max_regression_fraction=(
                cfg.weight_rounding_audit_max_regression_fraction
            ),
            min_relative_gain=(
                cfg.weight_rounding_audit_min_relative_gain
            ),
            dense_main=audit_dense_main,
            dense_weight=(
                dense_weight_2d if cfg.weight_rounding_combined_audit else None
            ),
        )
        rounded_codes = torch.where(
            selected,
            adjacent_codes,
            nearest_codes,
        )
    rounded_payload = pack_fp4_codes(rounded_codes)
    if cfg.weight_rounding_dgrad_consistency:
        if cache_owner is None:
            raise ValueError(
                "DGRAD-consistent weight rounding requires the weight qresult cache owner."
            )
        owner_quantizer = getattr(cache_owner, "_quantizer", None)
        column_quantizer = getattr(owner_quantizer, "_default_nvfp4_quantizer", None)
        if column_quantizer is None:
            raise ValueError(
                "DGRAD-consistent weight rounding requires the default NVFP4 weight quantizer."
            )
        column_quantizer = column_quantizer.copy()
        # TE's 2D NVFP4 weight quantizer currently produces the column view
        # only together with the identity (rowwise) view.  The temporary
        # identity payload is immediately discarded below.
        column_quantizer.set_usage(rowwise=True, columnwise=True)
        # This view is deliberately local to the current DP rank: the selected
        # moves depend on that rank's sampled activations, so no amax collective
        # is needed or desired here.
        column_quantizer.with_amax_reduction = False
        column_quantizer.amax_reduction_group = None
        rounded_target = dense_weight_2d.float() + torch.where(
            selected,
            delta,
            torch.zeros((), device=delta.device, dtype=delta.dtype),
        )
        rounded_column = column_quantizer(
            rounded_target.to(dtype=dense_weight_2d.dtype)
        )
        cache_owner.weight_rounding_data_t = rounded_column._columnwise_data
        cache_owner.weight_rounding_scale_t = rounded_column._columnwise_scale_inv
        cache_owner.weight_rounding_global_amax_col = rounded_column._amax_columnwise
        cache_owner._weight_rounding_column_quantizer = getattr(
            rounded_column,
            "_quantizer",
            column_quantizer,
        )
    elif cache_owner is not None:
        cache_owner.weight_rounding_data_t = None
        cache_owner.weight_rounding_scale_t = None
        cache_owner.weight_rounding_global_amax_col = None
        cache_owner._weight_rounding_column_quantizer = None
    if (
        cfg.weight_rounding_reuse_generation_payload
        and payload_cache_key is not None
        and can_cache
    ):
        setattr(
            cache_owner,
            "_weight_rounding_payload_state",
            (payload_cache_key, rounded_payload),
        )
        # The packed payload fully determines later FPROP calls for this
        # generation.  Do not retain the much larger FP32 error/delta state.
        setattr(cache_owner, "_weight_rounding_static_state", None)
    return rounded_payload
