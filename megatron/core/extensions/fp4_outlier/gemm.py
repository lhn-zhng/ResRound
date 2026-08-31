"""GEMM routing for the FP4 FPROP input-outlier recipe."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.distributed as dist
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.custom_recipes import quantization

from .config import get_config
from .fast_fprop import try_add_sparse_correction_inplace
from .runtime import log_rank0_once
from .sparse import compute_sparse_correction
from .storage import make_default_nvfp4_storage, make_primary_nvfp4_storage
from .tensor import OutlierAwareNVFP4TensorRef
from .weight_rounding import maybe_round_fprop_weight_data, unpack_fp4_codes


_ALIGNMENT_MODULE_IDS: dict[tuple, int] = {}
_ALIGNMENT_MODULE_OCCURRENCES: dict[int, int] = {}
_ALIGNMENT_RECORDS: dict[tuple, list[dict]] = {}
_ALIGNMENT_PENDING_GRAD_OUTPUTS: list[dict] = []
_ALIGNMENT_PREVIOUS_ROW_MASKS: dict[tuple[int, int], torch.Tensor] = {}
_ALIGNMENT_ROW_SCORE_EMA: dict[tuple[int, int], torch.Tensor] = {}
_ALIGNMENT_ROW_SCORE_EMA_STEPS: dict[tuple[int, int], int] = {}
_ALIGNMENT_ROW_STEP_DOT: dict[tuple[int, int], torch.Tensor] = {}
_ALIGNMENT_ROW_STEP_POSITIVE: dict[tuple[int, int], torch.Tensor] = {}
_ALIGNMENT_ROW_STEP_NONZERO: dict[tuple[int, int], torch.Tensor] = {}
_ALIGNMENT_ROW_STEP_COUNTS: dict[tuple[int, int], int] = {}
_FPROP_WEIGHT_MODULE_IDS: dict[tuple, int] = {}
_DENSE_TRANSFORMER_LINEAR_MODULES_PER_LAYER = 4


def _weight_rounding_alignment_enabled() -> bool:
    return os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT", "0") == "1"


def _weight_rounding_alignment_logging_enabled() -> bool:
    return os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_LOG", "1") == "1"


def _loss_feedback_eval_uses_input_only(
    qresult_x: OutlierAwareNVFP4TensorRef,
) -> bool:
    # `tensor.requires_grad` is not a reliable train/eval discriminator here:
    # fused LayerNormLinear creates its normalized activation inside a custom
    # autograd forward, so QKV/FC1 inputs can report requires_grad=False during
    # training. Megatron's TE wrappers capture the outer forward grad mode
    # before entering that custom Function and attach it to the quantizer.
    forward_grad_enabled = getattr(qresult_x, "forward_grad_enabled", None)
    if forward_grad_enabled is None:
        # Compatibility fallback for direct quantizer use outside the Megatron
        # TE wrappers.
        forward_grad_enabled = bool(
            getattr(qresult_x, "source_requires_grad", False)
        )
    generic_eval_fallback = (
        os.environ.get("FP4_WEIGHT_ROUNDING_EVAL_INPUT_ONLY", "0") == "1"
    )
    loss_feedback_eval_fallback = (
        os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EVAL_INPUT_ONLY",
            "1",
        )
        == "1"
        and os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_APPLY",
            "0",
        )
        == "1"
    )
    return (
        (generic_eval_fallback or loss_feedback_eval_fallback)
        and not bool(forward_grad_enabled)
    )


def _weight_rounding_alignment_targets() -> set[int]:
    raw = os.environ.get(
        "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_MODULES",
        "0,2,33,35,63,65",
    )
    return {int(value) for value in raw.split(",") if value.strip()}


def _alignment_weight_key(qresult_w: OutlierAwareNVFP4TensorRef) -> tuple:
    dense_weight = qresult_w.dense_ref
    return (
        int(dense_weight.untyped_storage().data_ptr()),
        int(dense_weight.storage_offset()),
        tuple(dense_weight.shape),
        tuple(dense_weight.stride()),
    )


def _alignment_module_id(qresult_w: OutlierAwareNVFP4TensorRef) -> int:
    key = _alignment_weight_key(qresult_w)
    module_id = _ALIGNMENT_MODULE_IDS.get(key)
    if module_id is None:
        module_id = len(_ALIGNMENT_MODULE_IDS)
        _ALIGNMENT_MODULE_IDS[key] = module_id
    return module_id


def _fprop_weight_module_and_layer(
    qresult_w: OutlierAwareNVFP4TensorRef,
) -> tuple[int, int]:
    """Assign stable FPROP module/layer IDs in transformer execution order.

    Dense GPT/LLaMA transformer layers visit four TE linear weights in order:
    QKV, attention projection, FC1, and FC2. The registry is local to a model
    process and keyed by the master-weight storage, so repeated microbatches
    reuse exactly the same IDs.
    """
    key = _alignment_weight_key(qresult_w)
    module_id = _FPROP_WEIGHT_MODULE_IDS.get(key)
    if module_id is None:
        module_id = len(_FPROP_WEIGHT_MODULE_IDS)
        _FPROP_WEIGHT_MODULE_IDS[key] = module_id
        if os.environ.get("FP4_WEIGHT_ROUNDING_LOG_LAYER_MAP", "0") == "1":
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print(
                    "FP4_WEIGHT_ROUNDING_LAYER_MAP "
                    + json.dumps(
                        {
                            "module_id": module_id,
                            "layer_id": (
                                module_id
                                // _DENSE_TRANSFORMER_LINEAR_MODULES_PER_LAYER
                            ),
                            "slot": (
                                module_id
                                % _DENSE_TRANSFORMER_LINEAR_MODULES_PER_LAYER
                            ),
                            "weight_shape": tuple(qresult_w.dense_ref.shape),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return (
        module_id,
        module_id // _DENSE_TRANSFORMER_LINEAR_MODULES_PER_LAYER,
    )


def _weight_rounding_enabled_for_layer(
    layer_id: int,
    *,
    layer_start: int,
    layer_end: int,
) -> bool:
    return layer_id >= layer_start and (layer_end < 0 or layer_id < layer_end)


def _weight_rounding_enabled_for_module(
    layer_id: int,
    module_slot: int,
    *,
    layer_start: int,
    layer_end: int,
    qkv_layer_end: int,
    proj_layer_end: int,
    fc1_layer_end: int,
) -> bool:
    if not _weight_rounding_enabled_for_layer(
        layer_id,
        layer_start=layer_start,
        layer_end=layer_end,
    ):
        return False
    module_layer_ends = {
        0: qkv_layer_end,
        1: proj_layer_end,
        2: fc1_layer_end,
    }
    module_layer_end = module_layer_ends.get(module_slot, -1)
    return module_layer_end < 0 or layer_id < module_layer_end


def _alignment_module_and_occurrence(
    qresult_w: OutlierAwareNVFP4TensorRef,
) -> tuple[tuple, int, int]:
    key = _alignment_weight_key(qresult_w)
    module_id = _alignment_module_id(qresult_w)
    occurrence = _ALIGNMENT_MODULE_OCCURRENCES.get(module_id, 0)
    _ALIGNMENT_MODULE_OCCURRENCES[module_id] = occurrence + 1
    return key, module_id, occurrence


def _chunked_alignment_stats(
    grad_output: torch.Tensor,
    output_delta: torch.Tensor,
) -> dict[str, float]:
    grad_flat = grad_output.reshape(-1)
    delta_flat = output_delta.reshape(-1)
    if grad_flat.numel() != delta_flat.numel():
        raise ValueError(
            "Weight-rounding gradient-alignment shape mismatch: "
            f"grad={tuple(grad_output.shape)}, delta={tuple(output_delta.shape)}."
        )
    dot = torch.zeros((), device=grad_flat.device, dtype=torch.float64)
    grad_square = torch.zeros_like(dot)
    delta_square = torch.zeros_like(dot)
    positive = torch.zeros((), device=grad_flat.device, dtype=torch.int64)
    chunk = int(
        os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_CHUNK_ELEMENTS",
            str(4 * 1024 * 1024),
        )
    )
    for start in range(0, grad_flat.numel(), chunk):
        end = min(start + chunk, grad_flat.numel())
        grad_chunk = grad_flat[start:end].float()
        delta_chunk = delta_flat[start:end].float()
        product = grad_chunk * delta_chunk
        dot += product.double().sum()
        grad_square += grad_chunk.double().square().sum()
        delta_square += delta_chunk.double().square().sum()
        positive += (product > 0).sum()
    cosine = dot / (grad_square.sqrt() * delta_square.sqrt()).clamp_min(1.0e-30)
    return {
        "first_order_dot": float(dot.item()),
        "cosine": float(cosine.item()),
        "positive_product_fraction": float(positive.item()) / max(1, grad_flat.numel()),
        "grad_rms": float((grad_square / max(1, grad_flat.numel())).sqrt().item()),
        "delta_rms": float((delta_square / max(1, delta_flat.numel())).sqrt().item()),
    }


def _sample_alignment_output_delta(
    output_delta: torch.Tensor,
    *,
    output_features: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Optionally retain only evenly spaced tokens for row-level diagnostics."""
    if os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_PREDICT", "0") != "1":
        return output_delta, None
    if output_delta.numel() % output_features:
        raise ValueError(
            "Weight-rounding output delta is not divisible by the weight rows: "
            f"delta={tuple(output_delta.shape)}, rows={output_features}."
        )
    output_2d = output_delta.reshape(-1, output_features)
    requested = int(
        os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_TOKENS",
            "128",
        )
    )
    sample_count = min(max(1, requested), output_2d.shape[0])
    if sample_count == output_2d.shape[0]:
        indices = torch.arange(
            output_2d.shape[0],
            device=output_delta.device,
            dtype=torch.int64,
        )
    else:
        indices = torch.div(
            (2 * torch.arange(sample_count, device=output_delta.device) + 1)
            * output_2d.shape[0],
            2 * sample_count,
            rounding_mode="floor",
        )
    return output_2d.index_select(0, indices).contiguous(), indices


def _row_alignment_prediction_stats(
    grad_output: torch.Tensor,
    output_delta: torch.Tensor,
    *,
    module_id: int,
    rank: int,
) -> dict[str, float | int]:
    """Measure whether the previous microbatch's row gate predicts this one."""
    grad_2d = grad_output.reshape(output_delta.shape)
    product = grad_2d.float() * output_delta.float()
    row_dot = product.sum(dim=0, dtype=torch.float64)
    nonzero = product != 0
    harmful_fraction = (product > 0).sum(dim=0).float() / nonzero.sum(
        dim=0
    ).clamp_min(1)
    max_harmful_fraction = float(
        os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_MAX_HARM_FRACTION",
            "0.5",
        )
    )
    instantaneous_mask = (row_dot < 0) & (
        harmful_fraction <= max_harmful_fraction
    )
    history_key = (module_id, rank)
    ema_decay = float(
        os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EMA_DECAY",
            "0.0",
        )
    )
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EMA_DECAY must be "
            f"in [0, 1), got {ema_decay}."
        )
    step_microbatches = max(
        1,
        int(
            os.environ.get(
                "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_STEP_MICROBATCHES",
                "1",
            )
        ),
    )
    if step_microbatches > 1 and ema_decay > 0.0:
        raise ValueError(
            "Step-aggregated BVR and row-score EMA are alternative temporal "
            "filters; enable only one."
        )
    ema_min_history = max(
        1,
        int(
            os.environ.get(
                "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EMA_MIN_HISTORY",
                "2",
            )
        ),
    )
    ema_margin = max(
        0.0,
        float(
            os.environ.get(
                "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_EMA_MARGIN",
                "0.0",
            )
        ),
    )
    ema_history = 0
    ema_kept_rows = int(instantaneous_mask.sum().item())
    if ema_decay > 0.0:
        # Normalize each row's directional derivative to [-1, 1] before
        # temporal aggregation.  Raw G·DeltaY magnitudes change with the loss
        # scale, gradient norm, and candidate perturbation amplitude, whereas
        # the normalized sign-consensus score is comparable across
        # microbatches.
        row_abs = product.abs().sum(dim=0, dtype=torch.float64)
        normalized_score = (
            row_dot / row_abs.clamp_min(torch.finfo(torch.float64).tiny)
        ).float()
        previous_ema = _ALIGNMENT_ROW_SCORE_EMA.get(history_key)
        previous_steps = _ALIGNMENT_ROW_SCORE_EMA_STEPS.get(history_key, 0)
        if previous_ema is None or previous_ema.shape != normalized_score.shape:
            score_ema = normalized_score
            ema_history = 1
        else:
            score_ema = previous_ema.mul(ema_decay).add(
                normalized_score,
                alpha=1.0 - ema_decay,
            )
            ema_history = previous_steps + 1
        _ALIGNMENT_ROW_SCORE_EMA[history_key] = score_ema.detach()
        _ALIGNMENT_ROW_SCORE_EMA_STEPS[history_key] = ema_history
        ema_mask = score_ema < -ema_margin
        if ema_history < ema_min_history:
            ema_mask = torch.zeros_like(ema_mask)
        current_mask = instantaneous_mask & ema_mask
        ema_kept_rows = int(current_mask.sum().item())
        log_rank0_once(
            "fprop_weight_rounding:bvr_row_ema",
            (
                "FP4 BVR row-score EMA active: decay=%.3f, "
                "min_history=%d, margin=%.6f"
            ),
            ema_decay,
            ema_min_history,
            ema_margin,
        )
    else:
        current_mask = instantaneous_mask
    previous_mask = _ALIGNMENT_PREVIOUS_ROW_MASKS.get(history_key)
    if previous_mask is None or previous_mask.shape != current_mask.shape:
        predicted_rows = 0
        predicted_dot = 0.0
        prediction_available = 0
    else:
        predicted_rows = int(previous_mask.sum().item())
        predicted_dot = float(row_dot[previous_mask].sum().item())
        prediction_available = 1
    step_history = 1
    step_mask_updated = 1
    if step_microbatches > 1:
        positive_count = (product > 0).sum(dim=0, dtype=torch.int64)
        nonzero_count = nonzero.sum(dim=0, dtype=torch.int64)
        accumulated_dot = _ALIGNMENT_ROW_STEP_DOT.get(history_key)
        if accumulated_dot is None or accumulated_dot.shape != row_dot.shape:
            accumulated_dot = row_dot.detach().clone()
            accumulated_positive = positive_count.detach().clone()
            accumulated_nonzero = nonzero_count.detach().clone()
            step_history = 1
        else:
            accumulated_dot = accumulated_dot.add(row_dot)
            accumulated_positive = _ALIGNMENT_ROW_STEP_POSITIVE[
                history_key
            ].add(positive_count)
            accumulated_nonzero = _ALIGNMENT_ROW_STEP_NONZERO[
                history_key
            ].add(nonzero_count)
            step_history = _ALIGNMENT_ROW_STEP_COUNTS.get(history_key, 0) + 1
        _ALIGNMENT_ROW_STEP_DOT[history_key] = accumulated_dot
        _ALIGNMENT_ROW_STEP_POSITIVE[history_key] = accumulated_positive
        _ALIGNMENT_ROW_STEP_NONZERO[history_key] = accumulated_nonzero
        _ALIGNMENT_ROW_STEP_COUNTS[history_key] = step_history
        step_mask_updated = int(step_history >= step_microbatches)
        if step_mask_updated:
            accumulated_harmful_fraction = accumulated_positive.float() / (
                accumulated_nonzero.clamp_min(1)
            )
            current_mask = (accumulated_dot < 0) & (
                accumulated_harmful_fraction <= max_harmful_fraction
            )
            _ALIGNMENT_PREVIOUS_ROW_MASKS[history_key] = current_mask.detach()
            del _ALIGNMENT_ROW_STEP_DOT[history_key]
            del _ALIGNMENT_ROW_STEP_POSITIVE[history_key]
            del _ALIGNMENT_ROW_STEP_NONZERO[history_key]
            del _ALIGNMENT_ROW_STEP_COUNTS[history_key]
        elif previous_mask is None:
            current_mask = torch.zeros_like(instantaneous_mask)
        else:
            current_mask = previous_mask
        log_rank0_once(
            "fprop_weight_rounding:bvr_step_aggregate",
            (
                "FP4 BVR optimizer-step aggregation active: "
                "microbatches_per_step=%d"
            ),
            step_microbatches,
        )
    else:
        _ALIGNMENT_PREVIOUS_ROW_MASKS[history_key] = current_mask.detach()
    return {
        "row_prediction_available": prediction_available,
        "row_count": int(row_dot.numel()),
        "current_helpful_rows": int(current_mask.sum().item()),
        "current_total_dot": float(row_dot.sum().item()),
        "current_oracle_kept_dot": float(row_dot[current_mask].sum().item()),
        "previous_kept_rows": predicted_rows,
        "previous_mask_current_dot": predicted_dot,
        "max_harmful_fraction": max_harmful_fraction,
        "row_ema_enabled": int(ema_decay > 0.0),
        "row_ema_history": ema_history,
        "row_ema_kept_rows": ema_kept_rows,
        "row_ema_decay": ema_decay,
        "row_ema_margin": ema_margin,
        "row_step_microbatches": step_microbatches,
        "row_step_history": step_history,
        "row_step_mask_updated": step_mask_updated,
    }


def _apply_previous_alignment_row_mask(
    qresult_w: OutlierAwareNVFP4TensorRef,
    original_payload: torch.Tensor,
    candidate_payload: torch.Tensor,
    *,
    module_slot: int,
) -> tuple[torch.Tensor, bool]:
    """Apply the previous microbatch's task-gradient row decision."""
    if os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_APPLY", "0") != "1":
        return candidate_payload, False
    target_slots = {
        int(value)
        for value in os.environ.get(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_ROW_APPLY_SLOTS",
            "0,1,2",
        ).split(",")
        if value.strip()
    }
    if module_slot not in target_slots:
        return candidate_payload, False
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    module_id = _alignment_module_id(qresult_w)
    row_mask = _ALIGNMENT_PREVIOUS_ROW_MASKS.get((module_id, rank))
    if row_mask is None or row_mask.numel() != candidate_payload.shape[0]:
        if (
            int(
                os.environ.get(
                    "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_STEP_MICROBATCHES",
                    "1",
                )
            )
            > 1
        ):
            return original_payload, True
        return candidate_payload, False
    applied = torch.where(
        row_mask[:, None],
        candidate_payload,
        original_payload,
    ).contiguous()
    return applied, True


def _cross_rank_flip_stats(
    original_payload: torch.Tensor,
    rounded_payload: torch.Tensor,
) -> dict[str, float | int]:
    changed = (
        unpack_fp4_codes(original_payload) != unpack_fp4_codes(rounded_payload)
    ).to(torch.uint8)
    local_flips = int(changed.sum().item())
    intersection = changed.clone()
    union = changed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(intersection, op=dist.ReduceOp.MIN)
        dist.all_reduce(union, op=dist.ReduceOp.MAX)
    intersection_count = int(intersection.sum().item())
    union_count = int(union.sum().item())
    return {
        "local_flips": local_flips,
        "all_rank_intersection": intersection_count,
        "all_rank_union": union_count,
        "all_rank_jaccard": (
            float(intersection_count) / union_count if union_count else 1.0
        ),
    }


def _record_weight_rounding_alignment_fprop(
    *,
    qresult_w: OutlierAwareNVFP4TensorRef,
    unrounded_weight,
    rounded_weight,
    candidate_weight,
    input_storage,
    rounded_result: torch.Tensor,
    out_dtype: torch.dtype,
    use_split_accumulator: bool,
) -> None:
    if not _weight_rounding_alignment_enabled():
        return
    weight_key, module_id, occurrence = _alignment_module_and_occurrence(qresult_w)
    if module_id not in _weight_rounding_alignment_targets():
        return
    if (
        os.environ.get("FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT_DEBUG_PENDING", "0")
        == "1"
        and module_id == 0
    ):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print(
                "FP4_WEIGHT_ROUNDING_PENDING "
                + json.dumps(
                    {
                        "occurrence": occurrence,
                        "pending_count": len(_ALIGNMENT_PENDING_GRAD_OUTPUTS),
                        "pending_tail": [
                            {
                                "module_id": int(item["module_id"]),
                                "weight_shape": tuple(item["weight_shape"]),
                            }
                            for item in _ALIGNMENT_PENDING_GRAD_OUTPUTS[-8:]
                        ],
                        "full_record_count": sum(
                            len(items) for items in _ALIGNMENT_RECORDS.values()
                        ),
                        "cuda_allocated": (
                            int(torch.cuda.memory_allocated())
                            if torch.cuda.is_available()
                            else 0
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    with torch.no_grad():
        unrounded_result, *_ = general_gemm(
            unrounded_weight,
            input_storage,
            out_dtype=out_dtype,
            use_split_accumulator=use_split_accumulator,
        )
        if candidate_weight is rounded_weight:
            candidate_result = rounded_result.detach()
        else:
            candidate_result, *_ = general_gemm(
                candidate_weight,
                input_storage,
                out_dtype=out_dtype,
                use_split_accumulator=use_split_accumulator,
            )
        output_delta = (
            candidate_result.detach() - unrounded_result.detach()
        ).to(out_dtype)
        output_delta, token_indices = _sample_alignment_output_delta(
            output_delta,
            output_features=int(qresult_w.dense_ref.shape[0]),
        )
        flip_stats = (
            _cross_rank_flip_stats(
                unrounded_weight._rowwise_data,
                candidate_weight._rowwise_data,
            )
            if occurrence == 0 and _weight_rounding_alignment_logging_enabled()
            else {}
        )
    record = {
        "module_id": module_id,
        "occurrence": occurrence,
        "weight_shape": tuple(qresult_w.dense_ref.shape),
        "output_delta": output_delta,
        "token_indices": token_indices,
        **flip_stats,
    }
    # The full DGRAD-side diagnostic is only useful when verbose alignment
    # logging is enabled.  In row-feedback training the dense grad-output hook
    # below already consumes the sampled delta.  Retaining another reference in
    # _ALIGNMENT_RECORDS is both redundant and unsafe: row-parallel projection
    # weights do not revisit this table through the same storage key, so those
    # CUDA tensors would otherwise accumulate once per microbatch.
    if _weight_rounding_alignment_logging_enabled():
        records = _ALIGNMENT_RECORDS.setdefault(weight_key, [])
        records.append(record)
    _ALIGNMENT_PENDING_GRAD_OUTPUTS.append(record.copy())
    if occurrence == 0 and _weight_rounding_alignment_logging_enabled():
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(
            "FP4_WEIGHT_ROUNDING_FLIP_ALIGNMENT "
            + json.dumps(
                {
                    "rank": rank,
                    "module_id": module_id,
                    "occurrence": occurrence,
                    "weight_shape": tuple(qresult_w.dense_ref.shape),
                    **flip_stats,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _score_pending_alignment_record(
    record: dict,
    grad_output: torch.Tensor,
) -> bool:
    """Score one FPROP perturbation if this is its matching grad-output."""
    output_delta = record["output_delta"]
    token_indices = record.get("token_indices")
    output_features = int(record["weight_shape"][0])
    # Divisibility is not an identity check: QKV's feature count is commonly
    # an integer multiple of Proj's. Requiring the exact final dimension keeps
    # one module's grad-output from silently scoring another module's delta.
    if not grad_output.shape or grad_output.shape[-1] != output_features:
        return False
    if token_indices is not None:
        if grad_output.numel() % output_features:
            return False
        grad_output = grad_output.reshape(-1, output_features)
        # Matching the feature dimension is necessary but not sufficient:
        # two linear modules may have the same output width while using
        # different token counts. Reject a stale/mismatched record before the
        # CUDA gather so it can never turn a bookkeeping mismatch into a
        # device-side assertion.
        if token_indices.numel() and int(token_indices[-1].item()) >= grad_output.shape[0]:
            return False
        grad_output = grad_output.index_select(0, token_indices).contiguous()
    elif grad_output.numel() != output_delta.numel():
        return False
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"token_indices", "output_delta"}
    }
    alignment = (
        _chunked_alignment_stats(
            grad_output.detach(),
            output_delta,
        )
        if _weight_rounding_alignment_logging_enabled()
        else {}
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    row_prediction = (
        _row_alignment_prediction_stats(
            grad_output.detach(),
            output_delta,
            module_id=int(payload["module_id"]),
            rank=rank,
        )
        if token_indices is not None
        else {}
    )
    if _weight_rounding_alignment_logging_enabled():
        print(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT "
            + json.dumps(
                {"rank": rank, **payload, **alignment, **row_prediction},
                sort_keys=True,
            ),
            flush=True,
        )
    return True


def consume_weight_rounding_grad_alignment(grad_output: torch.Tensor) -> None:
    """Consume a saved FPROP perturbation when its dense grad-output is quantized."""
    if (
        not _weight_rounding_alignment_enabled()
        or not _ALIGNMENT_PENDING_GRAD_OUTPUTS
    ):
        return
    record = _ALIGNMENT_PENDING_GRAD_OUTPUTS[-1]
    if _score_pending_alignment_record(record, grad_output):
        _ALIGNMENT_PENDING_GRAD_OUTPUTS.pop()


def _log_weight_rounding_alignment_dgrad(
    *,
    qresult_w: OutlierAwareNVFP4TensorRef,
    grad_output_storage,
) -> None:
    if not _weight_rounding_alignment_enabled():
        return
    pending_record = (
        _ALIGNMENT_PENDING_GRAD_OUTPUTS[-1]
        if _ALIGNMENT_PENDING_GRAD_OUTPUTS
        else None
    )
    if pending_record is not None and tuple(pending_record["weight_shape"]) != tuple(
        qresult_w.dense_ref.shape
    ):
        pending_record = None
    records = _ALIGNMENT_RECORDS.get(_alignment_weight_key(qresult_w))
    if not records and pending_record is None:
        return
    with torch.no_grad():
        grad_output = grad_output_storage.dequantize(dtype=torch.float32)
        # Row-parallel Proj can bypass the dense grad-output quantizer hook.
        # DGRAD still has the correctly paired quantized grad-output, so consume
        # the pending record here before it can retain CUDA memory or block the
        # LIFO feedback stream.
        if pending_record is not None and _score_pending_alignment_record(
            pending_record,
            grad_output,
        ):
            _ALIGNMENT_PENDING_GRAD_OUTPUTS.pop()
        if not records:
            return
        record = records.pop()
        token_indices = record.pop("token_indices", None)
        if token_indices is not None:
            output_features = int(record["weight_shape"][0])
            grad_output = (
                grad_output.reshape(-1, output_features)
                .index_select(0, token_indices)
                .contiguous()
            )
        alignment = _chunked_alignment_stats(
            grad_output,
            record.pop("output_delta"),
        )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    payload = {"rank": rank, **record, **alignment}
    if _weight_rounding_alignment_logging_enabled():
        print(
            "FP4_WEIGHT_ROUNDING_GRAD_ALIGNMENT "
            + json.dumps(payload, sort_keys=True),
            flush=True,
        )


def finalize_gemm_result(
    result: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    accumulate: bool,
    bias: Optional[torch.Tensor] = None,
    correction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    needs_postprocess = correction is not None or bias is not None or accumulate
    target_dtype = out.dtype if out is not None else out_dtype
    if not needs_postprocess:
        result_cast = result if result.dtype == target_dtype else result.to(target_dtype)
        if out is not None:
            out.copy_(result_cast)
            return out
        return result_cast

    result_fp32 = result.to(torch.float32)
    if correction is not None:
        result_fp32 = result_fp32 + correction.to(torch.float32)
    if bias is not None:
        result_fp32 = result_fp32 + bias.to(device=result.device, dtype=torch.float32).view(1, -1)
    if accumulate:
        if out is None:
            raise ValueError("Output tensor must be provided when accumulate is True.")
        result_fp32 = result_fp32 + out.to(torch.float32)

    result_cast = result_fp32.to(target_dtype)
    if out is not None:
        out.copy_(result_cast)
        return out
    return result_cast


def run_default_qgemm(
    *,
    qresult_x: OutlierAwareNVFP4TensorRef,
    qresult_w: OutlierAwareNVFP4TensorRef,
    out_dtype: torch.dtype,
    use_split_accumulator: bool,
    gemm_type: quantization.GEMMType,
) -> torch.Tensor:
    if gemm_type == quantization.GEMMType.FPROP:
        default_x = make_default_nvfp4_storage(qresult_x, use_rowwise=True, use_columnwise=False)
        default_w = make_default_nvfp4_storage(qresult_w, use_rowwise=True, use_columnwise=False)
        result, *_ = general_gemm(
            default_w,
            default_x,
            out_dtype=out_dtype,
            use_split_accumulator=use_split_accumulator,
        )
        return result

    if gemm_type == quantization.GEMMType.DGRAD:
        default_dy = make_default_nvfp4_storage(qresult_x, use_rowwise=True, use_columnwise=False)
        _log_weight_rounding_alignment_dgrad(
            qresult_w=qresult_w,
            grad_output_storage=default_dy,
        )
        cfg = get_config()
        consistent_data_t = (
            getattr(qresult_w, "weight_rounding_data_t", None)
            if cfg.weight_rounding_dgrad_consistency
            else None
        )
        default_w = make_default_nvfp4_storage(
            qresult_w,
            use_rowwise=False,
            use_columnwise=True,
            columnwise_data_override=consistent_data_t,
            columnwise_scale_override=(
                getattr(qresult_w, "weight_rounding_scale_t", None)
                if consistent_data_t is not None
                else None
            ),
            columnwise_amax_override=(
                getattr(qresult_w, "weight_rounding_global_amax_col", None)
                if consistent_data_t is not None
                else None
            ),
            quantizer_override=(
                getattr(qresult_w, "_weight_rounding_column_quantizer", None)
                if consistent_data_t is not None
                else None
            ),
        )
        if consistent_data_t is not None:
            log_rank0_once(
                "fprop_weight_rounding:dgrad_consistent",
                "FP4 weight-rounding dual view active: DGRAD uses the cached columnwise view.",
            )
        result, *_ = general_gemm(
            default_w,
            default_dy,
            out_dtype=out_dtype,
            layout="NN",
            grad=True,
            use_split_accumulator=use_split_accumulator,
        )
        return result

    if gemm_type == quantization.GEMMType.WGRAD:
        default_dy = make_default_nvfp4_storage(qresult_x, use_rowwise=False, use_columnwise=True)
        default_x = make_default_nvfp4_storage(qresult_w, use_rowwise=False, use_columnwise=True)
        result, *_ = general_gemm(
            default_x,
            default_dy,
            out_dtype=out_dtype,
            layout="NT",
            grad=True,
            use_split_accumulator=use_split_accumulator,
        )
        return result

    raise ValueError(f"Unsupported GEMM type {gemm_type}.")


def run_fprop_input_outlier_qgemm(
    *,
    qresult_x: OutlierAwareNVFP4TensorRef,
    qresult_w: OutlierAwareNVFP4TensorRef,
    out_dtype: torch.dtype,
    use_split_accumulator: bool,
) -> torch.Tensor:
    cfg = get_config()
    main_x = make_primary_nvfp4_storage(qresult_x, use_rowwise=True, use_columnwise=False)
    main_w = make_primary_nvfp4_storage(qresult_w, use_rowwise=True, use_columnwise=False)
    unrounded_main_w = main_w
    module_id, layer_id = _fprop_weight_module_and_layer(qresult_w)
    module_slot = module_id % _DENSE_TRANSFORMER_LINEAR_MODULES_PER_LAYER
    layer_rounding_enabled = _weight_rounding_enabled_for_module(
        layer_id,
        module_slot,
        layer_start=cfg.weight_rounding_layer_start,
        layer_end=cfg.weight_rounding_layer_end,
        qkv_layer_end=cfg.weight_rounding_qkv_layer_end,
        proj_layer_end=cfg.weight_rounding_proj_layer_end,
        fc1_layer_end=cfg.weight_rounding_fc1_layer_end,
    )
    rounded_weight_data = (
        maybe_round_fprop_weight_data(
            main_x,
            main_w,
            qresult_w.dense_ref,
            dense_main=getattr(qresult_x, "dense_main", None),
            cache_owner=qresult_w,
        )
        if (
            cfg.enable_weight_rounding
            and layer_rounding_enabled
            and not _loss_feedback_eval_uses_input_only(qresult_x)
        )
        else None
    )
    if (
        cfg.weight_rounding_combined_audit
        or cfg.weight_rounding_joint_objective
    ):
        # The unquantized dense main is needed only while selecting the shared
        # FPROP weight payload. Do not retain it in TE's saved activation state.
        qresult_x.dense_main = None
    if rounded_weight_data is not None:
        candidate_main_w = make_primary_nvfp4_storage(
            qresult_w,
            use_rowwise=True,
            use_columnwise=False,
            rowwise_data_override=rounded_weight_data,
        )
        applied_weight_data, row_mask_applied = _apply_previous_alignment_row_mask(
            qresult_w,
            unrounded_main_w._rowwise_data,
            rounded_weight_data,
            module_slot=module_slot,
        )
        main_w = (
            make_primary_nvfp4_storage(
                qresult_w,
                use_rowwise=True,
                use_columnwise=False,
                rowwise_data_override=applied_weight_data,
            )
            if row_mask_applied
            else candidate_main_w
        )
        log_rank0_once(
            "fprop_weight_rounding:active",
            (
                "FP4 ephemeral local-covariance weight rounding active: "
                "group_size=%d, rounds_per_group=%d, selection_tokens=%d, "
                "stratified_sampling=%s, stratified_batch_size=%d, "
                "crossfit_audit=%s, audit_tokens=%d, "
                "combined_audit=%s, joint_objective=%s, "
                "audit_max_regression_fraction=%.3f, "
                "audit_min_relative_gain=%.3f, "
                "offdiag_shrink=%.3f, expansion_only=%s, "
                "min_expansion_ratio=%.3f, max_expansion_ratio=%.3f, "
                "layer_start=%d, layer_end=%d, "
                "qkv_layer_end=%d, proj_layer_end=%d, fc1_layer_end=%d, "
                "reuse_generation_payload=%s, dgrad_consistency=%s"
            ),
            cfg.weight_rounding_group_size,
            cfg.weight_rounding_rounds_per_group,
            cfg.weight_rounding_selection_tokens,
            cfg.weight_rounding_stratified_sampling,
            cfg.weight_rounding_stratified_batch_size,
            cfg.weight_rounding_crossfit_audit,
            cfg.weight_rounding_audit_tokens,
            cfg.weight_rounding_combined_audit,
            cfg.weight_rounding_joint_objective,
            cfg.weight_rounding_audit_max_regression_fraction,
            cfg.weight_rounding_audit_min_relative_gain,
            cfg.weight_rounding_offdiag_shrink,
            cfg.weight_rounding_expansion_only,
            cfg.weight_rounding_min_expansion_ratio,
            cfg.weight_rounding_max_expansion_ratio,
            cfg.weight_rounding_layer_start,
            cfg.weight_rounding_layer_end,
            cfg.weight_rounding_qkv_layer_end,
            cfg.weight_rounding_proj_layer_end,
            cfg.weight_rounding_fc1_layer_end,
            cfg.weight_rounding_reuse_generation_payload,
            cfg.weight_rounding_dgrad_consistency,
        )
    result, *_ = general_gemm(
        main_w,
        main_x,
        out_dtype=out_dtype,
        use_split_accumulator=use_split_accumulator,
    )
    if rounded_weight_data is not None:
        _record_weight_rounding_alignment_fprop(
            qresult_w=qresult_w,
            unrounded_weight=unrounded_main_w,
            rounded_weight=main_w,
            candidate_weight=candidate_main_w,
            input_storage=main_x,
            rounded_result=result,
            out_dtype=out_dtype,
            use_split_accumulator=use_split_accumulator,
        )
    return result


def run_qgemm(
    *,
    qresult_x: OutlierAwareNVFP4TensorRef,
    qresult_w: OutlierAwareNVFP4TensorRef,
    m_params: quantization.MMParams,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    accumulate: bool,
    gemm_type: quantization.GEMMType,
) -> torch.Tensor:
    cfg = get_config()
    if gemm_type != quantization.GEMMType.FPROP or not cfg.enable_fprop:
        result = run_default_qgemm(
            qresult_x=qresult_x,
            qresult_w=qresult_w,
            out_dtype=out_dtype,
            use_split_accumulator=m_params.use_split_accumulator,
            gemm_type=gemm_type,
        )
        return finalize_gemm_result(
            result,
            out_dtype=out_dtype,
            out=out,
            accumulate=accumulate,
            bias=bias if gemm_type == quantization.GEMMType.FPROP else None,
        )

    result = run_fprop_input_outlier_qgemm(
        qresult_x=qresult_x,
        qresult_w=qresult_w,
        out_dtype=out_dtype,
        use_split_accumulator=m_params.use_split_accumulator,
    )
    if cfg.enable_fast_fprop and try_add_sparse_correction_inplace(
        result,
        qresult_x=qresult_x,
        qresult_w=qresult_w,
    ):
        return finalize_gemm_result(
            result,
            out_dtype=out_dtype,
            out=out,
            accumulate=accumulate,
            bias=bias,
        )

    correction = compute_sparse_correction(
        qresult_x,
        qresult_w.dense_ref,
        transpose_dense=True,
    )
    return finalize_gemm_result(
        result,
        out_dtype=out_dtype,
        out=out,
        accumulate=accumulate,
        bias=bias,
        correction=correction,
    )
