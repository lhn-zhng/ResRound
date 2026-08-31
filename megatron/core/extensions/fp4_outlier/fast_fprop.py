"""Optional fast kernels for the FP4 FPROP input-outlier path."""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from .runtime import log_rank0_once
from .split import OutlierRatio, has_values


@dataclasses.dataclass
class FastSelectQuantResult:
    """r207 select+quant output in the qtensor shape expected by storage.py."""

    _rowwise_data: torch.Tensor
    _rowwise_scale_inv: torch.Tensor
    _amax_rowwise: torch.Tensor
    _columnwise_data: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _amax_columnwise: Optional[torch.Tensor]
    _fp4_dtype: Any
    _quantizer: NVFP4Quantizer
    outlier_rows: Optional[torch.Tensor]
    outlier_cols: Optional[torch.Tensor]
    outlier_values: Optional[torch.Tensor]
    outlier_flat_indices: Optional[torch.Tensor]
    outlier_row_offsets: Optional[torch.Tensor]
    dense_main: Optional[torch.Tensor]
    effective_ratio: float
    actual_ratio: float
    heaviness: Optional[float]
    selected_nnz: int
    overflow: int
    requested_capacity: int
    payload_capacity: int
    backend: str
    columnwise_source: str
    active_rows: Optional[torch.Tensor] = None
    active_row_count: int = 0
    active_rows_num_rows: int = 0
    sparse_policy_hint: Optional[str] = None
    full_capacity_flat_indices: Optional[torch.Tensor] = None
    full_capacity_values: Optional[torch.Tensor] = None
    full_capacity_cols: Optional[torch.Tensor] = None
    local_count_hint: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return self._rowwise_data.device


def _repo_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.getenv("FP4_OUTLIER_FAST_KERNEL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "megatron").is_dir():
            candidates.append(parent)
            break
    for parent in current.parents:
        if parent.name == "Megatron-LM-312-fprop-input":
            candidates.append(parent.with_name("Megatron-LM-312"))
            break
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _find_existing_path(relative_path: str) -> Optional[Path]:
    for root in _repo_root_candidates():
        path = root / relative_path
        if path.is_file():
            return path
    return None


def _load_module_from_path(module_name: str, path: Path) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_r207_module() -> Any:
    path = _find_existing_path(
        "profile/r203_dense_sparse_fusion_report_20260629/"
        "kernels/rowcol_quant_packed_r207/python/adaptive_rowcol_quant_fast.py"
    )
    if path is None:
        raise RuntimeError("r207 select+quant module not found")
    return _load_module_from_path("megatron_fp4_outlier_r207_select_quant", path)


@lru_cache(maxsize=1)
def _load_sparse_fusion_module() -> Any:
    path = _find_existing_path(
        "collected/nvfp4_warpgroup_sparse_fusion/python/"
        "nvfp4_warpgroup_sparse_fusion.py"
    )
    if path is None:
        raise RuntimeError("nvfp4 sparse fusion module not found")
    return _load_module_from_path("megatron_fp4_outlier_sparse_fusion", path)


@lru_cache(maxsize=None)
def _training_rht_mask_t(device_index: int) -> int:
    with torch.cuda.device(device_index):
        quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=True,
        )
        return int(quantizer.rht_matrix_random_sign_mask_t)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _shape_key(rows: int, cols: int) -> str:
    return f"{int(rows)}x{int(cols)}"


def _env_shape_float(
    name: str,
    *,
    rows: Optional[int],
    cols: Optional[int],
    default: float,
) -> float:
    if rows is None or cols is None:
        return float(default)
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    key = _shape_key(rows, cols)
    for item in value.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        item_key, item_value = item.split(":", 1)
        if item_key.strip().lower() == key:
            return float(item_value.strip())
    return float(default)


def _sortk_fill_threads(cols: int, requested: int) -> int:
    if os.getenv("FP4_OUTLIER_FAST_FPROP_SORTK_PAYLOAD", "1") != "1":
        return int(requested)
    blocks_per_row = (int(cols) + 15) // 16
    if blocks_per_row <= 128:
        return max(int(requested), 128)
    if blocks_per_row <= 256:
        return max(int(requested), 256)
    return max(int(requested), 512)


def _capacity_from_ratio(
    numel: int,
    ratio: float,
    *,
    shape: Optional[tuple[int, int]] = None,
) -> int:
    rows, cols = shape if shape is not None else (None, None)
    floor = _env_float("FP4_OUTLIER_FAST_FPROP_CAPACITY_RATIO_FLOOR", 0.02)
    multiplier = _env_float("FP4_OUTLIER_FAST_FPROP_CAPACITY_MULTIPLIER", 20.0)
    max_ratio = _env_float("FP4_OUTLIER_FAST_FPROP_MAX_CAPACITY_RATIO", 0.20)
    floor = _env_shape_float(
        "FP4_OUTLIER_FAST_FPROP_CAPACITY_RATIO_FLOOR_BY_SHAPE",
        rows=rows,
        cols=cols,
        default=floor,
    )
    multiplier = _env_shape_float(
        "FP4_OUTLIER_FAST_FPROP_CAPACITY_MULTIPLIER_BY_SHAPE",
        rows=rows,
        cols=cols,
        default=multiplier,
    )
    max_ratio = _env_shape_float(
        "FP4_OUTLIER_FAST_FPROP_MAX_CAPACITY_RATIO_BY_SHAPE",
        rows=rows,
        cols=cols,
        default=max_ratio,
    )
    cap_ratio = max(float(ratio) * float(multiplier), float(floor))
    cap_ratio = min(float(max_ratio), cap_ratio)
    return max(1024, int(math.ceil(float(numel) * cap_ratio)))


def _sparse_correction_policy(*, rows: int, cols: int, out_cols: int) -> str:
    policy = os.getenv("FP4_OUTLIER_FAST_FPROP_SPARSE_POLICY", "auto").strip().lower()
    aliases = {
        "direct": "direct_poststore",
        "direct_poststore": "direct_poststore",
        "poststore": "direct_poststore",
        "r25": "direct_poststore",
        "r25_poststore": "direct_poststore",
        "dense_light_direct_hot": "dense_light_direct_hot",
        "light_direct_hot": "dense_light_direct_hot",
        "report_dense_light_direct_hot": "dense_light_direct_hot",
        "dense_light_padded_hot": "dense_light_padded_hot",
        "padded_dense_light_direct_hot": "dense_light_padded_hot",
        "fc2_hot_columns": "fc2_hot_columns",
        "hot_columns": "fc2_hot_columns",
    }
    if policy in aliases:
        return aliases[policy]
    if policy not in {"", "auto"}:
        log_rank0_once(
            "fast_fprop:sparse:unknown_policy",
            "Unknown FP4 fast FPROP sparse policy %r; using auto.",
            policy,
        )
    if (
        _env_bool("FP4_OUTLIER_FAST_FPROP_SPARSE_AUTO_PROJ_DENSE_LIGHT_HOT", False)
        and int(rows) == 16384
        and int(cols) == 2048
        and int(out_cols) == 4096
    ):
        return "dense_light_padded_hot"
    return "direct_poststore"


def _direct_sparse_variant(
    *,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    out_cols: Optional[int] = None,
) -> str:
    variant = os.getenv(
        "FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT", "auto"
    ).strip().lower()
    if variant in {"", "auto"}:
        return "vstore"
    if variant in {
        "vstore",
        "strict_vstore",
        "sum_then_add",
        "vec16",
        "col_shmem_sum_then_add",
        "shmem_sum_then_add",
    }:
        return variant
    log_rank0_once(
        "fast_fprop:sparse:unknown_direct_variant",
        "Unknown FP4 fast FPROP direct sparse variant %r; using vstore.",
        variant,
    )
    return "vstore"


def _apply_direct_sparse_correction(
    fusion: Any,
    result: torch.Tensor,
    row_payload: Any,
    weight_t: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    variant = _direct_sparse_variant(
        rows=int(result.shape[0]),
        cols=int(k),
        out_cols=int(result.shape[1]),
    )
    if variant == "vec16":
        return fusion.sparse_active_row_col_value_payload_vec16_inplace(
            result,
            row_payload,
            weight_t,
            active_rows,
            k=k,
        )
    if variant in {"col_shmem_sum_then_add", "shmem_sum_then_add"}:
        return fusion.sparse_active_row_col_value_payload_vec8_shmem_sum_then_add(
            result,
            row_payload,
            weight_t,
            active_rows,
            k=k,
        )
    if variant == "sum_then_add":
        return fusion.sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore(
            result,
            row_payload,
            weight_t,
            active_rows,
            k=k,
            flat_indices=flat_indices,
        )
    if variant == "strict_vstore":
        return fusion.sparse_active_row_value_payload_vec8_inplace_strict_vstore(
            result,
            row_payload,
            weight_t,
            active_rows,
            k=k,
            flat_indices=flat_indices,
        )
    return fusion.sparse_active_row_value_payload_vec8_inplace_vstore(
        result,
        row_payload,
        weight_t,
        active_rows,
        k=k,
        flat_indices=flat_indices,
    )


def _keep_full_capacity_cols_for_sparse_policy(
    *,
    rows: int,
    cols: int,
    policy_hint: Optional[str] = None,
) -> bool:
    if policy_hint in {
        "dense_light_direct_hot",
        "dense_light_padded_hot",
        "fc2_hot_columns",
    }:
        return True
    direct_variant = os.getenv(
        "FP4_OUTLIER_FAST_FPROP_DIRECT_SPARSE_VARIANT", "vstore"
    ).strip().lower()
    if direct_variant in {"col_shmem_sum_then_add", "shmem_sum_then_add"}:
        return True
    policy = os.getenv("FP4_OUTLIER_FAST_FPROP_SPARSE_POLICY", "auto").strip().lower()
    if policy in {"direct", "direct_poststore", "poststore", "r25", "r25_poststore"}:
        return False
    if policy in {
        "dense_light_direct_hot",
        "light_direct_hot",
        "report_dense_light_direct_hot",
        "dense_light_padded_hot",
        "padded_dense_light_direct_hot",
        "fc2_hot_columns",
        "hot_columns",
    }:
        return True
    return (
        _env_bool("FP4_OUTLIER_FAST_FPROP_SPARSE_AUTO_PROJ_DENSE_LIGHT_HOT", False)
        and int(rows) == 16384
        and int(cols) == 2048
    )


def _uses_cached_sparse_policy() -> bool:
    policy = os.getenv("FP4_OUTLIER_FAST_FPROP_SPARSE_POLICY", "auto").strip().lower()
    return policy in {"", "auto", "adaptive_cached", "cached_auto"} and _env_bool(
        "FP4_OUTLIER_FAST_FPROP_SPARSE_AUTO_CACHED", True
    )


def _cache_weight_t_enabled(qresult_w: Any) -> bool:
    globally_enabled = _env_bool("FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T", False)
    configured_layers = os.getenv(
        "FP4_OUTLIER_FAST_FPROP_CACHE_WEIGHT_T_LAYERS", ""
    ).strip().lower()
    if not globally_enabled and not configured_layers:
        return False
    if not globally_enabled:
        owner = getattr(qresult_w, "_quantizer", qresult_w)
        layer_name = str(getattr(owner, "layer_name", "")).lower()
        if not any(
            token.strip() and token.strip() in layer_name
            for token in configured_layers.split(",")
        ):
            return False
    if getattr(qresult_w, "weight_update_generation", None) is None:
        log_rank0_once(
            "fast_fprop:sparse:weight_t_cache_missing_generation",
            (
                "FP4 fast FPROP weight-T cache requested without a TE weight-workspace "
                "generation; bypassing cache to avoid stale weights."
            ),
        )
        return False
    return True


def _choose_cached_sparse_policy(
    *,
    owner: Any,
    rows: int,
    cols: int,
    actual_ratio: float,
) -> str:
    layer_name = str(getattr(owner, "layer_name", "")).lower()
    min_ratio = _env_float(
        "FP4_OUTLIER_FAST_FPROP_SPARSE_PADDED_HYBRID_MIN_ACTUAL_RATIO", 0.003
    )
    if (
        "linear_proj" in layer_name
        and int(rows) == 16384
        and int(cols) == 2048
        and float(actual_ratio) >= min_ratio
    ):
        return "dense_light_padded_hot"
    if (
        "linear_fc2" in layer_name
        and str(getattr(owner, "outlier_selection_method", "")).lower()
        == "normal_threshold"
        and int(rows) == 16384
        and int(cols) == 7168
        and float(actual_ratio) >= 0.005
    ):
        return "fc2_hot_columns"
    return "direct_poststore"


def _fc2_hot_column_state(
    qresult_x: Any,
    row_ks: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner = getattr(qresult_x, "_quantizer", qresult_x)
    configured_hot_cols = _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_COLS", 0)
    base_hot_cols = min(k, max(1, _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_BASE", 1024)))
    max_hot_cols = min(
        k,
        max(base_hot_cols, _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_MAX", 2048)),
    )
    hot_step = max(1, _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_STEP", 256))
    chunk_nnz_threshold = max(
        1,
        _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_CHUNK_NNZ", 85000),
    )
    refresh_interval = max(
        0,
        _env_int("FP4_OUTLIER_FAST_FPROP_FC2_HOT_REFRESH_INTERVAL", 64),
    )
    device_index = int(row_ks.device.index or torch.cuda.current_device())
    key = (
        k,
        configured_hot_cols,
        base_hot_cols,
        max_hot_cols,
        hot_step,
        chunk_nnz_threshold,
        device_index,
    )
    state = getattr(owner, "_fast_fprop_fc2_hot_column_state", None)
    calls = 0 if state is None or state[0] != key else int(state[3])
    capturing = torch.cuda.is_current_stream_capturing()
    refresh = state is None or state[0] != key or (
        refresh_interval > 0 and calls % refresh_interval == 0
    )
    if capturing and state is not None and state[0] == key:
        refresh = False
    if refresh:
        frequencies = torch.bincount(row_ks.to(torch.int64), minlength=k)
        if configured_hot_cols > 0 or capturing:
            hot_cols = min(k, configured_hot_cols if configured_hot_cols > 0 else 1536)
            hot_ids = torch.topk(
                frequencies,
                k=hot_cols,
                sorted=False,
            ).indices.contiguous()
        else:
            ranked_ids = torch.topk(
                frequencies,
                k=max_hot_cols,
                sorted=True,
            ).indices
            chunk_totals = []
            for chunk_start in range(base_hot_cols, max_hot_cols, hot_step):
                chunk_end = min(max_hot_cols, chunk_start + hot_step)
                chunk_totals.append(frequencies[ranked_ids[chunk_start:chunk_end]].sum())
            hot_cols = base_hot_cols
            if chunk_totals:
                for chunk_total in torch.stack(chunk_totals).detach().cpu().tolist():
                    if int(chunk_total) < chunk_nnz_threshold:
                        break
                    hot_cols = min(max_hot_cols, hot_cols + hot_step)
            hot_ids = ranked_ids[:hot_cols].contiguous()
        hot_lut = torch.full(
            (k,),
            -1,
            device=row_ks.device,
            dtype=torch.int16,
        )
        hot_lut[hot_ids] = torch.arange(
            hot_cols,
            device=row_ks.device,
            dtype=torch.int16,
        )
    else:
        hot_ids, hot_lut = state[1], state[2]
    setattr(
        owner,
        "_fast_fprop_fc2_hot_column_state",
        (key, hot_ids, hot_lut, calls + 1),
    )
    return hot_ids, hot_lut


def _read_selected_overflow(
    selected_t: torch.Tensor,
    overflow_t: torch.Tensor,
    active_count_t: Optional[torch.Tensor] = None,
) -> tuple[int, int, int]:
    selected_flat = selected_t.reshape(-1)
    if _env_bool("FP4_OUTLIER_FAST_FPROP_ASSUME_NO_OVERFLOW", False):
        if active_count_t is None:
            return int(selected_flat.detach().cpu()[0]), 0, 0
        meta = torch.stack(
            (
                selected_flat[0].to(torch.int64),
                active_count_t.reshape(-1)[0].to(torch.int64),
            )
        ).detach().cpu()
        return int(meta[0]), 0, int(meta[1])

    overflow_flat = overflow_t.reshape(-1)
    meta_tensors = [
        selected_flat[0].to(torch.int64),
        overflow_flat[0].to(torch.int64),
    ]
    if active_count_t is not None:
        meta_tensors.append(active_count_t.reshape(-1)[0].to(torch.int64))
    meta = torch.stack(tuple(meta_tensors)).detach().cpu()
    active_count = int(meta[2]) if active_count_t is not None else 0
    return int(meta[0]), int(meta[1]), active_count


def try_fast_select_quantize(
    tensor_2d: torch.Tensor,
    *,
    owner,
    ratio: OutlierRatio,
    selection_method: str,
    store_dense_main: bool,
) -> Optional[FastSelectQuantResult]:
    """Run r207 threshold select+quant when the requested semantics are compatible."""

    if selection_method != "normal_threshold":
        log_rank0_once(
            "fast_fprop:select:unsupported_method",
            "FP4 fast FPROP select+quant supports normal_threshold only; falling back.",
        )
        return None
    if ratio.effective <= 0.0:
        return None
    if (
        tensor_2d.dtype != torch.bfloat16
        or tensor_2d.device.type != "cuda"
        or tensor_2d.ndim != 2
    ):
        log_rank0_once(
            "fast_fprop:select:unsupported_tensor",
            "FP4 fast FPROP select+quant requires CUDA BF16 2D input; falling back.",
        )
        return None
    rows, cols = int(tensor_2d.shape[0]), int(tensor_2d.shape[1])
    if cols % 16 != 0:
        log_rank0_once(
            "fast_fprop:select:unsupported_shape",
            "FP4 fast FPROP select+quant requires K divisible by 16; falling back.",
        )
        return None

    try:
        r207 = _load_r207_module()
        capacity = _capacity_from_ratio(
            int(tensor_2d.numel()),
            float(ratio.effective),
            shape=(rows, cols),
        )
        if getattr(owner, "_fast_fprop_capacity_hint_key", None) == (rows, cols):
            capacity = max(
                capacity,
                int(getattr(owner, "_fast_fprop_capacity_hint", 0)),
            )
        rht_mask_t = _training_rht_mask_t(
            int(tensor_2d.device.index or torch.cuda.current_device())
        )
        columnwise_source = os.getenv(
            "FP4_OUTLIER_FAST_FPROP_COLUMNWISE_SOURCE",
            "direct",
        )
        trust_capacity = _env_bool("FP4_OUTLIER_FAST_FPROP_TRUST_CAPACITY", False)
        cached_sparse_policy_enabled = _uses_cached_sparse_policy()
        cached_sparse_policy_key = getattr(
            owner, "_fast_fprop_cached_sparse_policy_key", None
        )
        cached_sparse_policy = (
            getattr(owner, "_fast_fprop_cached_sparse_policy", None)
            if cached_sparse_policy_enabled
            and cached_sparse_policy_key == (rows, cols)
            else None
        )
        cached_sparse_policy_needs_sample = (
            cached_sparse_policy_enabled and cached_sparse_policy is None
        )
        defer_selected_sync_requested = _env_bool(
            "FP4_OUTLIER_FAST_FPROP_DEFER_SELECTED_SYNC", False
        )
        reuse_sorted_active_rows = _env_bool(
            "FP4_OUTLIER_FAST_FPROP_REUSE_ACTIVE_ROWS", False
        )
        reuse_unsorted_active_rows = _env_bool(
            "FP4_OUTLIER_FAST_FPROP_REUSE_UNSORTED_ACTIVE_ROWS", False
        ) or (
            cached_sparse_policy_enabled
            and cached_sparse_policy != "dense_light_padded_hot"
        )
        if reuse_sorted_active_rows and reuse_unsorted_active_rows:
            reuse_sorted_active_rows = False
            log_rank0_once(
                "fast_fprop:select:prefer_unsorted_active_rows",
                (
                    "Both sorted and unsorted active-row reuse were requested; "
                    "using the lower-cost unsorted path."
                ),
            )
        reuse_active_rows = reuse_sorted_active_rows or reuse_unsorted_active_rows
        defer_selected_sync = (
            defer_selected_sync_requested
            and trust_capacity
            and _env_bool("FP4_OUTLIER_FAST_FPROP_ASSUME_NO_OVERFLOW", False)
            and not reuse_sorted_active_rows
            and not cached_sparse_policy_needs_sample
        )
        if defer_selected_sync_requested and not defer_selected_sync:
            log_rank0_once(
                "fast_fprop:select:defer_disabled",
                (
                    "FP4 fast FPROP selected-count sync deferral requires "
                    "TRUST_CAPACITY=1, ASSUME_NO_OVERFLOW=1, no sorted active-row reuse, "
                    "and no pending cached-policy sample; "
                    "using one combined metadata read."
                ),
            )
        quant_fn = (
            r207.adaptive_rowcol_quant_fast
            if trust_capacity
            else r207.adaptive_rowcol_quant_fast_safe
        )
        direct_nomask = (
            _env_bool("FP4_OUTLIER_FAST_FPROP_DIRECT_NOMASK", False)
            and columnwise_source == "direct"
            and not bool(store_dense_main)
        )
        output = quant_fn(
            tensor_2d,
            base_ratio=float(ratio.effective),
            min_ratio=float(ratio.effective),
            max_ratio=float(ratio.effective),
            reference_heaviness=1.0,
            capacity=capacity,
            capacity_multiplier=1.0,
            min_capacity=1024,
            emit_dense_main=bool(store_dense_main),
            stats_threads=_env_int("FP4_OUTLIER_FAST_FPROP_STATS_THREADS", 128),
            fill_threads=_sortk_fill_threads(
                cols,
                _env_int("FP4_OUTLIER_FAST_FPROP_FILL_THREADS", 64),
            ),
            columnwise_source=columnwise_source,
            rht_random_sign_mask_t=rht_mask_t,
            overlap_columnwise=_env_bool(
                "FP4_OUTLIER_FAST_FPROP_OVERLAP_COLUMNWISE",
                columnwise_source not in {"direct", "te_default_direct"},
            ),
            direct_nomask=direct_nomask,
            build_active_schedule=reuse_sorted_active_rows,
            build_unsorted_active_rows=reuse_unsorted_active_rows,
            threshold_sigma_override=_env_shape_float(
                "FP4_OUTLIER_FAST_FPROP_THRESHOLD_SIGMA_BY_SHAPE",
                rows=rows,
                cols=cols,
                default=-1.0,
            ),
        )
        flat_full = output[0].to(torch.int32).contiguous()
        values_full = output[1].contiguous()
        if defer_selected_sync:
            selected = -1
            overflow = 0
            active_row_count = 0
        else:
            selected, overflow, active_row_count = _read_selected_overflow(
                output[4],
                output[5],
                output[15] if reuse_sorted_active_rows else None,
            )
        selected_known = selected >= 0
        has_sparse_payload = (not selected_known) or selected > 0
        if cached_sparse_policy_enabled and selected_known:
            capacity_headroom = max(
                1.0,
                _env_float("FP4_OUTLIER_FAST_FPROP_CAPACITY_HINT_HEADROOM", 1.5),
            )
            capacity_hint = min(
                int(tensor_2d.numel()),
                max(capacity, int(math.ceil(float(selected) * capacity_headroom))),
            )
            setattr(owner, "_fast_fprop_capacity_hint", capacity_hint)
            setattr(owner, "_fast_fprop_capacity_hint_key", (rows, cols))
        if trust_capacity and selected_known and selected > int(flat_full.numel()):
            log_rank0_once(
                "fast_fprop:select:trusted_capacity_overflow",
                (
                    "FP4 fast FPROP trusted capacity was too small; "
                    "selected=%d, capacity=%d, falling back."
                ),
                selected,
                int(flat_full.numel()),
            )
            return None
        if overflow != 0:
            log_rank0_once(
                "fast_fprop:select:overflow",
                "FP4 fast FPROP select+quant overflowed capacity; falling back.",
            )
            return None

        if defer_selected_sync:
            flat = flat_full
            values = values_full
        else:
            flat = flat_full.narrow(0, 0, selected)
            values = values_full.narrow(0, 0, selected)
        payload_has_fixed_capacity = int(flat_full.numel()) == int(capacity)
        packed_sparse_ag_payload = (
            has_sparse_payload
            and payload_has_fixed_capacity
            and _env_bool("FP4_OUTLIER_FAST_FPROP_PACKED_SPARSE_AG_PAYLOAD", False)
            and bool(getattr(owner, "enable_sparse_all_gather", False))
        )
        rows_i32 = None
        cols_i32 = None
        emit_coo = _env_bool("FP4_OUTLIER_FAST_FPROP_EMIT_COO", False)
        if selected > 0 and emit_coo:
            cols_i32 = (flat % cols).to(torch.int32).contiguous()
            rows_i32 = torch.div(flat, cols, rounding_mode="floor").to(torch.int32).contiguous()
        row_offsets = output[3].to(torch.int32).contiguous()
        dense_main_tensor = output[10]
        dense_main = (
            dense_main_tensor
            if bool(store_dense_main)
            and dense_main_tensor is not None
            and dense_main_tensor.numel() != 0
            else None
        )
        stats = output[9].detach()
        heaviness = None
        effective_ratio = float(ratio.effective)
        if _env_bool("FP4_OUTLIER_FAST_FPROP_READ_STATS", False) and int(stats.numel()) >= 4:
            try:
                stats_cpu = stats.cpu()
                heaviness = float(stats_cpu[2])
                effective_ratio = float(stats_cpu[3])
            except RuntimeError:
                heaviness = ratio.heaviness
        active_rows = None
        if len(output) >= 16 and reuse_unsorted_active_rows and has_sparse_payload:
            active_row_count = -1
            active_rows = output[14].to(torch.int32).contiguous()
        elif selected > 0 and len(output) >= 16 and reuse_sorted_active_rows:
            try:
                active_rows = output[14][:active_row_count].to(torch.int32).contiguous()
            except RuntimeError:
                active_row_count = 0
                active_rows = None

        payload_capacity = int(flat_full.numel()) if has_sparse_payload else 0
        actual_ratio = (
            float(selected) / float(max(1, int(tensor_2d.numel())))
            if selected_known
            else float("nan")
        )
        sparse_policy_hint = cached_sparse_policy
        if cached_sparse_policy_needs_sample and selected_known:
            sparse_policy_hint = _choose_cached_sparse_policy(
                owner=owner,
                rows=rows,
                cols=cols,
                actual_ratio=actual_ratio,
            )
            setattr(owner, "_fast_fprop_cached_sparse_policy", sparse_policy_hint)
            setattr(owner, "_fast_fprop_cached_sparse_policy_key", (rows, cols))
            log_rank0_once(
                f"fast_fprop:sparse:cached_policy:{getattr(owner, 'layer_name', id(owner))}",
                (
                    "FP4 fast FPROP cached sparse policy initialized: layer=%s, "
                    "shape=%s, actual_ratio=%.6f, policy=%s"
                ),
                getattr(owner, "layer_name", "unknown"),
                tuple(tensor_2d.shape),
                actual_ratio,
                sparse_policy_hint,
            )
        log_rank0_once(
            "fast_fprop:select:active",
            (
                "FP4 fast FPROP select+quant active: backend=r207, shape=%s, "
                "selected=%s, defer_selected_sync=%s, emit_coo=%s, packed_sparse_ag=%s, "
                "requested_capacity=%d, payload_capacity=%d"
            ),
            tuple(tensor_2d.shape),
            str(selected) if selected_known else "deferred",
            defer_selected_sync,
            emit_coo,
            packed_sparse_ag_payload,
            int(capacity),
            payload_capacity,
        )
        return FastSelectQuantResult(
            _rowwise_data=output[6],
            _rowwise_scale_inv=output[7],
            _amax_rowwise=output[8],
            _columnwise_data=output[11],
            _columnwise_scale_inv=output[12],
            _amax_columnwise=output[13],
            _fp4_dtype=owner._main_nvfp4_quantizer.dtype,
            _quantizer=owner._main_nvfp4_quantizer,
            outlier_rows=rows_i32 if selected > 0 else None,
            outlier_cols=cols_i32 if selected > 0 else None,
            outlier_values=values if has_sparse_payload else None,
            outlier_flat_indices=flat if has_sparse_payload else None,
            outlier_row_offsets=row_offsets if has_sparse_payload else None,
            dense_main=dense_main,
            effective_ratio=effective_ratio,
            actual_ratio=actual_ratio,
            heaviness=heaviness,
            selected_nnz=selected,
            overflow=overflow,
            requested_capacity=int(capacity),
            payload_capacity=payload_capacity,
            backend="r207_normal_threshold",
            columnwise_source=columnwise_source,
            active_rows=active_rows,
            active_row_count=active_row_count,
            active_rows_num_rows=rows,
            sparse_policy_hint=sparse_policy_hint,
            full_capacity_flat_indices=flat_full
            if (defer_selected_sync or packed_sparse_ag_payload)
            else None,
            full_capacity_values=values_full
            if (defer_selected_sync or packed_sparse_ag_payload)
            else None,
            full_capacity_cols=output[2]
            if _keep_full_capacity_cols_for_sparse_policy(
                rows=rows,
                cols=cols,
                policy_hint=sparse_policy_hint,
            )
            else None,
            local_count_hint=output[4]
            if (defer_selected_sync or packed_sparse_ag_payload)
            else None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log_rank0_once(
            f"fast_fprop:select:fallback:{type(exc).__name__}",
            "FP4 fast FPROP select+quant unavailable (%s); falling back.",
            str(exc),
        )
        return None


def try_add_sparse_correction_inplace(
    result: torch.Tensor,
    *,
    qresult_x,
    qresult_w,
) -> bool:
    """Add A2 @ W into an existing BF16 dense GEMM output with the report sparse kernel."""

    if _env_bool("FP4_OUTLIER_FAST_FPROP_DISABLE_DIRECT_SPARSE", False):
        log_rank0_once(
            "fast_fprop:sparse:disabled_by_env",
            "FP4 fast FPROP direct sparse correction disabled by environment; falling back.",
        )
        return False

    values = getattr(qresult_x, "outlier_values", None)
    flat = getattr(qresult_x, "outlier_flat_indices", None)
    row_offsets = getattr(qresult_x, "outlier_row_offsets", None)
    dense_weight = getattr(qresult_w, "dense_ref", None)
    if not has_values(values):
        return True
    if flat is None or row_offsets is None or dense_weight is None:
        return False
    if result.dtype != torch.bfloat16 or result.device.type != "cuda" or result.ndim != 2:
        log_rank0_once(
            "fast_fprop:sparse:unsupported_output",
            "FP4 fast FPROP sparse correction requires CUDA BF16 2D output; falling back.",
        )
        return False
    if dense_weight.dtype != torch.bfloat16 or dense_weight.device != result.device:
        log_rank0_once(
            "fast_fprop:sparse:unsupported_weight",
            "FP4 fast FPROP sparse correction requires BF16 dense weight on output device; falling back.",
        )
        return False

    try:
        fusion = _load_sparse_fusion_module()
        n = int(result.shape[1])
        weight_n, k = int(dense_weight.shape[0]), int(dense_weight.shape[1])
        if weight_n != n:
            log_rank0_once(
                "fast_fprop:sparse:shape_mismatch",
                "FP4 fast FPROP sparse correction shape mismatch; falling back.",
            )
            return False
        m = int(result.shape[0])
        flat = flat.to(device=result.device, dtype=torch.int32).contiguous()
        row_offsets = row_offsets.to(device=result.device, dtype=torch.int32).contiguous()
        values = values.to(device=result.device, dtype=torch.bfloat16).contiguous()
        policy = getattr(qresult_x, "_fast_fprop_sparse_policy_hint", None)
        if policy not in {
            "direct_poststore",
            "dense_light_direct_hot",
            "dense_light_padded_hot",
            "fc2_hot_columns",
        }:
            policy = _sparse_correction_policy(rows=m, cols=k, out_cols=n)
        direct_variant = _direct_sparse_variant(rows=m, cols=k, out_cols=n)
        if policy == "fc2_hot_columns":
            row_ks = getattr(qresult_x, "_fast_fprop_row_ks_i16", None)
            if row_ks is None or int(row_ks.numel()) < int(values.numel()):
                full_cols = getattr(qresult_x, "_fast_fprop_full_capacity_cols", None)
                if full_cols is None or int(full_cols.numel()) < int(values.numel()):
                    return False
                row_ks = full_cols.to(device=result.device, dtype=torch.int16).contiguous()
                if int(row_ks.numel()) != int(values.numel()):
                    row_ks = row_ks.narrow(0, 0, int(values.numel()))
                setattr(qresult_x, "_fast_fprop_row_ks_i16", row_ks)
        elif policy in {"dense_light_direct_hot", "dense_light_padded_hot"}:
            row_ks = getattr(qresult_x, "_fast_fprop_row_ks_i32", None)
            if row_ks is None or int(row_ks.numel()) < int(values.numel()):
                full_cols = getattr(qresult_x, "_fast_fprop_full_capacity_cols", None)
                if (
                    full_cols is not None
                    and int(full_cols.numel()) >= int(values.numel())
                ):
                    row_ks = full_cols.to(device=result.device, dtype=torch.int32).contiguous()
                    if int(row_ks.numel()) != int(values.numel()):
                        row_ks = row_ks.narrow(0, 0, int(values.numel()))
                else:
                    row_ks = torch.remainder(flat, k).to(torch.int32).contiguous()
                setattr(qresult_x, "_fast_fprop_row_ks_i32", row_ks)
        elif direct_variant in {
            "vec16",
            "col_shmem_sum_then_add",
            "shmem_sum_then_add",
        }:
            row_ks = getattr(qresult_x, "_fast_fprop_row_ks_i16", None)
            if row_ks is None or int(row_ks.numel()) < int(values.numel()):
                full_cols = getattr(qresult_x, "_fast_fprop_full_capacity_cols", None)
                if (
                    full_cols is not None
                    and int(full_cols.numel()) >= int(values.numel())
                ):
                    row_ks = full_cols.to(device=result.device, dtype=torch.int16).contiguous()
                    if int(row_ks.numel()) != int(values.numel()):
                        row_ks = row_ks.narrow(0, 0, int(values.numel()))
                else:
                    row_ks = torch.remainder(flat, k).to(torch.int16).contiguous()
                setattr(qresult_x, "_fast_fprop_row_ks_i16", row_ks)
        else:
            row_ks = torch.empty((0,), device=result.device, dtype=torch.int32)
        row_payload = fusion.build_row_indexed_payload(
            row_offsets,
            row_ks,
            values,
            selected_count=int(values.numel()),
            target_ratio=float(getattr(qresult_x, "effective_outlier_ratio", 0.0) or 0.0),
        )
        if _cache_weight_t_enabled(qresult_w):
            cache_key = (
                int(qresult_w.weight_update_generation),
                int(dense_weight.untyped_storage().data_ptr()),
                int(dense_weight.storage_offset()),
                int(getattr(dense_weight, "_version", -1)),
                tuple(dense_weight.shape),
                tuple(dense_weight.stride()),
                int(dense_weight.device.index or torch.cuda.current_device()),
            )
            cached_weight_t = getattr(qresult_w, "_fast_fprop_weight_t_state", None)
            if (
                cached_weight_t is not None
                and cached_weight_t[0] == cache_key
                and not torch.cuda.is_current_stream_capturing()
            ):
                weight_t = cached_weight_t[1]
            else:
                if (
                    cached_weight_t is not None
                    and tuple(cached_weight_t[1].shape)
                    == (int(dense_weight.shape[1]), int(dense_weight.shape[0]))
                    and cached_weight_t[1].device == dense_weight.device
                    and cached_weight_t[1].dtype == dense_weight.dtype
                ):
                    weight_t = cached_weight_t[1]
                    weight_t.copy_(dense_weight.t())
                else:
                    weight_t = dense_weight.t().contiguous()
                setattr(
                    qresult_w,
                    "_fast_fprop_weight_t_state",
                    (cache_key, weight_t),
                )
        else:
            weight_t = dense_weight.t().contiguous()
        if policy == "fc2_hot_columns":
            hot_ids, hot_lut = _fc2_hot_column_state(qresult_x, row_ks, k=k)
            hot_cols = int(hot_ids.numel())
            cold_capacity = max(
                1,
                _env_int("FP4_OUTLIER_FAST_FPROP_FC2_COLD_CAPACITY", 256),
            )
            hot_dense = torch.empty(
                (m, hot_cols),
                device=result.device,
                dtype=torch.bfloat16,
            )
            cold_values = torch.empty(
                (m, cold_capacity),
                device=result.device,
                dtype=torch.bfloat16,
            )
            cold_cols = torch.empty(
                (m, cold_capacity),
                device=result.device,
                dtype=torch.int16,
            )
            cold_counts = torch.empty(
                (m,),
                device=result.device,
                dtype=torch.int32,
            )
            split_overflow = torch.empty(
                (),
                device=result.device,
                dtype=torch.int32,
            )
            hot_weight = torch.empty(
                (hot_cols, n),
                device=result.device,
                dtype=torch.bfloat16,
            )
            fusion.split_hot_dense_padded_cold_rows(
                hot_dense,
                cold_values,
                cold_cols,
                cold_counts,
                split_overflow,
                row_payload,
                hot_lut,
            )
            torch.index_select(weight_t, 0, hot_ids, out=hot_weight)
            torch.addmm(result, hot_dense, hot_weight, beta=1.0, alpha=1.0, out=result)
            fusion.sparse_padded_cold_col_vec16_inplace(
                result,
                cold_values,
                cold_cols,
                cold_counts,
                row_payload,
                hot_lut,
                weight_t,
            )
            log_rank0_once(
                "fast_fprop:sparse:active:fc2_hot_columns",
                (
                    "FP4 fast FPROP sparse correction active: "
                    "backend=fc2_hot_columns, output=%s, hot_cols=%d, cold_capacity=%d"
                ),
                tuple(result.shape),
                hot_cols,
                cold_capacity,
            )
        elif policy == "dense_light_direct_hot":
            split_key = (
                "dense_light_direct_hot",
                int(row_offsets.numel()),
                _env_int("FP4_OUTLIER_FAST_FPROP_SPARSE_HEAVY_THRESHOLD", 64),
            )
            cached_split = getattr(qresult_x, "_fast_fprop_sparse_row_split", None)
            if cached_split is not None and cached_split[0] == split_key:
                light_rows, heavy_rows = cached_split[1], cached_split[2]
            else:
                counts = row_offsets[1:] - row_offsets[:-1]
                heavy_threshold = split_key[2]
                heavy_rows = (
                    torch.nonzero(counts >= heavy_threshold, as_tuple=False)
                    .flatten()
                    .to(torch.int32)
                    .contiguous()
                )
                light_rows = (
                    torch.nonzero((counts > 0) & (counts < heavy_threshold), as_tuple=False)
                    .flatten()
                    .to(torch.int32)
                    .contiguous()
                )
                setattr(qresult_x, "_fast_fprop_sparse_row_split", (split_key, light_rows, heavy_rows))
            if int(light_rows.numel()) != 0:
                _apply_direct_sparse_correction(
                    fusion,
                    result,
                    row_payload,
                    weight_t,
                    light_rows,
                    k=k,
                    flat_indices=flat,
                )
            if int(heavy_rows.numel()) != 0:
                scratch_key = (
                    int(heavy_rows.numel()),
                    k,
                    n,
                    int(result.device.index or torch.cuda.current_device()),
                )
                cached_scratch = getattr(qresult_x, "_fast_fprop_dense_light_hot_scratch", None)
                if cached_scratch is not None and cached_scratch[0] == scratch_key:
                    dense_residual, hot_dense_out = cached_scratch[1], cached_scratch[2]
                else:
                    dense_residual = torch.empty(
                        (int(heavy_rows.numel()), k),
                        device=result.device,
                        dtype=torch.bfloat16,
                    )
                    hot_dense_out = torch.empty(
                        (int(heavy_rows.numel()), n),
                        device=result.device,
                        dtype=torch.bfloat16,
                    )
                    setattr(
                        qresult_x,
                        "_fast_fprop_dense_light_hot_scratch",
                        (scratch_key, dense_residual, hot_dense_out),
                    )
                fusion.build_compact_dense_residual_active_rows(
                    dense_residual,
                    row_payload,
                    heavy_rows,
                    k=k,
                )
                torch.mm(dense_residual, weight_t, out=hot_dense_out)
                fusion.merge_compact_delta_active_rows(result, hot_dense_out, heavy_rows)
            log_rank0_once(
                "fast_fprop:sparse:active:dense_light_direct_hot",
                (
                    "FP4 fast FPROP sparse correction active: "
                    "backend=dense_light_direct_hot, variant=%s, output=%s, light_rows=%d, heavy_rows=%d"
                ),
                direct_variant,
                tuple(result.shape),
                int(light_rows.numel()),
                int(heavy_rows.numel()),
            )
        elif policy == "dense_light_padded_hot":
            heavy_threshold = _env_int(
                "FP4_OUTLIER_FAST_FPROP_SPARSE_HEAVY_THRESHOLD", 64
            )
            heavy_capacity = min(
                m,
                max(
                    1,
                    _env_int(
                        "FP4_OUTLIER_FAST_FPROP_SPARSE_PADDED_HEAVY_CAPACITY", 512
                    ),
                ),
            )
            split_key = (
                "dense_light_padded_hot",
                int(row_offsets.numel()),
                heavy_threshold,
                heavy_capacity,
            )
            cached_split = getattr(qresult_x, "_fast_fprop_sparse_row_split", None)
            if cached_split is not None and cached_split[0] == split_key:
                light_rows, heavy_rows = cached_split[1], cached_split[2]
            else:
                split = fusion.build_padded_light_heavy_rows(
                    row_offsets,
                    heavy_threshold=heavy_threshold,
                    heavy_capacity=heavy_capacity,
                )
                light_rows, heavy_rows = split[0], split[1]
                setattr(
                    qresult_x,
                    "_fast_fprop_sparse_row_split",
                    (split_key, light_rows, heavy_rows, split[2], split[3]),
                )
            _apply_direct_sparse_correction(
                fusion,
                result,
                row_payload,
                weight_t,
                light_rows,
                k=k,
                flat_indices=flat,
            )
            scratch_key = (
                "dense_light_padded_hot",
                heavy_capacity,
                k,
                n,
                int(result.device.index or torch.cuda.current_device()),
            )
            cached_scratch = getattr(qresult_x, "_fast_fprop_dense_light_hot_scratch", None)
            if cached_scratch is not None and cached_scratch[0] == scratch_key:
                dense_residual, hot_dense_out = cached_scratch[1], cached_scratch[2]
            else:
                dense_residual = torch.empty(
                    (heavy_capacity, k),
                    device=result.device,
                    dtype=torch.bfloat16,
                )
                hot_dense_out = torch.empty(
                    (heavy_capacity, n),
                    device=result.device,
                    dtype=torch.bfloat16,
                )
                setattr(
                    qresult_x,
                    "_fast_fprop_dense_light_hot_scratch",
                    (scratch_key, dense_residual, hot_dense_out),
                )
            fusion.build_compact_dense_residual_active_rows(
                dense_residual,
                row_payload,
                heavy_rows,
                k=k,
            )
            torch.mm(dense_residual, weight_t, out=hot_dense_out)
            fusion.merge_compact_delta_active_rows(result, hot_dense_out, heavy_rows)
            log_rank0_once(
                "fast_fprop:sparse:active:dense_light_padded_hot",
                (
                    "FP4 fast FPROP sparse correction active: "
                    "backend=dense_light_padded_hot, variant=%s, output=%s, "
                    "heavy_threshold=%d, heavy_capacity=%d"
                ),
                direct_variant,
                tuple(result.shape),
                heavy_threshold,
                heavy_capacity,
            )
        else:
            active_rows = getattr(qresult_x, "_fast_fprop_active_rows", None)
            active_rows_num_rows = getattr(qresult_x, "_fast_fprop_active_rows_num_rows", None)
            if (
                active_rows is not None
                and active_rows_num_rows is not None
                and int(active_rows_num_rows) + 1 == int(row_offsets.numel())
            ):
                active_rows = active_rows.to(device=result.device, dtype=torch.int32).contiguous()
            else:
                counts = row_offsets[1:] - row_offsets[:-1]
                active_rows = (
                    torch.nonzero(counts > 0, as_tuple=False)
                    .flatten()
                    .to(torch.int32)
                    .contiguous()
                )
                setattr(qresult_x, "_fast_fprop_active_rows", active_rows)
                setattr(qresult_x, "_fast_fprop_active_rows_num_rows", int(row_offsets.numel()) - 1)
            _apply_direct_sparse_correction(
                fusion,
                result,
                row_payload,
                weight_t,
                active_rows,
                k=k,
                flat_indices=flat,
            )
            log_rank0_once(
                "fast_fprop:sparse:active",
                "FP4 fast FPROP sparse correction active: backend=direct_poststore, variant=%s, output=%s",
                direct_variant,
                tuple(result.shape),
            )
        return True
    except Exception as exc:  # pylint: disable=broad-except
        log_rank0_once(
            f"fast_fprop:sparse:fallback:{type(exc).__name__}",
            "FP4 fast FPROP sparse correction unavailable (%s); falling back.",
            str(exc),
        )
        return False
