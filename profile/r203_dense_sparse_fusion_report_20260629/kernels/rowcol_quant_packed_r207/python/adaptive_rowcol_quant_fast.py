from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT_NAME = "nvfp4_sparse_comm_rowcol_quant_packed_r207"
_HARDCAP_HISTOGRAM_BINS = 2048
_PHASE_DIR = Path(__file__).resolve().parents[1]
_SRC_DIR = _PHASE_DIR / "src"
_REPO_ROOT = _PHASE_DIR.parents[3]
_TE_ROOT = Path(
    os.getenv("TRANSFORMER_ENGINE_ROOT", str(_REPO_ROOT.parent / "TransformerEngine"))
)


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        return None
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability()
        if major >= 12:
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
    build_directory = _PHASE_DIR / "build_adaptive_rowcol_quant_fast"
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=_EXT_NAME,
        sources=[
            str(_SRC_DIR / "adaptive_rowcol_quant_fast.cpp"),
            str(_SRC_DIR / "adaptive_rowcol_quant_fast.cu"),
        ],
        build_directory=str(build_directory),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        extra_include_paths=[
            str(_TE_ROOT / "transformer_engine/common/include"),
            str(_TE_ROOT / "transformer_engine"),
            str(_TE_ROOT / "transformer_engine/common"),
        ],
        extra_ldflags=[
            f"-L{_TE_ROOT}",
            "-L/usr/local/cuda/lib64",
            "-ltransformer_engine",
            "-lnvrtc",
            f"-Wl,-rpath,{_TE_ROOT}",
            "-Wl,-rpath,/usr/local/cuda/lib64",
        ],
        verbose=os.getenv("ADAPTIVE_ROWCOL_QUANT_FAST_VERBOSE", "0") == "1",
    )


def capacity_from_ratio(
    numel: int,
    ratio: float,
    *,
    multiplier: float = 2.0,
    min_capacity: int = 1024,
) -> int:
    if ratio < 0.0:
        raise ValueError("ratio must be non-negative")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    return max(int(numel * float(ratio) * float(multiplier) + 0.999999), int(min_capacity))


class AdaptiveRowcolWorkspace:
    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        capacity: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("adaptive rowcol quant extension is unavailable.")
        if dtype != torch.bfloat16:
            raise ValueError("workspace currently supports bfloat16 only")
        self.rows = int(rows)
        self.cols = int(cols)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        row_scale_outer = _round_up(self.rows, 128)
        row_scale_inner = _round_up((self.cols + 15) // 16, 4)
        column_scale_outer = _round_up(self.cols, 128)
        column_scale_inner = _round_up((self.rows + 15) // 16, 4)
        scan_temp_bytes = int(ext.adaptive_rowcol_scan_temp_bytes(self.rows))

        self.flat_indices = torch.empty((self.capacity,), device=self.device, dtype=torch.int32)
        self.outlier_values = torch.empty((self.capacity,), device=self.device, dtype=self.dtype)
        self.outlier_cols = torch.empty((self.capacity,), device=self.device, dtype=torch.int16)
        self.packed_records = torch.empty((self.capacity,), device=self.device, dtype=torch.int64)
        self.row_counts = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.row_offsets = torch.empty((self.rows + 1,), device=self.device, dtype=torch.int32)
        self.selection_masks = torch.empty(
            (self.rows, self.cols // 16), device=self.device, dtype=torch.int16
        )
        self.num_selected = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.overflow = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.stats_amax = torch.empty((10,), device=self.device, dtype=torch.float32)
        self.rowwise_data = torch.empty(
            (self.rows, self.cols // 2), device=self.device, dtype=torch.uint8
        )
        self.rowwise_scale = torch.empty(
            (row_scale_outer, row_scale_inner), device=self.device, dtype=torch.uint8
        )
        self.columnwise_data = torch.empty(
            (self.cols, self.rows // 2), device=self.device, dtype=torch.uint8
        )
        self.columnwise_scale = torch.empty(
            (column_scale_outer, column_scale_inner), device=self.device, dtype=torch.uint8
        )
        self.columnwise_amax = torch.empty((1,), device=self.device, dtype=torch.float32)
        self.rht_output_t = torch.empty((0,), device=self.device, dtype=self.dtype)
        self.scan_temp = torch.empty((scan_temp_bytes,), device=self.device, dtype=torch.uint8)
        active_sort_temp_bytes = int(ext.adaptive_rowcol_active_sort_temp_bytes(self.rows))
        self.active_rows_heavy_light = torch.empty(
            (self.rows,), device=self.device, dtype=torch.int32
        )
        self.active_row_count = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.active_sort_rows_in = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.active_sort_rows_out = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.active_sort_keys_in = torch.empty((self.rows,), device=self.device, dtype=torch.int64)
        self.active_sort_keys_out = torch.empty((self.rows,), device=self.device, dtype=torch.int64)
        self.active_sort_temp = torch.empty(
            (active_sort_temp_bytes,), device=self.device, dtype=torch.uint8
        )
        self.hardcap_score_max = torch.empty((1,), device=self.device, dtype=torch.float32)
        self.hardcap_histogram = torch.empty(
            (_HARDCAP_HISTOGRAM_BINS,), device=self.device, dtype=torch.int32
        )
        self.log_hist = torch.empty((256,), device=self.device, dtype=torch.int32)
        self.log_params = torch.empty((8,), device=self.device, dtype=torch.int64)
        self.last_overflow_fallback = False
        self.last_output_capacity = self.capacity
        self.last_selected_nnz = 0
        self.last_emit_packed_records = False

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.flat_indices,
            self.outlier_values,
            self.outlier_cols,
            self.row_counts,
            self.row_offsets,
            self.selection_masks,
            self.num_selected,
            self.overflow,
            self.stats_amax,
            self.rowwise_data,
            self.rowwise_scale,
            self.columnwise_data,
            self.columnwise_scale,
            self.columnwise_amax,
            self.rht_output_t,
            self.scan_temp,
        )

    def active_schedule_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.active_rows_heavy_light,
            self.active_row_count,
            self.active_sort_rows_in,
            self.active_sort_rows_out,
            self.active_sort_keys_in,
            self.active_sort_keys_out,
            self.active_sort_temp,
        )

    def hardcap_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hardcap_score_max, self.hardcap_histogram

    def packed_record_tensors(self) -> tuple[torch.Tensor]:
        return (self.packed_records,)

    def loghist_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_hist, self.log_params


class AdaptiveRowcolPaddedWorkspace:
    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        max_per_row: int,
        overflow_capacity: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        ext = _load_extension()
        if ext is None:
            raise RuntimeError("adaptive rowcol quant extension is unavailable.")
        if dtype != torch.bfloat16:
            raise ValueError("workspace currently supports bfloat16 only")
        if max_per_row < 0:
            raise ValueError("max_per_row must be non-negative")
        if overflow_capacity < 0:
            raise ValueError("overflow_capacity must be non-negative")
        self.rows = int(rows)
        self.cols = int(cols)
        self.max_per_row = int(max_per_row)
        self.overflow_capacity = int(overflow_capacity)
        self.device = torch.device(device)
        self.dtype = dtype

        row_scale_outer = _round_up(self.rows, 128)
        row_scale_inner = _round_up((self.cols + 15) // 16, 4)
        column_scale_outer = _round_up(self.cols, 128)
        column_scale_inner = _round_up((self.rows + 15) // 16, 4)

        self.padded_values = torch.empty(
            (self.rows, self.max_per_row), device=self.device, dtype=self.dtype
        )
        self.padded_cols = torch.empty(
            (self.rows, self.max_per_row), device=self.device, dtype=torch.int16
        )
        self.row_counts = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.selection_masks = torch.empty(
            (self.rows, self.cols // 16), device=self.device, dtype=torch.int16
        )
        self.num_selected = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.overflow = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.overflow_rows = torch.empty(
            (self.overflow_capacity,), device=self.device, dtype=torch.int32
        )
        self.overflow_cols = torch.empty(
            (self.overflow_capacity,), device=self.device, dtype=torch.int16
        )
        self.overflow_values = torch.empty(
            (self.overflow_capacity,), device=self.device, dtype=self.dtype
        )
        self.overflow_count = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.stats_amax = torch.empty((10,), device=self.device, dtype=torch.float32)
        self.rowwise_data = torch.empty(
            (self.rows, self.cols // 2), device=self.device, dtype=torch.uint8
        )
        self.rowwise_scale = torch.empty(
            (row_scale_outer, row_scale_inner), device=self.device, dtype=torch.uint8
        )
        self.columnwise_data = torch.empty(
            (self.cols, self.rows // 2), device=self.device, dtype=torch.uint8
        )
        self.columnwise_scale = torch.empty(
            (column_scale_outer, column_scale_inner), device=self.device, dtype=torch.uint8
        )
        self.columnwise_amax = torch.empty((1,), device=self.device, dtype=torch.float32)
        self.rht_output_t = torch.empty((0,), device=self.device, dtype=self.dtype)
        active_sort_temp_bytes = int(ext.adaptive_rowcol_active_sort_temp_bytes(self.rows))
        self.active_rows_heavy_light = torch.empty(
            (self.rows,), device=self.device, dtype=torch.int32
        )
        self.active_row_count = torch.empty((1,), device=self.device, dtype=torch.int32)
        self.active_sort_rows_in = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.active_sort_rows_out = torch.empty((self.rows,), device=self.device, dtype=torch.int32)
        self.active_sort_keys_in = torch.empty((self.rows,), device=self.device, dtype=torch.int64)
        self.active_sort_keys_out = torch.empty((self.rows,), device=self.device, dtype=torch.int64)
        self.active_sort_temp = torch.empty(
            (active_sort_temp_bytes,), device=self.device, dtype=torch.uint8
        )
        self.hardcap_score_max = torch.empty((1,), device=self.device, dtype=torch.float32)
        self.hardcap_histogram = torch.empty(
            (_HARDCAP_HISTOGRAM_BINS,), device=self.device, dtype=torch.int32
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.padded_values,
            self.padded_cols,
            self.row_counts,
            self.selection_masks,
            self.num_selected,
            self.overflow,
            self.overflow_rows,
            self.overflow_cols,
            self.overflow_values,
            self.overflow_count,
            self.stats_amax,
            self.rowwise_data,
            self.rowwise_scale,
            self.columnwise_data,
            self.columnwise_scale,
            self.columnwise_amax,
            self.rht_output_t,
        )

    def active_schedule_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.active_rows_heavy_light,
            self.active_row_count,
            self.active_sort_rows_in,
            self.active_sort_rows_out,
            self.active_sort_keys_in,
            self.active_sort_keys_out,
            self.active_sort_temp,
        )

    def hardcap_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hardcap_score_max, self.hardcap_histogram


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def adaptive_rowcol_quant_fast(
    tensor: torch.Tensor,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    capacity: int | None = None,
    capacity_multiplier: float = 2.0,
    min_capacity: int = 1024,
    emit_dense_main: bool = False,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    direct_nomask: bool = False,
    build_active_schedule: bool = False,
    build_unsorted_active_rows: bool = False,
    threshold_sigma_override: float = -1.0,
):
    """Adaptive CSR + rowwise NVFP4 + TE-RHT columnwise NVFP4.

    Hot-path assumptions:
      - BF16 input.
      - No hardcap/refine retry. The caller should pass 2x max_ratio capacity.

    Return order:
      flat_indices, outlier_values, outlier_cols, row_offsets, num_selected,
      overflow, rowwise_data, rowwise_scale, main_amax, stats, dense_main,
      columnwise_data, columnwise_scale, columnwise_amax,
      active_rows_heavy_light, active_row_count
    """

    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    if base_ratio <= 0.0:
        raise ValueError("base_ratio must be positive")
    source_to_mode = {
        "direct": 1,
        "te_default_direct": 1,
        "outlier_reuse": 2,
    }
    if columnwise_source not in source_to_mode:
        raise ValueError(
            f"unsupported columnwise_source={columnwise_source!r}; "
            f"expected one of {sorted(source_to_mode)}"
        )
    if direct_nomask and columnwise_source != "direct":
        raise ValueError("direct_nomask requires columnwise_source='direct'")
    if build_active_schedule and build_unsorted_active_rows:
        raise ValueError(
            "build_active_schedule and build_unsorted_active_rows are mutually exclusive"
        )
    if capacity is None:
        capacity = capacity_from_ratio(
            tensor.numel(),
            max_ratio,
            multiplier=capacity_multiplier,
            min_capacity=min_capacity,
        )
    return ext.adaptive_rowcol_quant_fast(
        tensor.contiguous(),
        float(base_ratio),
        float(min_ratio),
        float(max_ratio),
        float(reference_heaviness),
        int(capacity),
        bool(emit_dense_main),
        int(stats_threads),
        int(fill_threads),
        int(source_to_mode[columnwise_source]),
        int(rht_random_sign_mask_t),
        bool(overlap_columnwise),
        bool(direct_nomask),
        bool(build_active_schedule),
        bool(build_unsorted_active_rows),
        float(threshold_sigma_override),
    )


def adaptive_rowcol_quant_fast_safe(
    tensor: torch.Tensor,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    capacity: int | None = None,
    capacity_multiplier: float = 2.0,
    min_capacity: int = 1024,
    emit_dense_main: bool = False,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    direct_nomask: bool = False,
    build_active_schedule: bool = False,
    build_unsorted_active_rows: bool = False,
    threshold_sigma_override: float = -1.0,
):
    """Allocation-returning variant with exact CSR fallback.

    This convenience path reruns the full fused op with capacity=selected_nnz
    if the first allocation was too small. Prefer adaptive_rowcol_quant_fast_out_safe
    when a workspace is available; it reuses masks and row offsets on fallback.
    """

    payload = adaptive_rowcol_quant_fast(
        tensor,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        capacity=capacity,
        capacity_multiplier=capacity_multiplier,
        min_capacity=min_capacity,
        emit_dense_main=emit_dense_main,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        direct_nomask=direct_nomask,
        build_active_schedule=build_active_schedule,
        build_unsorted_active_rows=build_unsorted_active_rows,
        threshold_sigma_override=threshold_sigma_override,
    )
    if int(payload[5].detach().cpu()[0]) == 0:
        return payload

    selected_nnz = int(payload[4].detach().cpu()[0])
    return adaptive_rowcol_quant_fast(
        tensor,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        capacity=selected_nnz,
        emit_dense_main=emit_dense_main,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        direct_nomask=direct_nomask,
        build_active_schedule=build_active_schedule,
        build_unsorted_active_rows=build_unsorted_active_rows,
        threshold_sigma_override=threshold_sigma_override,
    )


def adaptive_rowcol_quant_fast_out(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    auto_expand_capacity: bool = False,
    emit_packed_records: bool = False,
    build_active_schedule: bool = True,
    emit_direct_split: bool = False,
    direct_policy_mode: int = -1,
    direct_param0: int = 0,
    direct_param1: int = 0,
    direct_use_bucket: bool = False,
    direct_no_host_slice: bool = False,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    source_to_mode = {
        "direct": 1,
        "te_default_direct": 1,
        "outlier_reuse": 2,
    }
    if columnwise_source not in source_to_mode:
        raise ValueError(
            f"unsupported columnwise_source={columnwise_source!r}; "
            f"expected one of {sorted(source_to_mode)}"
        )
    if tensor.size(0) != workspace.rows or tensor.size(1) != workspace.cols:
        raise ValueError("workspace shape does not match tensor")
    payload = ext.adaptive_rowcol_quant_fast_out(
        tensor.contiguous(),
        *workspace.tensors(),
        *workspace.packed_record_tensors(),
        *workspace.active_schedule_tensors(),
        *workspace.hardcap_tensors(),
        float(base_ratio),
        float(min_ratio),
        float(max_ratio),
        float(reference_heaviness),
        int(workspace.capacity),
        int(stats_threads),
        int(fill_threads),
        int(source_to_mode[columnwise_source]),
        int(rht_random_sign_mask_t),
        bool(overlap_columnwise),
        bool(auto_expand_capacity),
        bool(emit_packed_records),
        bool(build_active_schedule),
        bool(emit_direct_split),
        int(direct_policy_mode),
        int(direct_param0),
        int(direct_param1),
        bool(direct_use_bucket),
        bool(direct_no_host_slice),
    )
    workspace.last_overflow_fallback = bool(auto_expand_capacity and payload[0].numel() > workspace.capacity)
    workspace.last_output_capacity = int(payload[0].numel())
    workspace.last_emit_packed_records = bool(emit_packed_records)
    return payload


def adaptive_rowcol_quant_fast_direct_split_out(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    direct_policy_mode: int,
    direct_param0: int,
    direct_param1: int,
    direct_use_bucket: bool = False,
    direct_no_host_slice: bool = False,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    auto_expand_capacity: bool = False,
    build_active_schedule: bool = False,
):
    return adaptive_rowcol_quant_fast_out(
        tensor,
        workspace,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        auto_expand_capacity=auto_expand_capacity,
        emit_packed_records=False,
        build_active_schedule=build_active_schedule,
        emit_direct_split=True,
        direct_policy_mode=direct_policy_mode,
        direct_param0=direct_param0,
        direct_param1=direct_param1,
        direct_use_bucket=direct_use_bucket,
        direct_no_host_slice=direct_no_host_slice,
    )


def adaptive_rowcol_quant_fast_padded_out(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolPaddedWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    direct_nomask: bool = False,
):
    """Padded row-slot sparse payload + rowwise/columnwise NVFP4.

    Return order:
      padded_values, padded_cols, row_counts, num_selected, overflow,
      overflow_rows, overflow_cols, overflow_values, overflow_count,
      rowwise_data, rowwise_scale, main_amax, stats,
      columnwise_data, columnwise_scale, columnwise_amax,
      active_rows_heavy_light, active_row_count, selection_masks

    `row_counts[row]` is the true selected count. The hot payload for a row is:
      padded_*[row, :min(row_counts[row], max_per_row)]
    Entries beyond max_per_row are appended to the overflow buffers.
    If `overflow[0] != 0`, `overflow_count[0]` exceeded the allocated
    overflow capacity and the caller must rerun with a larger overflow buffer.
    `direct_nomask=True` is exact only for columnwise_source="direct"; it skips
    writing/reading selection_masks and recomputes the threshold predicate in
    fill.
    """

    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    source_to_mode = {
        "direct": 1,
        "te_default_direct": 1,
        "outlier_reuse": 2,
    }
    if columnwise_source not in source_to_mode:
        raise ValueError(
            f"unsupported columnwise_source={columnwise_source!r}; "
            f"expected one of {sorted(source_to_mode)}"
        )
    if direct_nomask and columnwise_source != "direct":
        raise ValueError("direct_nomask requires columnwise_source='direct'")
    if tensor.size(0) != workspace.rows or tensor.size(1) != workspace.cols:
        raise ValueError("workspace shape does not match tensor")
    return ext.adaptive_rowcol_quant_fast_padded_out(
        tensor.contiguous(),
        *workspace.tensors(),
        *workspace.active_schedule_tensors(),
        *workspace.hardcap_tensors(),
        float(base_ratio),
        float(min_ratio),
        float(max_ratio),
        float(reference_heaviness),
        int(workspace.max_per_row),
        int(stats_threads),
        int(fill_threads),
        int(source_to_mode[columnwise_source]),
        int(rht_random_sign_mask_t),
        bool(overlap_columnwise),
        bool(direct_nomask),
    )


def adaptive_rowcol_cap_split_packed(
    row_offsets: torch.Tensor,
    row_ks: torch.Tensor,
    row_values: torch.Tensor,
    *,
    rows: int,
    cols: int,
    cap: int,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    return ext.adaptive_rowcol_cap_split_packed(
        row_offsets.contiguous(),
        row_ks.contiguous(),
        row_values.contiguous(),
        int(rows),
        int(cols),
        int(cap),
    )


def adaptive_rowcol_policy_split_packed(
    row_offsets: torch.Tensor,
    row_ks: torch.Tensor,
    row_values: torch.Tensor,
    *,
    rows: int,
    cols: int,
    policy_mode: int,
    param0: int,
    param1: int,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    return ext.adaptive_rowcol_policy_split_packed(
        row_offsets.contiguous(),
        row_ks.contiguous(),
        row_values.contiguous(),
        int(rows),
        int(cols),
        int(policy_mode),
        int(param0),
        int(param1),
    )


def adaptive_rowcol_policy_bucket_split_packed(
    row_offsets: torch.Tensor,
    row_ks: torch.Tensor,
    row_values: torch.Tensor,
    *,
    rows: int,
    cols: int,
    policy_mode: int,
    param0: int,
    param1: int,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    return ext.adaptive_rowcol_policy_bucket_split_packed(
        row_offsets.contiguous(),
        row_ks.contiguous(),
        row_values.contiguous(),
        int(rows),
        int(cols),
        int(policy_mode),
        int(param0),
        int(param1),
    )


def adaptive_rowcol_direct_split_packed(
    tensor: torch.Tensor,
    row_offsets: torch.Tensor,
    selection_masks: torch.Tensor,
    *,
    selected_nnz: int,
    policy_mode: int,
    param0: int,
    param1: int,
    use_bucket: bool = False,
    fill_threads: int = 128,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    return ext.adaptive_rowcol_direct_split_packed(
        tensor.contiguous(),
        row_offsets.contiguous(),
        selection_masks.contiguous(),
        int(selected_nnz),
        int(policy_mode),
        int(param0),
        int(param1),
        bool(use_bucket),
        int(fill_threads),
    )


def adaptive_rowcol_quant_fast_padded_direct_nomask_out(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolPaddedWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
):
    return adaptive_rowcol_quant_fast_padded_out(
        tensor,
        workspace,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source="direct",
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        direct_nomask=True,
    )


def adaptive_rowcol_quant_fast_out_auto(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    emit_packed_records: bool = False,
    build_active_schedule: bool = True,
):
    return adaptive_rowcol_quant_fast_out(
        tensor,
        workspace,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        auto_expand_capacity=True,
        emit_packed_records=emit_packed_records,
        build_active_schedule=build_active_schedule,
    )


def payload_active_rows_heavy_light(payload, *, sync_count: bool = True) -> torch.Tensor:
    """Return the valid active-row prefix appended by the r25 prep path.

    The extension keeps `active_row_count` as a CUDA int32 tensor so a future
    fused sparse-add kernel can consume it without host synchronization. This
    convenience helper slices for Python checks/benchmarks.
    """

    active_rows = payload[-2]
    active_count = payload[-1]
    if not sync_count:
        return active_rows
    count = int(active_count.detach().cpu()[0])
    return active_rows[:count]


def adaptive_rowcol_loghist_quant_fast_out(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    ratio: float,
    hist_bins: int = 256,
    min_exp: int = -8,
    seed: int = 2026,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "outlier_reuse",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
    auto_expand_capacity: bool = False,
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("adaptive rowcol quant extension is unavailable.")
    if hist_bins not in (64, 128, 256):
        raise ValueError("hist_bins must be one of 64, 128, 256")
    source_to_mode = {
        "direct": 1,
        "te_default_direct": 1,
        "outlier_reuse": 2,
    }
    if columnwise_source not in source_to_mode:
        raise ValueError(
            f"unsupported columnwise_source={columnwise_source!r}; "
            f"expected one of {sorted(source_to_mode)}"
        )
    if tensor.size(0) != workspace.rows or tensor.size(1) != workspace.cols:
        raise ValueError("workspace shape does not match tensor")
    payload = ext.adaptive_rowcol_loghist_quant_fast_out(
        tensor.contiguous(),
        *workspace.tensors(),
        *workspace.loghist_tensors(),
        float(ratio),
        int(hist_bins),
        int(min_exp),
        int(seed),
        int(workspace.capacity),
        int(stats_threads),
        int(fill_threads),
        int(source_to_mode[columnwise_source]),
        int(rht_random_sign_mask_t),
        bool(overlap_columnwise),
        bool(auto_expand_capacity),
    )
    workspace.last_overflow_fallback = bool(auto_expand_capacity and payload[0].numel() > workspace.capacity)
    workspace.last_output_capacity = int(payload[0].numel())
    return payload


def adaptive_rowcol_loghist_quant_fast_out_auto(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    ratio: float,
    hist_bins: int = 256,
    min_exp: int = -8,
    seed: int = 2026,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "outlier_reuse",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
):
    return adaptive_rowcol_loghist_quant_fast_out(
        tensor,
        workspace,
        ratio=ratio,
        hist_bins=hist_bins,
        min_exp=min_exp,
        seed=seed,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
        auto_expand_capacity=True,
    )


def adaptive_rowcol_quant_fast_out_safe(
    tensor: torch.Tensor,
    workspace: AdaptiveRowcolWorkspace,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    reference_heaviness: float = 1.0,
    stats_threads: int = 128,
    fill_threads: int = 64,
    columnwise_source: str = "direct",
    rht_random_sign_mask_t: int = 55272,
    overlap_columnwise: bool = True,
):
    """Preallocated auto-expand path with a final overflow assertion.

    The C++ out path checks selected_nnz after row counting. If it is larger
    than the workspace payload capacity, fill writes into freshly allocated
    exact-size CSR payload tensors instead of truncating. This wrapper adds a
    CPU-side overflow check and keeps a refill fallback for debugging.
    """

    payload = adaptive_rowcol_quant_fast_out_auto(
        tensor,
        workspace,
        base_ratio=base_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        reference_heaviness=reference_heaviness,
        stats_threads=stats_threads,
        fill_threads=fill_threads,
        columnwise_source=columnwise_source,
        rht_random_sign_mask_t=rht_random_sign_mask_t,
        overlap_columnwise=overlap_columnwise,
    )

    overflow_value = int(payload[5].detach().cpu()[0])
    selected_nnz = int(payload[4].detach().cpu()[0])
    workspace.last_selected_nnz = selected_nnz
    if overflow_value == 0:
        return payload

    tensor = tensor.contiguous()
    ext = _load_extension()
    refill_flat, refill_values, refill_cols, refill_overflow = ext.adaptive_rowcol_refill_csr(
        tensor,
        workspace.row_offsets,
        workspace.selection_masks,
        selected_nnz,
        int(fill_threads),
    )
    workspace.last_overflow_fallback = True
    workspace.last_output_capacity = selected_nnz
    return (
        refill_flat,
        refill_values,
        refill_cols,
        payload[3],
        payload[4],
        refill_overflow,
        *payload[6:],
    )
