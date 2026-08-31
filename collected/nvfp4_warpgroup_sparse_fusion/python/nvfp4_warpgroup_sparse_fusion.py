from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXT_NAME = "nvfp4_warpgroup_sparse_fusion"


@dataclass(frozen=True)
class RowIndexedPayload:
    row_offsets: torch.Tensor
    row_ks: torch.Tensor
    row_values: torch.Tensor
    selected_count: int
    r: int = 8
    kb: int = 32
    c: int = 32
    target_ratio: float | None = None


@dataclass(frozen=True)
class PackedLocalDeltaPayload:
    tile_offsets: torch.Tensor
    row_records: torch.Tensor
    entry_records: torch.Tensor
    payload_mode: int = 1


@dataclass(frozen=True)
class KMajorPayload:
    group_offsets: torch.Tensor
    group_ks: torch.Tensor
    entry_offsets: torch.Tensor
    entry_rows: torch.Tensor
    entry_values: torch.Tensor
    active_mblocks: torch.Tensor


@dataclass(frozen=True)
class KMajorTileMetadata:
    group_starts: torch.Tensor
    group_counts: torch.Tensor
    group_meta: torch.Tensor


def densify_kmajor_payload_by_mtile(
    payload: KMajorPayload,
    tile_count: int,
) -> KMajorPayload:
    """Make group_offsets directly indexed by M tile.

    The CUDA side has a fast path when active_mblocks is exactly
    [0, ..., tile_count - 1].  This moves the sparse active-mblock lookup out
    of the dense kernel and into the payload producer.
    """
    tile_count = int(tile_count)
    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if int(payload.group_offsets.numel()) != int(payload.active_mblocks.numel()) + 1:
        raise ValueError("group_offsets must have active_mblocks+1 elements")
    device = payload.group_offsets.device
    active_blocks = payload.active_mblocks.detach().cpu().to(torch.int64).tolist()
    old_offsets = payload.group_offsets.detach().cpu().to(torch.int64).tolist()
    full_offsets = [0 for _ in range(tile_count + 1)]
    cursor = 0
    active_idx = 0
    last_block = -1
    for block in range(tile_count):
        if active_idx < len(active_blocks):
            active_block = int(active_blocks[active_idx])
            if active_block < last_block:
                raise ValueError("active_mblocks must be sorted")
            if active_block == block:
                cursor = int(old_offsets[active_idx + 1])
                active_idx += 1
                last_block = active_block
            elif active_block < block:
                raise ValueError("active_mblocks contains duplicate or stale block id")
        full_offsets[block + 1] = cursor
    if active_idx != len(active_blocks):
        raise ValueError("active_mblocks contains block id outside tile_count")
    return KMajorPayload(
        group_offsets=torch.tensor(full_offsets, device=device, dtype=torch.int32).contiguous(),
        group_ks=payload.group_ks.contiguous(),
        entry_offsets=payload.entry_offsets.contiguous(),
        entry_rows=payload.entry_rows.contiguous(),
        entry_values=payload.entry_values.contiguous(),
        active_mblocks=torch.arange(tile_count, device=device, dtype=torch.int32).contiguous(),
    )


def build_kmajor_tile_metadata(
    payload: KMajorPayload,
    tile_count: int,
) -> KMajorTileMetadata:
    """Precompute direct M-tile -> K-major group range metadata.

    The CUDA stage-split path otherwise has to map the dense M tile back to an
    active payload index and subtract adjacent group offsets.  This mirrors the
    metadata that a fused select/quant payload producer would emit.
    """
    tile_count = int(tile_count)
    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if int(payload.group_offsets.numel()) != int(payload.active_mblocks.numel()) + 1:
        raise ValueError("group_offsets must have active_mblocks+1 elements")
    device = payload.group_offsets.device
    active_blocks = payload.active_mblocks.detach().cpu().to(torch.int64).tolist()
    group_offsets = payload.group_offsets.detach().cpu().to(torch.int64).tolist()
    starts = [0 for _ in range(tile_count)]
    counts = [0 for _ in range(tile_count)]
    last_block = -1
    for active_idx, block in enumerate(active_blocks):
        block = int(block)
        if block < last_block:
            raise ValueError("active_mblocks must be sorted")
        if block < 0 or block >= tile_count:
            raise ValueError("active_mblocks contains block id outside tile_count")
        group_start = int(group_offsets[active_idx])
        group_end = int(group_offsets[active_idx + 1])
        starts[block] = group_start
        counts[block] = group_end - group_start
        last_block = block
    return KMajorTileMetadata(
        group_starts=torch.tensor(starts, device=device, dtype=torch.int32).contiguous(),
        group_counts=torch.tensor(counts, device=device, dtype=torch.int32).contiguous(),
        group_meta=torch.tensor(
            [
                (int(start) & 0xFFFFFFFF) | ((int(count) & 0xFFFFFFFF) << 32)
                for start, count in zip(starts, counts)
            ],
            device=device,
            dtype=torch.int64,
        ).contiguous(),
    )


def build_kmajor_merge_row_payload(
    payload: KMajorPayload,
    tile_count: int,
) -> PackedLocalDeltaPayload:
    """Build packed row metadata for merging a K-major local-delta tile.

    K-major compute keeps entries grouped by K for B reuse.  Epilogue merge only
    needs to know which M-local rows can contain nonzero delta, so this compact
    payload lets the merge loop visit unique active rows instead of the full CTA
    M tile.
    """
    tile_count = int(tile_count)
    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if int(payload.group_offsets.numel()) != int(payload.active_mblocks.numel()) + 1:
        raise ValueError("group_offsets must have active_mblocks+1 elements")
    device = payload.group_offsets.device
    active_blocks = payload.active_mblocks.detach().cpu().to(torch.int64).tolist()
    group_offsets = payload.group_offsets.detach().cpu().to(torch.int64).tolist()
    entry_offsets = payload.entry_offsets.detach().cpu().to(torch.int64).tolist()
    entry_rows = payload.entry_rows.detach().cpu().to(torch.int64).tolist()
    tile_offsets = [0]
    row_records: list[int] = []
    active_idx = 0
    last_block = -1
    for block in range(tile_count):
        rows: set[int] = set()
        if active_idx < len(active_blocks):
            active_block = int(active_blocks[active_idx])
            if active_block < last_block:
                raise ValueError("active_mblocks must be sorted")
            if active_block == block:
                group_start = int(group_offsets[active_idx])
                group_end = int(group_offsets[active_idx + 1])
                for group_idx in range(group_start, group_end):
                    entry_start = int(entry_offsets[group_idx])
                    entry_end = int(entry_offsets[group_idx + 1])
                    for entry_idx in range(entry_start, entry_end):
                        row = int(entry_rows[entry_idx])
                        if 0 <= row <= 0xFFFF:
                            rows.add(row)
                active_idx += 1
                last_block = active_block
            elif active_block < block:
                raise ValueError("active_mblocks contains duplicate or stale block id")
        for row in sorted(rows):
            row_records.append(int(row) & 0xFFFF)
        tile_offsets.append(len(row_records))
    if active_idx != len(active_blocks):
        raise ValueError("active_mblocks contains block id outside tile_count")
    return PackedLocalDeltaPayload(
        tile_offsets=torch.tensor(tile_offsets, device=device, dtype=torch.int32).contiguous(),
        row_records=torch.tensor(row_records, device=device, dtype=torch.int64).contiguous(),
        entry_records=torch.empty((0,), device=device, dtype=torch.int32),
        payload_mode=3,
    )


@dataclass(frozen=True)
class TmaScaleTiles:
    a_scale_tile: torch.Tensor
    b_scale_tile: torch.Tensor


@lru_cache(maxsize=2)
def _load_extension(lineinfo: bool = False):
    if not torch.cuda.is_available():
        return None
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv("NVFP4_DIRECT_ADD_CUDA_ARCH_LIST", "12.0a")
    try:
        suffix = "lineinfo" if lineinfo else "opt"
        build_dir = SRC / f"build_{suffix}"
        build_dir.mkdir(parents=True, exist_ok=True)
        extra_cuda_cflags = ["-O3", "-std=c++17"]
        if lineinfo:
            extra_cuda_cflags.append("-lineinfo")
        return load(
            name=f"{EXT_NAME}_{suffix}",
            sources=[str(SRC / "direct_add.cpp"), str(SRC / "direct_add.cu")],
            build_directory=str(build_dir),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=["-lcuda"],
            verbose=os.getenv("NVFP4_DIRECT_ADD_VERBOSE", "0") == "1",
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch


@lru_cache(maxsize=2)
def _load_tma_extension(lineinfo: bool = False):
    if not torch.cuda.is_available():
        return None
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv("NVFP4_DIRECT_ADD_CUDA_ARCH_LIST", "12.0a")
    try:
        base_suffix = "lineinfo" if lineinfo else "opt"
        variant = os.getenv("NVFP4_DIRECT_ADD_TMA_VARIANT", "").strip()
        safe_variant = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in variant)
        suffix = f"{base_suffix}_{safe_variant}" if safe_variant else base_suffix
        build_dir = SRC / f"build_tma_{suffix}"
        build_dir.mkdir(parents=True, exist_ok=True)
        extra_cuda_cflags = ["-O3", "--use_fast_math", "-std=c++17"]
        extra_env_flags = os.getenv("NVFP4_DIRECT_ADD_TMA_EXTRA_CUDA_CFLAGS", "").strip()
        if extra_env_flags:
            extra_cuda_cflags.extend(extra_env_flags.split())
        if lineinfo:
            extra_cuda_cflags.append("-lineinfo")
        return load(
            name=f"{EXT_NAME}_tma_{suffix}",
            sources=[str(SRC / "tma_direct_add.cpp"), str(SRC / "tma_direct_add.cu")],
            build_directory=str(build_dir),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=["-lcuda"],
            verbose=os.getenv("NVFP4_DIRECT_ADD_VERBOSE", "0") == "1",
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch


def _nvfp4_args(qx, qw):
    m = int(qx._rowwise_data.shape[0])
    k = int(qx._rowwise_data.shape[1] * 2)
    n = int(qw._rowwise_data.shape[0])
    return (
        qx._rowwise_data.contiguous(),
        qx._rowwise_scale_inv.contiguous(),
        qw._rowwise_data.contiguous(),
        qw._rowwise_scale_inv.contiguous(),
        qx._amax_rowwise.contiguous(),
        qw._amax_rowwise.contiguous(),
        m,
        k,
        n,
    )


def make_tma_scale_tiles(qx, qw, *, lineinfo: bool = False) -> TmaScaleTiles:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, _a_amax, _b_amax, m, k, n = _nvfp4_args(qx, qw)
    if m % 128 != 0 or n % 128 != 0 or k % 128 != 0:
        raise ValueError("TMA dense path requires M/N/K divisible by 128")
    return TmaScaleTiles(
        a_scale_tile=ext.swizzle_scale_to_tma_tile_major(a_scale, m, k).contiguous(),
        b_scale_tile=ext.swizzle_scale_to_tma_tile_major(b_scale, n, k).contiguous(),
    )


def make_b_comp(weight_nk: torch.Tensor) -> torch.Tensor:
    if weight_nk.dim() != 2:
        raise ValueError("weight_nk must be [N, K]")
    if weight_nk.dtype != torch.bfloat16:
        raise ValueError("weight_nk must be BF16")
    return weight_nk.contiguous().T.contiguous()


def select_topk_csr(
    x: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be BF16 [M, K]")
    rows, k = int(x.shape[0]), int(x.shape[1])
    total = rows * k
    nnz = max(1, min(total, int(round(float(total) * float(ratio)))))
    flat = torch.topk(x.abs().float().reshape(-1), k=nnz, sorted=False).indices
    flat = flat.sort().values.to(torch.long)
    row_ids = torch.div(flat, k, rounding_mode="floor")
    row_ks = (flat - row_ids * k).to(torch.int32).contiguous()
    values = x.reshape(-1).index_select(0, flat).contiguous()
    counts = torch.bincount(row_ids, minlength=rows).to(torch.int32)
    row_offsets = torch.empty((rows + 1,), device=x.device, dtype=torch.int32)
    row_offsets[0] = 0
    row_offsets[1:] = counts.cumsum(0)
    return flat.to(torch.int32).contiguous(), row_ks, values, row_offsets


def build_row_indexed_payload(
    row_offsets: torch.Tensor,
    row_ks: torch.Tensor,
    row_values: torch.Tensor,
    *,
    selected_count: int,
    target_ratio: float,
    r: int = 8,
    kb: int = 32,
    c: int = 32,
) -> RowIndexedPayload:
    return RowIndexedPayload(
        row_offsets=row_offsets.contiguous(),
        row_ks=row_ks.contiguous(),
        row_values=row_values.contiguous(),
        selected_count=int(selected_count),
        r=r,
        kb=kb,
        c=c,
        target_ratio=target_ratio,
    )


def build_packed_local_delta_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    bm: int = 128,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    light_row_nnz_lt: int | None = None,
    light_min_block_nnz: int = 4096,
    max_entries_per_tile: int | None = None,
    balance_probe_warps: int = 0,
    light_rows_per_warp: int = 8,
    strict_prefiltered: bool = False,
) -> PackedLocalDeltaPayload:
    if payload.row_offsets.dim() != 1:
        raise ValueError("row_offsets must be 1D")
    if int(payload.row_offsets.numel()) != int(m) + 1:
        raise ValueError("row_offsets must be M+1")
    if int(payload.row_ks.numel()) != int(payload.row_values.numel()):
        raise ValueError("row_ks and row_values length mismatch")
    if int(payload.row_ks.numel()) > 0 and int(payload.row_ks.max().item()) > 0xFFFF:
        raise ValueError("packed local-delta payload only supports K <= 65535")
    if int(bm) <= 0 or int(bm) > 0xFFFF:
        raise ValueError("bm must be in (0, 65535]")
    if light_row_nnz_lt is not None and int(light_row_nnz_lt) <= 0:
        raise ValueError("light_row_nnz_lt must be positive")
    if max_entries_per_tile is not None and int(max_entries_per_tile) <= 0:
        raise ValueError("max_entries_per_tile must be positive")
    if int(balance_probe_warps) < 0:
        raise ValueError("balance_probe_warps must be non-negative")
    if int(light_rows_per_warp) <= 0:
        raise ValueError("light_rows_per_warp must be positive")
    if strict_prefiltered and light_row_nnz_lt is None and max_entries_per_tile is None:
        raise ValueError("strict_prefiltered requires producer-side light/cap filtering")

    device = payload.row_offsets.device
    row_offsets_cpu = payload.row_offsets.detach().cpu().to(torch.int64).tolist()
    row_ks_cpu = payload.row_ks.detach().cpu().to(torch.int64).tolist()
    values_bits_cpu = (
        payload.row_values.detach().cpu().contiguous().view(torch.int16).to(torch.int64)
        & 0xFFFF
    ).tolist()

    tile_offsets: list[int] = [0]
    row_records: list[int] = []
    entry_records: list[int] = []
    tile_count = (int(m) + int(bm) - 1) // int(bm)
    active_row_offsets_cpu: list[int] | None = None
    active_rows_cpu: list[int] | None = None
    if active_row_offsets is not None or active_rows is not None:
        if active_row_offsets is None or active_rows is None:
            raise ValueError("active_row_offsets and active_rows must be provided together")
        if int(active_row_offsets.numel()) != tile_count + 1:
            raise ValueError("active_row_offsets must be tile_m_count+1")
        active_row_offsets_cpu = active_row_offsets.detach().cpu().to(torch.int64).tolist()
        active_rows_cpu = active_rows.detach().cpu().to(torch.int64).tolist()
    for tile_m in range(tile_count):
        global_row0 = tile_m * int(bm)
        global_row1 = min(global_row0 + int(bm), int(m))
        tile_entry_base = len(entry_records)
        block_entry_count = int(row_offsets_cpu[global_row1]) - int(row_offsets_cpu[global_row0])
        prefilter_light = (
            light_row_nnz_lt is not None
            and block_entry_count >= int(light_min_block_nnz)
        )
        if active_row_offsets_cpu is not None and active_rows_cpu is not None:
            local_rows = active_rows_cpu[
                int(active_row_offsets_cpu[tile_m]) : int(active_row_offsets_cpu[tile_m + 1])
            ]
        else:
            local_rows = range(global_row1 - global_row0)
        row_items: list[tuple[int, int, int, int]] = []
        for local_row_value in local_rows:
            local_row = int(local_row_value)
            if local_row < 0 or local_row >= global_row1 - global_row0:
                continue
            global_row = global_row0 + local_row
            start = int(row_offsets_cpu[global_row])
            end = int(row_offsets_cpu[global_row + 1])
            count = end - start
            if count <= 0:
                continue
            if prefilter_light and count >= int(light_row_nnz_lt):
                continue
            row_items.append((local_row, start, end, count))
        if int(balance_probe_warps) > 1 and len(row_items) > 1:
            buckets: list[list[tuple[int, int, int, int]]] = [
                [] for _ in range(int(balance_probe_warps))
            ]
            loads = [0 for _ in range(int(balance_probe_warps))]
            for item in sorted(row_items, key=lambda x: (-x[3], x[0])):
                bucket = min(range(int(balance_probe_warps)), key=lambda idx: loads[idx])
                buckets[bucket].append(item)
                loads[bucket] += item[3]
            balanced_items: list[tuple[int, int, int, int]] = []
            cursor = 0
            while True:
                added = False
                for bucket in buckets:
                    chunk = bucket[cursor : cursor + int(light_rows_per_warp)]
                    if chunk:
                        balanced_items.extend(chunk)
                        added = True
                if not added:
                    break
                cursor += int(light_rows_per_warp)
            row_items = balanced_items
        for local_row, start, end, count in row_items:
            if max_entries_per_tile is not None:
                remaining = int(max_entries_per_tile) - (len(entry_records) - tile_entry_base)
                if remaining <= 0:
                    break
                if count > remaining:
                    count = remaining
                    end = start + count
            if count > 0xFFFF:
                raise ValueError("packed local-delta row count exceeds 16-bit capacity")
            entry_start = len(entry_records)
            if entry_start > 0xFFFFFFFF:
                raise ValueError("packed local-delta entry offset exceeds 32-bit capacity")
            for entry_idx in range(start, end):
                k_value = int(row_ks_cpu[entry_idx])
                if k_value < 0 or k_value > 0xFFFF:
                    raise ValueError("packed local-delta k index must fit uint16")
                value_bits = int(values_bits_cpu[entry_idx]) & 0xFFFF
                packed_entry = ((k_value & 0xFFFF) << 16) | value_bits
                if packed_entry >= 0x80000000:
                    packed_entry -= 0x100000000
                entry_records.append(packed_entry)
            row_record = (
                (int(entry_start) << 32)
                | ((int(count) & 0xFFFF) << 16)
                | (int(local_row) & 0xFFFF)
            )
            row_records.append(row_record)
        tile_offsets.append(len(row_records))

    return PackedLocalDeltaPayload(
        tile_offsets=torch.tensor(tile_offsets, device=device, dtype=torch.int32).contiguous(),
        row_records=torch.tensor(row_records, device=device, dtype=torch.int64).contiguous(),
        entry_records=torch.tensor(entry_records, device=device, dtype=torch.int32).contiguous(),
        payload_mode=(
            3
            if strict_prefiltered
            else (2 if light_row_nnz_lt is not None or max_entries_per_tile is not None else 1)
        ),
    )


def build_packed_rowblock_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    rows_per_block: int = 8,
    order: str = "stripe_balance",
    stripe_count: int | None = None,
) -> tuple[torch.Tensor, PackedLocalDeltaPayload]:
    if int(rows_per_block) != 8:
        raise ValueError("packed rowblock path currently expects rows_per_block=8")
    if payload.row_offsets.dim() != 1:
        raise ValueError("row_offsets must be 1D")
    if int(payload.row_offsets.numel()) != int(m) + 1:
        raise ValueError("row_offsets must be M+1")
    if int(payload.row_ks.numel()) != int(payload.row_values.numel()):
        raise ValueError("row_ks and row_values length mismatch")
    if int(payload.row_ks.numel()) > 0 and int(payload.row_ks.max().item()) > 0xFFFF:
        raise ValueError("packed rowblock payload only supports K <= 65535")

    device = payload.row_offsets.device
    active_rowblocks = compact_active_rowblocks_from_offsets(
        payload.row_offsets,
        m,
        rows_per_block=rows_per_block,
        order=order,
        stripe_count=stripe_count,
    )
    row_offsets_cpu = payload.row_offsets.detach().cpu().to(torch.int64).tolist()
    row_ks_cpu = payload.row_ks.detach().cpu().to(torch.int64).tolist()
    values_bits_cpu = (
        payload.row_values.detach().cpu().contiguous().view(torch.int16).to(torch.int64)
        & 0xFFFF
    ).tolist()
    active_blocks_cpu = active_rowblocks.detach().cpu().to(torch.int64).tolist()

    tile_offsets: list[int] = [0]
    row_records: list[int] = []
    entry_records: list[int] = []
    preferred_slots = [0, 2, 4, 6, 1, 3, 5, 7]
    for block in active_blocks_cpu:
        block = int(block)
        row0 = block * int(rows_per_block)
        row1 = min(row0 + int(rows_per_block), int(m))
        row_items: list[tuple[int, int, int, int]] = []
        for row in range(row0, row1):
            start = int(row_offsets_cpu[row])
            end = int(row_offsets_cpu[row + 1])
            count = end - start
            if count > 0:
                row_items.append((count, row - row0, start, end))
        row_items.sort(key=lambda item: (-item[0], item[1]))
        slotted: list[tuple[int, int, int, int] | None] = [None for _ in range(int(rows_per_block))]
        spill_slots = [idx for idx in range(int(rows_per_block)) if idx not in preferred_slots]
        spill_cursor = 0
        for item_idx, item in enumerate(row_items[: int(rows_per_block)]):
            slot = preferred_slots[item_idx] if item_idx < len(preferred_slots) else spill_slots[spill_cursor]
            if item_idx >= len(preferred_slots):
                spill_cursor += 1
            slotted[slot] = item
        for item in slotted:
            if item is None:
                row_records.append(0)
                continue
            count, local_row, start, end = item
            entry_start = len(entry_records)
            for entry_idx in range(start, end):
                k_value = int(row_ks_cpu[entry_idx])
                if k_value < 0 or k_value > 0xFFFF:
                    raise ValueError("packed rowblock k index must fit uint16")
                value_bits = int(values_bits_cpu[entry_idx]) & 0xFFFF
                packed_entry = ((k_value & 0xFFFF) << 16) | value_bits
                if packed_entry >= 0x80000000:
                    packed_entry -= 0x100000000
                entry_records.append(packed_entry)
            row_record = (
                (int(entry_start) << 32)
                | ((int(count) & 0xFFFF) << 16)
                | (int(local_row) & 0xFFFF)
            )
            row_records.append(row_record)
        tile_offsets.append(len(row_records))

    return (
        active_rowblocks.contiguous(),
        PackedLocalDeltaPayload(
            tile_offsets=torch.tensor(tile_offsets, device=device, dtype=torch.int32).contiguous(),
            row_records=torch.tensor(row_records, device=device, dtype=torch.int64).contiguous(),
            entry_records=torch.tensor(entry_records, device=device, dtype=torch.int32).contiguous(),
            payload_mode=4,
        ),
    )


def row_payload_flat_indices(payload: RowIndexedPayload, m: int, k: int) -> torch.Tensor:
    counts = payload.row_offsets[1:] - payload.row_offsets[:-1]
    rows = torch.repeat_interleave(
        torch.arange(int(m), device=payload.row_offsets.device, dtype=torch.int32),
        counts.to(torch.long),
    )
    return (rows * int(k) + payload.row_ks.to(device=rows.device, dtype=torch.int32)).contiguous()


def reorder_active_rows_by_nnz(
    row_offsets: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    mode: str = "heavy_light",
) -> torch.Tensor:
    if mode == "row_id":
        return active_rows.to(device=row_offsets.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=row_offsets.device, dtype=torch.int32).contiguous()
    if int(active_rows.numel()) <= 1:
        return active_rows
    counts_all = row_offsets[1:] - row_offsets[:-1]
    counts = counts_all.index_select(0, active_rows.to(torch.long))
    stable = torch.arange(int(active_rows.numel()), device=active_rows.device, dtype=torch.int64)
    desc_keys = counts.to(torch.int64) * (int(active_rows.numel()) + 1) - stable
    asc_keys = -counts.to(torch.int64) * (int(active_rows.numel()) + 1) - stable
    if mode == "nnz_desc":
        order = torch.argsort(desc_keys, descending=True)
    elif mode == "nnz_asc":
        order = torch.argsort(asc_keys, descending=True)
    elif mode == "heavy_light":
        desc = torch.argsort(desc_keys, descending=True)
        heavy = desc[0::2]
        light = desc[1::2].flip(0)
        pieces: list[torch.Tensor] = []
        for idx in range(max(int(heavy.numel()), int(light.numel()))):
            if idx < int(heavy.numel()):
                pieces.append(heavy[idx : idx + 1])
            if idx < int(light.numel()):
                pieces.append(light[idx : idx + 1])
        order = torch.cat(pieces, dim=0)
    else:
        raise ValueError("mode must be 'row_id', 'nnz_desc', 'nnz_asc', or 'heavy_light'")
    return active_rows.index_select(0, order).contiguous()


def balance_active_rows_for_persistent_workers(
    row_offsets: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    m: int,
    n: int,
    sparse_warpgroups: int = 1,
    worker_count: int | None = None,
) -> torch.Tensor:
    active_rows = active_rows.to(device=row_offsets.device, dtype=torch.int32).contiguous()
    if int(active_rows.numel()) <= 1:
        return active_rows
    sparse_warpgroups = abs(int(sparse_warpgroups))
    if sparse_warpgroups not in (1, 2):
        raise ValueError("sparse_warpgroups must be 1 or 2")
    if int(n) <= 0 or int(n) % 8 != 0:
        raise ValueError("n must be positive and divisible by 8")
    if worker_count is None:
        props = torch.cuda.get_device_properties(active_rows.device)
        dense_ctas = ((int(m) + 127) // 128) * ((int(n) + 127) // 128)
        worker_count = min(int(props.multi_processor_count), dense_ctas)
    worker_count = int(worker_count)
    groups_per_row = int(n) // 8
    threads_per_worker = sparse_warpgroups * 128
    if (
        worker_count <= 0
        or groups_per_row % threads_per_worker != 0
        or worker_count % (groups_per_row // threads_per_worker) != 0
    ):
        return reorder_active_rows_by_nnz(row_offsets, active_rows, mode="heavy_light")

    workers_per_row = groups_per_row // threads_per_worker
    bucket_count = worker_count // workers_per_row
    row_count = int(active_rows.numel())
    capacities = [
        (row_count + bucket_count - 1 - bucket) // bucket_count
        for bucket in range(bucket_count)
    ]
    counts = (row_offsets[1:] - row_offsets[:-1]).index_select(
        0, active_rows.to(torch.long)
    )
    rows_cpu = active_rows.detach().cpu().to(torch.int64).tolist()
    counts_cpu = counts.detach().cpu().to(torch.int64).tolist()
    ordered = sorted(
        zip(rows_cpu, counts_cpu),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    buckets: list[list[int]] = [[] for _ in range(bucket_count)]
    loads = [0 for _ in range(bucket_count)]
    for row, count in ordered:
        candidates = [
            bucket
            for bucket in range(bucket_count)
            if len(buckets[bucket]) < capacities[bucket]
        ]
        target = min(
            candidates,
            key=lambda bucket: (loads[bucket], len(buckets[bucket]), bucket),
        )
        buckets[target].append(int(row))
        loads[target] += int(count)

    scheduled: list[int] = []
    for position in range(row_count):
        bucket = position % bucket_count
        depth = position // bucket_count
        scheduled.append(buckets[bucket][depth])
    return torch.tensor(
        scheduled, device=active_rows.device, dtype=torch.int32
    ).contiguous()


def balanced_active_rows_by_block_from_offsets(
    row_offsets: torch.Tensor,
    m: int,
    *,
    bm: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = (row_offsets[1:] - row_offsets[:-1]).detach().cpu().tolist()
    tiles_m = (int(m) + bm - 1) // bm
    offsets = [0]
    scheduled_rows: list[int] = []
    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        rows = [row - row0 for row in range(row0, row1) if counts[row] > 0]
        if not rows:
            offsets.append(len(scheduled_rows))
            continue
        inactive = next((row - row0 for row in range(row0, row1) if counts[row] == 0), None)
        rows.sort(key=lambda local_row: counts[row0 + local_row], reverse=True)
        pairs = [rows[i : i + 2] for i in range(0, len(rows), 2)]
        warp_pairs: list[list[list[int]]] = [[] for _ in range(16)]
        warp_loads = [0] * 16
        for pair in pairs:
            pair_load = max(counts[row0 + local_row] for local_row in pair)
            target = min(range(16), key=lambda idx: warp_loads[idx])
            warp_pairs[target].append(pair)
            warp_loads[target] += pair_load
        max_rounds = max(len(bucket) for bucket in warp_pairs)
        for round_idx in range(max_rounds):
            for warp in range(16):
                if round_idx >= len(warp_pairs[warp]):
                    if inactive is not None:
                        scheduled_rows.extend([inactive, inactive])
                    continue
                pair = warp_pairs[warp][round_idx]
                scheduled_rows.extend(pair)
                if len(pair) == 1 and inactive is not None:
                    scheduled_rows.append(inactive)
        offsets.append(len(scheduled_rows))
    return (
        torch.tensor(offsets, device=row_offsets.device, dtype=torch.int32),
        torch.tensor(scheduled_rows, device=row_offsets.device, dtype=torch.int32),
    )


def compact_active_rows_by_block_from_offsets(
    row_offsets: torch.Tensor,
    m: int,
    *,
    bm: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = (row_offsets[1:] - row_offsets[:-1]).detach().cpu().tolist()
    tiles_m = (int(m) + bm - 1) // bm
    offsets = [0]
    scheduled_rows: list[int] = []
    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        rows = [row - row0 for row in range(row0, row1) if counts[row] > 0]
        rows.sort(key=lambda local_row: counts[row0 + local_row], reverse=True)
        scheduled_rows.extend(rows)
        offsets.append(len(scheduled_rows))
    return (
        torch.tensor(offsets, device=row_offsets.device, dtype=torch.int32),
        torch.tensor(scheduled_rows, device=row_offsets.device, dtype=torch.int32),
    )


def compact_active_rowblocks_from_offsets(
    row_offsets: torch.Tensor,
    m: int,
    *,
    rows_per_block: int = 8,
    order: str = "row_id",
    stripe_count: int | None = None,
) -> torch.Tensor:
    counts = (row_offsets[1:] - row_offsets[:-1]).detach().cpu().tolist()
    rows_per_block = int(rows_per_block)
    if rows_per_block <= 0:
        raise ValueError("rows_per_block must be positive")
    blocks = (int(m) + rows_per_block - 1) // rows_per_block
    active_blocks: list[tuple[int, int]] = []
    for block in range(blocks):
        row0 = block * rows_per_block
        row1 = min(row0 + rows_per_block, int(m))
        block_nnz = sum(int(counts[row]) for row in range(row0, row1))
        if block_nnz > 0:
            active_blocks.append((block, block_nnz))
    if not active_blocks:
        active_blocks = [(0, 0)]

    order = str(order)
    if order == "row_id":
        ordered_blocks = [block for block, _nnz in active_blocks]
    elif order == "nnz_desc":
        ordered_blocks = [
            block for block, _nnz in sorted(active_blocks, key=lambda item: (-item[1], item[0]))
        ]
    elif order == "heavy_light":
        desc = sorted(active_blocks, key=lambda item: (-item[1], item[0]))
        heavy = desc[0::2]
        light = list(reversed(desc[1::2]))
        ordered_pairs: list[tuple[int, int]] = []
        for idx in range(max(len(heavy), len(light))):
            if idx < len(heavy):
                ordered_pairs.append(heavy[idx])
            if idx < len(light):
                ordered_pairs.append(light[idx])
        ordered_blocks = [block for block, _nnz in ordered_pairs]
    elif order == "stripe_balance":
        stripes = int(stripe_count) if stripe_count is not None else (int(m) + 127) // 128
        if stripes <= 0:
            raise ValueError("stripe_count must be positive")
        buckets: list[list[int]] = [[] for _ in range(stripes)]
        loads = [0 for _ in range(stripes)]
        for block, block_nnz in sorted(active_blocks, key=lambda item: (-item[1], item[0])):
            target = min(range(stripes), key=lambda idx: (loads[idx], idx))
            buckets[target].append(block)
            loads[target] += int(block_nnz)
        ordered_blocks = []
        max_depth = max((len(bucket) for bucket in buckets), default=0)
        for depth in range(max_depth):
            for bucket in buckets:
                if depth < len(bucket):
                    ordered_blocks.append(bucket[depth])
    else:
        raise ValueError("order must be 'row_id', 'nnz_desc', 'heavy_light', or 'stripe_balance'")
    return torch.tensor(ordered_blocks, device=row_offsets.device, dtype=torch.int32)


def build_rowblock_nblock_rowpair_task_records(
    active_rowblocks: torch.Tensor,
    n: int,
    *,
    task_n: int = 128,
) -> torch.Tensor:
    """Encode persistent row-pair tasks as [rowblock, nblock, row_pair] records.

    Layout: bits [0:2) row_pair, bits [2:14) nblock, bits [14:] rowblock.
    This moves active-rowblock lookup and nblock division out of the hot path.
    """
    if active_rowblocks.device.type != "cuda" or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32")
    task_n = int(task_n)
    if task_n <= 0:
        raise ValueError("task_n must be positive")
    nblock_count = (int(n) + task_n - 1) // task_n
    if nblock_count >= (1 << 12):
        raise ValueError("task record encoding supports fewer than 4096 N blocks")
    active_blocks = active_rowblocks.detach().cpu().to(torch.int64).tolist()
    records: list[int] = []
    for block in active_blocks:
        block = int(block)
        if block < 0:
            raise ValueError("active_rowblocks must be non-negative")
        block_base = block << 14
        for nblock in range(nblock_count):
            base = block_base | (nblock << 2)
            records.extend((base | 0, base | 1, base | 2, base | 3))
    if not records:
        records = [0]
    return torch.tensor(records, device=active_rowblocks.device, dtype=torch.int32)


def compact_active_rows_by_block_from_offsets_range(
    row_offsets: torch.Tensor,
    m: int,
    *,
    min_nnz_exclusive: int = 0,
    max_nnz_exclusive: int | None = None,
    bm: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = (row_offsets[1:] - row_offsets[:-1]).detach().cpu().tolist()
    tiles_m = (int(m) + bm - 1) // bm
    min_nnz = int(min_nnz_exclusive)
    max_nnz = None if max_nnz_exclusive is None else int(max_nnz_exclusive)
    offsets = [0]
    scheduled_rows: list[int] = []
    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        rows = [
            row - row0
            for row in range(row0, row1)
            if counts[row] > min_nnz and (max_nnz is None or counts[row] < max_nnz)
        ]
        rows.sort(key=lambda local_row: counts[row0 + local_row], reverse=True)
        scheduled_rows.extend(rows)
        offsets.append(len(scheduled_rows))
    return (
        torch.tensor(offsets, device=row_offsets.device, dtype=torch.int32),
        torch.tensor(scheduled_rows, device=row_offsets.device, dtype=torch.int32),
    )


def build_kmajor_payload_from_row_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    bm: int = 128,
) -> KMajorPayload:
    row_offsets_cpu = payload.row_offsets.detach().cpu()
    row_ks_cpu = payload.row_ks.detach().cpu()
    row_values_cpu = payload.row_values.detach().cpu()
    tiles_m = (int(m) + bm - 1) // bm
    active_mblocks: list[int] = []
    group_offsets = [0]
    group_ks: list[int] = []
    entry_offsets = [0]
    entry_rows: list[int] = []
    entry_values_chunks: list[torch.Tensor] = []

    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        groups: dict[int, list[tuple[int, int]]] = {}
        for row in range(row0, row1):
            start = int(row_offsets_cpu[row].item())
            end = int(row_offsets_cpu[row + 1].item())
            if start == end:
                continue
            local_row = row - row0
            for entry_idx in range(start, end):
                k_col = int(row_ks_cpu[entry_idx].item())
                groups.setdefault(k_col, []).append((local_row, entry_idx))
        if not groups:
            continue
        active_mblocks.append(block)
        # Hot groups first improves early cache reuse and keeps long groups balanced across warps.
        for k_col, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            group_ks.append(k_col)
            for local_row, entry_idx in entries:
                entry_rows.append(local_row)
                entry_values_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
            entry_offsets.append(len(entry_rows))
        group_offsets.append(len(group_ks))

    device = payload.row_offsets.device
    if not active_mblocks:
        active_mblocks = [0]
        group_offsets = [0, 0]
    if entry_values_chunks:
        entry_values_cpu = torch.cat(entry_values_chunks, dim=0).contiguous()
    else:
        entry_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)
    return KMajorPayload(
        group_offsets=torch.tensor(group_offsets, device=device, dtype=torch.int32),
        group_ks=torch.tensor(group_ks, device=device, dtype=torch.int32),
        entry_offsets=torch.tensor(entry_offsets, device=device, dtype=torch.int32),
        entry_rows=torch.tensor(entry_rows, device=device, dtype=torch.int32),
        entry_values=entry_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        active_mblocks=torch.tensor(active_mblocks, device=device, dtype=torch.int32),
    )


def build_kmajor_payload_from_active_rows(
    payload: RowIndexedPayload,
    m: int,
    active_row_offsets: torch.Tensor,
    active_rows_local: torch.Tensor,
    *,
    bm: int = 128,
) -> KMajorPayload:
    row_offsets_cpu = payload.row_offsets.detach().cpu()
    row_ks_cpu = payload.row_ks.detach().cpu()
    row_values_cpu = payload.row_values.detach().cpu()
    active_offsets_cpu = active_row_offsets.detach().cpu()
    active_rows_cpu = active_rows_local.detach().cpu()
    tiles_m = (int(m) + bm - 1) // bm
    active_mblocks: list[int] = []
    group_offsets = [0]
    group_ks: list[int] = []
    entry_offsets = [0]
    entry_rows: list[int] = []
    entry_values_chunks: list[torch.Tensor] = []

    for block in range(tiles_m):
        active_start = int(active_offsets_cpu[block].item())
        active_end = int(active_offsets_cpu[block + 1].item())
        if active_start == active_end:
            continue
        row0 = block * bm
        groups: dict[int, list[tuple[int, int]]] = {}
        for active_idx in range(active_start, active_end):
            local_row = int(active_rows_cpu[active_idx].item())
            if local_row < 0 or local_row >= bm:
                continue
            row = row0 + local_row
            if row < 0 or row >= int(m):
                continue
            start = int(row_offsets_cpu[row].item())
            end = int(row_offsets_cpu[row + 1].item())
            if start == end:
                continue
            for entry_idx in range(start, end):
                k_col = int(row_ks_cpu[entry_idx].item())
                groups.setdefault(k_col, []).append((local_row, entry_idx))
        if not groups:
            continue
        active_mblocks.append(block)
        for k_col, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            group_ks.append(k_col)
            for local_row, entry_idx in entries:
                entry_rows.append(local_row)
                entry_values_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
            entry_offsets.append(len(entry_rows))
        group_offsets.append(len(group_ks))

    device = payload.row_offsets.device
    if not active_mblocks:
        active_mblocks = [0]
        group_offsets = [0, 0]
    if entry_values_chunks:
        entry_values_cpu = torch.cat(entry_values_chunks, dim=0).contiguous()
    else:
        entry_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)
    return KMajorPayload(
        group_offsets=torch.tensor(group_offsets, device=device, dtype=torch.int32),
        group_ks=torch.tensor(group_ks, device=device, dtype=torch.int32),
        entry_offsets=torch.tensor(entry_offsets, device=device, dtype=torch.int32),
        entry_rows=torch.tensor(entry_rows, device=device, dtype=torch.int32),
        entry_values=entry_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        active_mblocks=torch.tensor(active_mblocks, device=device, dtype=torch.int32),
    )


def build_hot_row_schedule_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    row_nnz_threshold: int,
    min_block_nnz: int,
    schedule_warps: int = 4,
    bm: int = 128,
) -> KMajorPayload:
    if row_nnz_threshold <= 0:
        raise ValueError("row_nnz_threshold must be positive")
    if min_block_nnz < 0:
        raise ValueError("min_block_nnz must be non-negative")
    if schedule_warps < 1:
        raise ValueError("schedule_warps must be positive")
    row_offsets_cpu = payload.row_offsets.detach().cpu()
    tiles_m = (int(m) + bm - 1) // bm
    active_mblocks: list[int] = []
    group_offsets = [0]
    group_ks: list[int] = []
    entry_offsets = [0]
    entry_rows: list[int] = []

    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        block_nnz = int(row_offsets_cpu[row1].item()) - int(row_offsets_cpu[row0].item())
        if block_nnz < min_block_nnz:
            continue
        hot_rows: list[tuple[int, int]] = []
        for row in range(row0, row1):
            row_nnz = int(row_offsets_cpu[row + 1].item()) - int(row_offsets_cpu[row].item())
            if row_nnz >= row_nnz_threshold:
                hot_rows.append((row_nnz, row - row0))
        if not hot_rows:
            continue
        active_mblocks.append(block)
        hot_rows.sort(key=lambda item: (-item[0], item[1]))
        bins: list[list[tuple[int, int]]] = [[] for _ in range(int(schedule_warps))]
        loads = [0] * int(schedule_warps)
        for row_nnz, local_row in hot_rows:
            target = min(range(int(schedule_warps)), key=lambda idx: loads[idx])
            bins[target].append((row_nnz, local_row))
            loads[target] += row_nnz
        ordered_hot_rows: list[tuple[int, int]] = []
        max_len = max((len(bin_rows) for bin_rows in bins), default=0)
        for slot in range(max_len):
            for bin_rows in bins:
                if slot < len(bin_rows):
                    ordered_hot_rows.append(bin_rows[slot])
        for _row_nnz, local_row in ordered_hot_rows:
            group_ks.append(int(row_nnz_threshold))
            entry_rows.append(local_row)
            entry_offsets.append(len(entry_rows))
        group_offsets.append(len(group_ks))

    device = payload.row_offsets.device
    if not active_mblocks:
        active_mblocks = [0]
        group_offsets = [0, 0]
    values = torch.empty((len(entry_rows),), dtype=payload.row_values.dtype)
    if len(entry_rows):
        values.zero_()
    return KMajorPayload(
        group_offsets=torch.tensor(group_offsets, device=device, dtype=torch.int32),
        group_ks=torch.tensor(group_ks, device=device, dtype=torch.int32),
        entry_offsets=torch.tensor(entry_offsets, device=device, dtype=torch.int32),
        entry_rows=torch.tensor(entry_rows, device=device, dtype=torch.int32),
        entry_values=values.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        active_mblocks=torch.tensor(active_mblocks, device=device, dtype=torch.int32),
    )


def build_hot_kmajor_cold_row_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    min_hot_freq: int,
    bm: int = 128,
) -> tuple[KMajorPayload, RowIndexedPayload]:
    if min_hot_freq < 2:
        raise ValueError("min_hot_freq must be >= 2")
    row_offsets_cpu = payload.row_offsets.detach().cpu()
    row_ks_cpu = payload.row_ks.detach().cpu()
    row_values_cpu = payload.row_values.detach().cpu()
    tiles_m = (int(m) + bm - 1) // bm
    active_mblocks: list[int] = []
    group_offsets = [0]
    group_ks: list[int] = []
    entry_offsets = [0]
    entry_rows: list[int] = []
    hot_value_chunks: list[torch.Tensor] = []
    cold_entries_by_row: list[list[int]] = [[] for _ in range(int(m))]

    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        groups: dict[int, list[tuple[int, int, int]]] = {}
        block_nnz = 0
        for row in range(row0, row1):
            start = int(row_offsets_cpu[row].item())
            end = int(row_offsets_cpu[row + 1].item())
            block_nnz += end - start
            if start == end:
                continue
            local_row = row - row0
            for entry_idx in range(start, end):
                k_col = int(row_ks_cpu[entry_idx].item())
                groups.setdefault(k_col, []).append((row, local_row, entry_idx))
        if block_nnz == 0:
            continue
        active_mblocks.append(block)
        for k_col, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(entries) >= min_hot_freq:
                group_ks.append(k_col)
                for _row, local_row, entry_idx in entries:
                    entry_rows.append(local_row)
                    hot_value_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
                entry_offsets.append(len(entry_rows))
            else:
                for row, _local_row, entry_idx in entries:
                    cold_entries_by_row[row].append(entry_idx)
        group_offsets.append(len(group_ks))

    cold_row_offsets = [0]
    cold_ks: list[int] = []
    cold_value_chunks: list[torch.Tensor] = []
    for row_entries in cold_entries_by_row:
        row_entries.sort()
        for entry_idx in row_entries:
            cold_ks.append(int(row_ks_cpu[entry_idx].item()))
            cold_value_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
        cold_row_offsets.append(len(cold_ks))

    device = payload.row_offsets.device
    if not active_mblocks:
        active_mblocks = [0]
        group_offsets = [0, 0]
    if hot_value_chunks:
        hot_values_cpu = torch.cat(hot_value_chunks, dim=0).contiguous()
    else:
        hot_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)
    if cold_value_chunks:
        cold_values_cpu = torch.cat(cold_value_chunks, dim=0).contiguous()
    else:
        cold_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)

    kmajor_payload = KMajorPayload(
        group_offsets=torch.tensor(group_offsets, device=device, dtype=torch.int32),
        group_ks=torch.tensor(group_ks, device=device, dtype=torch.int32),
        entry_offsets=torch.tensor(entry_offsets, device=device, dtype=torch.int32),
        entry_rows=torch.tensor(entry_rows, device=device, dtype=torch.int32),
        entry_values=hot_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        active_mblocks=torch.tensor(active_mblocks, device=device, dtype=torch.int32),
    )
    cold_payload = RowIndexedPayload(
        row_offsets=torch.tensor(cold_row_offsets, device=device, dtype=torch.int32),
        row_ks=torch.tensor(cold_ks, device=device, dtype=torch.int32),
        row_values=cold_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        selected_count=len(cold_ks),
        r=payload.r,
        kb=payload.kb,
        c=payload.c,
        target_ratio=payload.target_ratio,
    )
    return kmajor_payload, cold_payload


def build_budgeted_hot_kmajor_cold_row_payload(
    payload: RowIndexedPayload,
    m: int,
    *,
    min_hot_freq: int,
    max_hot_groups_per_block: int,
    max_cold_entries_per_block: int,
    bm: int = 128,
) -> tuple[KMajorPayload, RowIndexedPayload, dict[str, float]]:
    if min_hot_freq < 2:
        raise ValueError("min_hot_freq must be >= 2")
    if max_hot_groups_per_block < 0 or max_cold_entries_per_block < 0:
        raise ValueError("budget caps must be non-negative")
    row_offsets_cpu = payload.row_offsets.detach().cpu()
    row_ks_cpu = payload.row_ks.detach().cpu()
    row_values_cpu = payload.row_values.detach().cpu()
    abs_values_cpu = row_values_cpu.float().abs()
    tiles_m = (int(m) + bm - 1) // bm
    active_mblocks: list[int] = []
    group_offsets = [0]
    group_ks: list[int] = []
    entry_offsets = [0]
    entry_rows: list[int] = []
    hot_value_chunks: list[torch.Tensor] = []
    cold_entries_by_row: list[list[int]] = [[] for _ in range(int(m))]

    total_abs_mass = float(abs_values_cpu.sum().item())
    covered_abs_mass = 0.0
    total_nnz = int(payload.selected_count)
    covered_hot_nnz = 0
    covered_cold_nnz = 0
    selected_hot_groups = 0
    full_hot_group_count = 0
    full_cold_nnz = 0

    for block in range(tiles_m):
        row0 = block * bm
        row1 = min(row0 + bm, int(m))
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for row in range(row0, row1):
            start = int(row_offsets_cpu[row].item())
            end = int(row_offsets_cpu[row + 1].item())
            if start == end:
                continue
            local_row = row - row0
            for entry_idx in range(start, end):
                k_col = int(row_ks_cpu[entry_idx].item())
                groups.setdefault(k_col, []).append((row, local_row, entry_idx))
        if not groups:
            continue

        hot_groups: list[tuple[float, int, list[tuple[int, int, int]]]] = []
        cold_entries: list[tuple[float, int, int]] = []
        for k_col, entries in groups.items():
            score = float(sum(float(abs_values_cpu[entry_idx].item()) for _row, _local, entry_idx in entries))
            if len(entries) >= min_hot_freq:
                hot_groups.append((score, k_col, entries))
                full_hot_group_count += 1
            else:
                for row, _local_row, entry_idx in entries:
                    cold_entries.append((float(abs_values_cpu[entry_idx].item()), row, entry_idx))

        hot_groups.sort(key=lambda item: (-item[0], item[1]))
        cold_entries.sort(key=lambda item: (-item[0], item[2]))
        chosen_hot = hot_groups[:max_hot_groups_per_block] if max_hot_groups_per_block else []
        chosen_cold = cold_entries[:max_cold_entries_per_block] if max_cold_entries_per_block else []
        full_cold_nnz += len(cold_entries)
        if not chosen_hot and not chosen_cold:
            continue

        active_mblocks.append(block)
        for score, k_col, entries in chosen_hot:
            group_ks.append(k_col)
            selected_hot_groups += 1
            for _row, local_row, entry_idx in entries:
                entry_rows.append(local_row)
                hot_value_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
                covered_abs_mass += float(abs_values_cpu[entry_idx].item())
                covered_hot_nnz += 1
            entry_offsets.append(len(entry_rows))
        for score, row, entry_idx in chosen_cold:
            cold_entries_by_row[row].append(entry_idx)
            covered_abs_mass += score
            covered_cold_nnz += 1
        group_offsets.append(len(group_ks))

    cold_row_offsets = [0]
    cold_ks: list[int] = []
    cold_value_chunks: list[torch.Tensor] = []
    for row_entries in cold_entries_by_row:
        row_entries.sort()
        for entry_idx in row_entries:
            cold_ks.append(int(row_ks_cpu[entry_idx].item()))
            cold_value_chunks.append(row_values_cpu[entry_idx : entry_idx + 1])
        cold_row_offsets.append(len(cold_ks))

    device = payload.row_offsets.device
    if not active_mblocks:
        active_mblocks = [0]
        group_offsets = [0, 0]
    if hot_value_chunks:
        hot_values_cpu = torch.cat(hot_value_chunks, dim=0).contiguous()
    else:
        hot_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)
    if cold_value_chunks:
        cold_values_cpu = torch.cat(cold_value_chunks, dim=0).contiguous()
    else:
        cold_values_cpu = torch.empty((0,), dtype=payload.row_values.dtype)

    kmajor_payload = KMajorPayload(
        group_offsets=torch.tensor(group_offsets, device=device, dtype=torch.int32),
        group_ks=torch.tensor(group_ks, device=device, dtype=torch.int32),
        entry_offsets=torch.tensor(entry_offsets, device=device, dtype=torch.int32),
        entry_rows=torch.tensor(entry_rows, device=device, dtype=torch.int32),
        entry_values=hot_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        active_mblocks=torch.tensor(active_mblocks, device=device, dtype=torch.int32),
    )
    cold_payload = RowIndexedPayload(
        row_offsets=torch.tensor(cold_row_offsets, device=device, dtype=torch.int32),
        row_ks=torch.tensor(cold_ks, device=device, dtype=torch.int32),
        row_values=cold_values_cpu.to(device=device, dtype=payload.row_values.dtype).contiguous(),
        selected_count=len(cold_ks),
        r=payload.r,
        kb=payload.kb,
        c=payload.c,
        target_ratio=payload.target_ratio,
    )
    covered_nnz = covered_hot_nnz + covered_cold_nnz
    stats = {
        "total_nnz": float(total_nnz),
        "covered_nnz": float(covered_nnz),
        "covered_hot_nnz": float(covered_hot_nnz),
        "covered_cold_nnz": float(covered_cold_nnz),
        "selected_hot_groups": float(selected_hot_groups),
        "full_hot_groups": float(full_hot_group_count),
        "full_cold_nnz": float(full_cold_nnz),
        "covered_nnz_pct": (covered_nnz / total_nnz * 100.0) if total_nnz else 0.0,
        "covered_abs_mass": covered_abs_mass,
        "total_abs_mass": total_abs_mass,
        "covered_abs_mass_pct": (covered_abs_mass / total_abs_mass * 100.0)
        if total_abs_mass
        else 0.0,
        "effective_ratio": (float(payload.target_ratio or 0.0) * covered_nnz / total_nnz)
        if total_nnz
        else 0.0,
    }
    return kmajor_payload, cold_payload, stats


def preallocated_nvfp4_dense(output: torch.Tensor, qx, qw, *, lineinfo: bool = False) -> torch.Tensor:
    ext = _load_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    return ext.preallocated_nvfp4_dense(output, *_nvfp4_args(qx, qw))


def preallocated_nvfp4_tma_dense(
    output: torch.Tensor,
    qx,
    qw,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_dense_4wg(
    output: torch.Tensor,
    qx,
    qw,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_4wg(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_compact_consumer_posttail(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    prepared_max_row_nnz: int,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    """Run a static-N 4WG compact-input consumer specialization.

    ``prepared_max_row_nnz`` is deliberately explicit: payload preparation must
    prove the compile-time row cap once, outside the timed runtime call.  This
    entry point rejects overflow; use the explicit tail-fallback wrapper when a
    full payload can contain longer rows.
    """
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(
            output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo
        )
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    compiled_cap = int(ext.compact_consumer_max_nnz())
    if int(prepared_max_row_nnz) < 0 or int(prepared_max_row_nnz) > compiled_cap:
        raise ValueError(
            "compact consumer post-tail requires prepared max row nnz "
            f"<= compiled cap {compiled_cap}"
        )
    return _preallocated_nvfp4_tma_compact_consumer_posttail_launch(
        ext,
        output,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        lineinfo=lineinfo,
    )


def _preallocated_nvfp4_tma_compact_consumer_posttail_launch(
    ext,
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None,
    lineinfo: bool,
) -> torch.Tensor:
    a_data, _a_scale, b_data, _b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    compiled_n = int(ext.compact_consumer_static_n())
    if n != compiled_n:
        raise ValueError(
            f"output N={n} does not match compiled compact-consumer N={compiled_n}"
        )
    if tuple(b_comp.shape) != (k, n) or b_comp.dtype != torch.bfloat16:
        raise ValueError("b_comp must be BF16 [K, N]")
    if b_comp.device != output.device:
        raise ValueError("b_comp must be on the output device")
    if int(payload.row_offsets.numel()) != m + 1:
        raise ValueError("row_offsets must contain M+1 elements")
    if int(payload.row_ks.numel()) != int(payload.row_values.numel()):
        raise ValueError("row_ks and row_values length mismatch")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_compact_consumer_posttail_with_tail_fallback(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    weight_t_bf16: torch.Tensor,
    overflow_rows: torch.Tensor,
    *,
    prepared_max_row_nnz: int,
    flat_indices: torch.Tensor | None = None,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    """Run bounded same-kernel correction plus an exact post-store row tail.

    The compact consumer handles the first ``compiled_cap`` entries of every
    row.  Only rows listed in ``overflow_rows`` are revisited, starting at that
    cap, so entries are neither silently truncated nor recomputed.  Payload
    preparation owns ``prepared_max_row_nnz`` and ``overflow_rows``; this hot
    path performs no device-to-host synchronization.
    """
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(
            output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo
        )
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    compiled_cap = int(ext.compact_consumer_max_nnz())
    if int(prepared_max_row_nnz) < 0:
        raise ValueError("prepared_max_row_nnz must be non-negative")
    overflow_rows = overflow_rows.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(prepared_max_row_nnz) > compiled_cap and overflow_rows.numel() == 0:
        raise ValueError(
            "payload exceeds the compiled compact cap but overflow_rows is empty"
        )
    _preallocated_nvfp4_tma_compact_consumer_posttail_launch(
        ext,
        output,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        lineinfo=lineinfo,
    )
    if int(prepared_max_row_nnz) <= compiled_cap:
        return output
    return sparse_active_row_value_payload_vec8_inplace_skip_vstore(
        output,
        payload,
        weight_t_bf16,
        overflow_rows,
        k=int(weight_t_bf16.shape[0]),
        skip_per_row=compiled_cap,
        flat_indices=flat_indices,
        lineinfo=lineinfo,
    )


def preallocated_nvfp4_tma_active_row_ready_flags_vstore(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    flat_indices: torch.Tensor | None = None,
    worker_blocks: int = 0,
    sleep_ns: int = 64,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0 or int(active_rows.numel()) == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(weight_t_bf16.shape) != (k, n):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if weight_t_bf16.dtype != torch.bfloat16 or weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be BF16 CUDA on the output device")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(payload, m, k)
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        payload.row_offsets.contiguous(),
        active_rows,
        m,
        k,
        n,
        int(worker_blocks),
        int(sleep_ns),
    )


def preallocated_nvfp4_tma_active_row_ready_queue_vstore(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    flat_indices: torch.Tensor | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows_local: torch.Tensor | None = None,
    worker_blocks: int = 0,
    worker_threads: int = 256,
    sleep_ns: int = 64,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(weight_t_bf16.shape) != (k, n):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if weight_t_bf16.dtype != torch.bfloat16 or weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be BF16 CUDA on the output device")
    if int(worker_threads) not in (128, 256):
        raise ValueError("worker_threads must be 128 or 256")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(payload, m, k)
    if active_row_offsets is None or active_rows_local is None:
        active_row_offsets, active_rows_local = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows_local = active_rows_local.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        payload.row_offsets.contiguous(),
        active_row_offsets,
        active_rows_local,
        m,
        k,
        n,
        int(worker_blocks),
        int(worker_threads),
        int(sleep_ns),
    )


def preallocated_nvfp4_tma_active_mtile_ready_queue_vstore(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    flat_indices: torch.Tensor | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows_local: torch.Tensor | None = None,
    worker_blocks: int = 0,
    worker_threads: int = 256,
    sleep_ns: int = 64,
    mtile_slices: int = 8,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(weight_t_bf16.shape) != (k, n):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if weight_t_bf16.dtype != torch.bfloat16 or weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be BF16 CUDA on the output device")
    if int(worker_threads) not in (128, 256):
        raise ValueError("worker_threads must be 128 or 256")
    if int(mtile_slices) < 1:
        raise ValueError("mtile_slices must be >= 1")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(payload, m, k)
    if active_row_offsets is None or active_rows_local is None:
        active_row_offsets, active_rows_local = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows_local = active_rows_local.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        payload.row_offsets.contiguous(),
        active_row_offsets,
        active_rows_local,
        m,
        k,
        n,
        int(worker_blocks),
        int(worker_threads),
        int(sleep_ns),
        int(mtile_slices),
    )


def preallocated_nvfp4_tma_direct_add_active(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = balanced_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_direct_add_active_4wg(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense_4wg(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = balanced_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_direct_smem_active(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    direct_smem_mode: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = balanced_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(direct_smem_mode),
    )


def preallocated_nvfp4_tma_direct_smem_delta_active(
    output: torch.Tensor,
    qx,
    qw,
    delta_output: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor,
    active_rows: torch.Tensor,
    direct_smem_mode: int = 11,
    lineinfo: bool = False,
) -> torch.Tensor:
    if int(direct_smem_mode) not in (11, 12):
        raise ValueError("direct_smem_delta_active expects direct_smem_mode=11 or 12")
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(delta_output.shape) != (m, n):
        raise ValueError("delta_output shape must be [M, N]")
    if delta_output.dtype != torch.bfloat16:
        raise ValueError("delta_output must be BF16")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    active_row_offsets = active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        delta_output.contiguous(),
        active_row_offsets,
        active_rows,
        m,
        k,
        n,
        int(direct_smem_mode),
    )


def preallocated_nvfp4_tma_tile_cols_vec16(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(n) % 16 != 0:
        raise ValueError("N must be divisible by 16")
    if int(k) > 32767:
        raise ValueError("tile-col int16-column path requires K <= 32767")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    outlier_cols = payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        outlier_cols,
        b_comp.contiguous(),
        payload.row_offsets.contiguous(),
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_tile_cols_vec16_threads(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    correction_threads: int = 256,
    lineinfo: bool = False,
) -> torch.Tensor:
    if int(correction_threads) not in (128, 256):
        raise ValueError("correction_threads must be 128 or 256")
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(n) % 16 != 0:
        raise ValueError("N must be divisible by 16")
    if int(k) > 32767:
        raise ValueError("tile-col int16-column path requires K <= 32767")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    outlier_cols = payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        outlier_cols,
        b_comp.contiguous(),
        payload.row_offsets.contiguous(),
        m,
        k,
        n,
        int(correction_threads),
    )


def preallocated_nvfp4_tma_tile_cols_vec32(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(n) % 32 != 0:
        raise ValueError("N must be divisible by 32")
    if int(k) > 32767:
        raise ValueError("tile-col int16-column path requires K <= 32767")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    outlier_cols = payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        outlier_cols,
        b_comp.contiguous(),
        payload.row_offsets.contiguous(),
        m,
        k,
        n,
    )


def preallocated_nvfp4_tma_persistent_cols_vec16(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    worker_blocks: int = 128,
    worker_threads: int = 256,
    scheduler_mode: int = 0,
    sleep_ns: int = 0,
    start_delay_us: int = 0,
    lineinfo: bool = False,
) -> torch.Tensor:
    if int(worker_threads) not in (128, 256):
        raise ValueError("worker_threads must be 128 or 256")
    if int(scheduler_mode) not in (0, 1, 2, 3):
        raise ValueError("scheduler_mode must be 0, 1, 2, or 3")
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(n) % 16 != 0:
        raise ValueError("N must be divisible by 16")
    if int(k) > 32767:
        raise ValueError("persistent int16-column path requires K <= 32767")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    outlier_cols = payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16(
        output,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_values.contiguous(),
        outlier_cols,
        b_comp.contiguous(),
        payload.row_offsets.contiguous(),
        m,
        k,
        n,
        int(worker_blocks),
        int(worker_threads),
        int(scheduler_mode),
        int(sleep_ns),
        int(start_delay_us),
    )


def preallocated_nvfp4_tma_loadfma_probe_active(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 4,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 4:
        raise ValueError("persistent active-rowblock sidewarp probe_counter must have at least four elements")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    sparse_warps = sparse_warpgroups_abs * 4
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * sparse_warps:
        raise ValueError("probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    active_counts = active_row_offsets[1:] - active_row_offsets[:-1]
    active_mblocks = torch.nonzero(active_counts > 0, as_tuple=False).flatten().to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(active_mblocks.numel()) == 0:
        active_mblocks = torch.zeros((1,), device=output.device, dtype=torch.int32)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_mblocks,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_probe_active_sidewarp(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("same-CTA sidewarp path currently supports exactly one sparse warpgroup")
    sparse_warps = sparse_warpgroups_abs * 4
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * sparse_warps:
        raise ValueError("probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    active_counts = active_row_offsets[1:] - active_row_offsets[:-1]
    active_mblocks = torch.nonzero(active_counts > 0, as_tuple=False).flatten().to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(active_mblocks.numel()) == 0:
        active_mblocks = torch.zeros((1,), device=output.device, dtype=torch.int32)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_mblocks,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_probe_kmajor(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    kmajor_payload: KMajorPayload | None = None,
    sparse_warpgroups: int = 4,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    sparse_warps = sparse_warpgroups_abs * 4
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * sparse_warps:
        raise ValueError("probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if kmajor_payload is None:
        kmajor_payload = build_kmajor_payload_from_row_payload(payload, m, bm=128)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_active(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 4:
        raise ValueError("probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    active_counts = active_row_offsets[1:] - active_row_offsets[:-1]
    active_mblocks = torch.nonzero(active_counts > 0, as_tuple=False).flatten().to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(active_mblocks.numel()) == 0:
        active_mblocks = torch.zeros((1,), device=output.device, dtype=torch.int32)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_mblocks,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_active_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 4:
        raise ValueError("probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    active_counts = active_row_offsets[1:] - active_row_offsets[:-1]
    active_mblocks = torch.nonzero(active_counts > 0, as_tuple=False).flatten().to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(active_mblocks.numel()) == 0:
        active_mblocks = torch.zeros((1,), device=output.device, dtype=torch.int32)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_mblocks,
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_rowblock_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("rowblock sidewarp path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("rowblock sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_active_rowblock_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("active-rowblock sidewarp path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("active-rowblock sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if active_rows.device != output.device or active_rows.dtype != torch.int32:
        raise ValueError("active_rows must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("active-rowblock static persistent sidewarp path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("active-rowblock static persistent sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if int(active_rows.numel()) < 1:
        raise ValueError("active_rows must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        active_rows.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_only_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if active_rows.device != output.device or active_rows.dtype != torch.int32:
        raise ValueError("active_rows must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("active-rowblock static persistent sidewarp no-store path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("active-rowblock static persistent sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if int(active_rows.numel()) < 1:
        raise ValueError("active_rows must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        active_rows.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def _preallocated_nvfp4_tma_loadfma_warp256_active_rowblock_static_persistent_sidewarp(
    ext_name: str,
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None,
    sparse_warpgroups: int,
    lineinfo: bool,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if active_rows.device != output.device or active_rows.dtype != torch.int32:
        raise ValueError("active_rows must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    if abs(int(sparse_warpgroups)) != 1:
        raise ValueError("active-rowblock static persistent warp256 sidewarp currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("active-rowblock static persistent warp256 sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if int(active_rows.numel()) < 1:
        raise ValueError("active_rows must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return getattr(ext, ext_name)(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        active_rows.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    return _preallocated_nvfp4_tma_loadfma_warp256_active_rowblock_static_persistent_sidewarp(
        "preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp",
        output,
        delta_output,
        probe_sink,
        probe_counter,
        active_rowblocks,
        active_rows,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        sparse_warpgroups=sparse_warpgroups,
        lineinfo=lineinfo,
    )


def preallocated_nvfp4_tma_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    return _preallocated_nvfp4_tma_loadfma_warp256_active_rowblock_static_persistent_sidewarp(
        "preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp",
        output,
        delta_output,
        probe_sink,
        probe_counter,
        active_rowblocks,
        active_rows,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        sparse_warpgroups=sparse_warpgroups,
        lineinfo=lineinfo,
    )


def preallocated_nvfp4_tma_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    return _preallocated_nvfp4_tma_loadfma_warp256_active_rowblock_static_persistent_sidewarp(
        "preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp",
        output,
        delta_output,
        probe_sink,
        probe_counter,
        active_rowblocks,
        active_rows,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        sparse_warpgroups=sparse_warpgroups,
        lineinfo=lineinfo,
    )


def preallocated_nvfp4_tma_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    return _preallocated_nvfp4_tma_loadfma_warp256_active_rowblock_static_persistent_sidewarp(
        "preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp",
        output,
        delta_output,
        probe_sink,
        probe_counter,
        active_rowblocks,
        active_rows,
        qx,
        qw,
        payload,
        b_comp,
        scale_tiles=scale_tiles,
        sparse_warpgroups=sparse_warpgroups,
        lineinfo=lineinfo,
    )


def preallocated_nvfp4_tma_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    active_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(delta_output.shape) != (int(active_rows.numel()), n):
        raise ValueError("compact delta_output shape must be [active_rows, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if active_rows.device != output.device or active_rows.dtype != torch.int32:
        raise ValueError("active_rows must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    if abs(int(sparse_warpgroups)) not in (1, 2):
        raise ValueError("compact persistent path supports one or two sparse warpgroups")
    if int(probe_sink.numel()) < 1:
        raise ValueError("compact persistent path needs one dummy sink element")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if int(active_rows.numel()) < 1:
        raise ValueError("active_rows must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        active_rows.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


preallocated_nvfp4_tma_loadfma_write_compact_noprefetch_persistent = (
    preallocated_nvfp4_tma_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp
)


def preallocated_nvfp4_tma_adaptive_compact_persistent(
    output: torch.Tensor,
    light_delta: torch.Tensor,
    dense_residual: torch.Tensor,
    dense_correction: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    light_rows: torch.Tensor,
    heavy_rows: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 2,
) -> torch.Tensor:
    if int(light_rows.numel()) > 0:
        preallocated_nvfp4_tma_loadfma_write_compact_noprefetch_persistent(
            output,
            light_delta,
            probe_sink,
            probe_counter,
            active_rowblocks,
            light_rows,
            qx,
            qw,
            payload,
            b_comp,
            scale_tiles=scale_tiles,
            sparse_warpgroups=sparse_warpgroups,
        )
    else:
        preallocated_nvfp4_tma_dense(
            output, qx, qw, scale_tiles=scale_tiles
        )
    if int(heavy_rows.numel()) > 0:
        build_compact_dense_residual_active_rows(
            dense_residual, payload, heavy_rows, k=int(weight_t.shape[0])
        )
        torch.mm(dense_residual, weight_t, out=dense_correction)
    if int(light_rows.numel()) > 0 and int(heavy_rows.numel()) > 0:
        merge_two_compact_delta_active_rows(
            output,
            light_delta,
            light_rows,
            dense_correction,
            heavy_rows,
        )
    elif int(light_rows.numel()) > 0:
        merge_compact_delta_active_rows(output, light_delta, light_rows)
    elif int(heavy_rows.numel()) > 0:
        merge_compact_delta_active_rows(output, dense_correction, heavy_rows)
    return output


def preallocated_nvfp4_tma_loadfma_write_active_rowblock_persistent_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    task_records: torch.Tensor | None = None,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if task_records is not None:
        if task_records.device != output.device or task_records.dtype != torch.int32:
            raise ValueError("task_records must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("persistent active-rowblock sidewarp path currently supports exactly one sparse warpgroup")
    if int(probe_sink.numel()) < 4:
        raise ValueError("persistent active-rowblock sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    active_or_records = task_records if task_records is not None else active_rowblocks
    if int(active_or_records.numel()) < 1:
        raise ValueError("persistent active-rowblock task list must be non-empty")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    probe_counter.zero_()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_or_records.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_wg3_ready_active_direct_add(
    output: torch.Tensor,
    ready_flags: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows_local: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if ready_flags.device != output.device or ready_flags.dtype != torch.int32:
        raise ValueError("ready_flags must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("WG3 ready active direct-add path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(ready_flags.numel()) < dense_tiles:
        raise ValueError("ready_flags needs one int32 per dense tile")
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("WG3 ready direct-add probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if b_comp.dtype != torch.bfloat16 or b_comp.device != output.device:
        raise ValueError("b_comp must be BF16 CUDA on the output device")
    if n % 8 != 0:
        raise ValueError("WG3 ready active direct-add path requires N divisible by 8")
    if active_row_offsets is None or active_rows_local is None:
        active_row_offsets, active_rows_local = compact_active_rows_by_block_from_offsets(
            payload.row_offsets,
            m,
            bm=128,
        )
    if active_row_offsets.device != output.device or active_row_offsets.dtype != torch.int32:
        raise ValueError("active_row_offsets must be CUDA int32 on the output device")
    if active_rows_local.device != output.device or active_rows_local.dtype != torch.int32:
        raise ValueError("active_rows_local must be CUDA int32 on the output device")
    if int(active_row_offsets.numel()) != ((m + 127) // 128) + 1:
        raise ValueError("active_row_offsets must have one offset per M tile plus one")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add(
        output,
        ready_flags.contiguous(),
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets.contiguous(),
        active_rows_local.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_packed_rowblock_sidewarp(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    active_rowblocks: torch.Tensor,
    packed_payload: PackedLocalDeltaPayload,
    qx,
    qw,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    if int(packed_payload.entry_records.numel()) == 0:
        return preallocated_nvfp4_tma_dense(output, qx, qw, scale_tiles=scale_tiles, lineinfo=lineinfo)
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if active_rowblocks.device != output.device or active_rowblocks.dtype != torch.int32:
        raise ValueError("active_rowblocks must be CUDA int32 on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    if packed_payload.tile_offsets.device != output.device or packed_payload.tile_offsets.dtype != torch.int32:
        raise ValueError("packed tile_offsets must be CUDA int32 on the output device")
    if packed_payload.row_records.device != output.device or packed_payload.row_records.dtype != torch.int64:
        raise ValueError("packed row_records must be CUDA int64 on the output device")
    if packed_payload.entry_records.device != output.device or packed_payload.entry_records.dtype != torch.int32:
        raise ValueError("packed entry_records must be CUDA int32 on the output device")
    if int(probe_counter.numel()) < 1:
        raise ValueError("probe_counter must have at least one element")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs != 1:
        raise ValueError("packed rowblock sidewarp path currently supports exactly one sparse warpgroup")
    dense_tiles = ((m + 127) // 128) * ((n + 127) // 128)
    if int(probe_sink.numel()) < dense_tiles * 4:
        raise ValueError("packed rowblock sidewarp probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if int(active_rowblocks.numel()) < 1:
        raise ValueError("active_rowblocks must be non-empty")
    if int(packed_payload.tile_offsets.numel()) != int(active_rowblocks.numel()) + 1:
        raise ValueError("packed tile_offsets must be active_rowblocks+1")
    if int(packed_payload.row_records.numel()) != int(active_rowblocks.numel()) * 8:
        raise ValueError("packed rowblock payload expects exactly 8 row records per rowblock")
    if int(packed_payload.payload_mode) != 4:
        raise ValueError("packed rowblock payload_mode must be 4")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        active_rowblocks.contiguous(),
        packed_payload.tile_offsets.contiguous(),
        packed_payload.row_records.contiguous(),
        packed_payload.entry_records.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_probe_hybrid(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 4,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * max(24, sparse_warpgroups_abs * 4):
        raise ValueError("hybrid probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_probe_incta_hybrid(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 8:
        raise ValueError("in-CTA hybrid probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_probe_idlechunk_hybrid(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    group_budget: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    group_budget = int(group_budget)
    if group_budget < 1:
        raise ValueError("group_budget must be positive")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 2:
        raise ValueError("idlechunk hybrid probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid(
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        group_budget,
    )


def preallocated_nvfp4_tma_loadfma_probe_scheduler_hybrid(
    output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    group_budget: int = 1,
    side_warps: int = 1,
    side_mode: str = "tail",
    phase_trace: torch.Tensor | None = None,
    phase_trace_max_ctas: int | None = None,
    phase_trace_stride: int | None = None,
    probe_do_math: bool = True,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    group_budget = int(group_budget)
    if group_budget < 1:
        raise ValueError("group_budget must be positive")
    side_warps = int(side_warps)
    if side_warps < 1 or side_warps > 4:
        raise ValueError("side_warps must be in [1, 4]")
    side_mode_id = {
        "tail": 0,
        "idle_stage": 1,
        "extra_wg": 2,
        "extra_wg_idle": 3,
        "extra_wg_noprobe": 4,
        "extra_wg_write_noprobe": 5,
        "extra_wg_add_noprobe": 6,
        "extra_wg_rowadd_noprobe": 7,
        "extra_wg_metaadd_noprobe": 8,
        "extra_wg_rowchunk_add_noprobe": 9,
        "extra_wg_rowchunk_atomic_noprobe": 10,
        "extra_wg_smem_merge_noprobe": 11,
        "extra_wg_sharedacc_smem_noprobe": 12,
        "extra_wg_subacc32_smem_noprobe": 13,
        "extra_wg_local_delta_smem_noprobe": 14,
        "extra_wg_local_delta_hotsched_smem_noprobe": 15,
        "extra_wg_local_delta_hotsched_reg_smem_noprobe": 16,
        "extra_wg_local_delta_hotsched_db_smem_noprobe": 17,
        "extra_wg_local_delta_hotsched_colsplit_db_smem_noprobe": 18,
        "extra_wg_local_delta_hotsched_db_heavyrowadd_smem_noprobe": 19,
        "extra_wg_local_delta_hotsched_db_heavysmem_smem_noprobe": 20,
        "extra_wg_local_delta_hotsched_db_superhot_colsplit_smem_noprobe": 21,
        "extra_wg_local_delta_hotsched_db_superhot_atomic_smem_noprobe": 22,
        "extra_wg_local_delta_hotsched_db_heavypipe_smem_noprobe": 23,
        "extra_wg_local_delta_hotsched_db_sidesmemheavy_smem_noprobe": 24,
        "extra_wg_kmajor_local_delta_db_smem_noprobe": 25,
        "extra_wg_local_delta_stage_split_smem_noprobe": 26,
        "extra_wg_kmajor_local_delta_stage_split_smem_noprobe": 27,
        "extra_wg_kmajor_local_delta_stage_split_prestore_smem_noprobe": 28,
        "extra_wg_kmajor_local_delta_stage_split_side_merge_smem_noprobe": 29,
        "extra_wg_kmajor_local_delta_stage_split_all_lane_wait_smem_noprobe": 30,
    }.get(str(side_mode))
    if side_mode_id is None:
        raise ValueError("unsupported side_mode")
    if side_mode_id == 1 and side_warps != 2:
        raise ValueError("idle_stage side_mode currently requires side_warps=2")
    if side_mode_id == 3 and side_warps != 4:
        raise ValueError("extra_wg_idle side_mode currently requires side_warps=4")
    if side_mode_id in {18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29} and side_warps != 4:
        raise ValueError("this local-delta side_mode currently requires side_warps=4")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * side_warps:
        raise ValueError("scheduler hybrid probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    call_args = (
        output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        group_budget if bool(probe_do_math) else -group_budget,
        side_warps,
        side_mode_id,
    )
    if phase_trace is not None:
        if phase_trace.device != output.device or phase_trace.dtype != torch.int64:
            raise ValueError("phase_trace must be CUDA int64 on the output device")
        if not phase_trace.is_contiguous() or phase_trace.dim() != 2:
            raise ValueError("phase_trace must be contiguous [ctas, slots]")
        trace_stride = int(phase_trace_stride or phase_trace.shape[1])
        trace_max_ctas = int(
            phase_trace.shape[0] if phase_trace_max_ctas is None else phase_trace_max_ctas
        )
        if trace_stride < 52 or trace_stride > int(phase_trace.shape[1]):
            raise ValueError("phase_trace_stride must be in [52, phase_trace.shape[1]]")
        if trace_max_ctas < 0 or trace_max_ctas > int(phase_trace.shape[0]):
            raise ValueError("phase_trace_max_ctas must be in [0, phase_trace.shape[0]]")
        return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_phase_trace(
            *call_args,
            phase_trace,
            trace_max_ctas,
            trace_stride,
        )
    if not bool(probe_do_math):
        raise ValueError("probe_do_math=False requires phase_trace path")
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid(
        *call_args
    )


def preallocated_nvfp4_tma_loadfma_write_scheduler_hybrid(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    group_budget: int = 1,
    direct_delta_write_mode: int = 1,
    direct_delta_chunk_limit: int = 0,
    side_warps: int = 1,
    side_mode: str = "tail",
    phase_trace: torch.Tensor | None = None,
    phase_trace_max_ctas: int | None = None,
    phase_trace_stride: int | None = None,
    packed_payload: PackedLocalDeltaPayload | None = None,
    kmajor_tile_meta: KMajorTileMetadata | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    direct_delta_write_mode = int(direct_delta_write_mode)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if direct_delta_write_mode == 4:
        if delta_output.dim() != 2 or int(delta_output.shape[1]) != n:
            raise ValueError("compact delta_output must be [entries, N]")
        if int(delta_output.shape[0]) < int(kmajor_payload.entry_values.numel()):
            raise ValueError("compact delta_output must cover kmajor entries")
        if delta_output.dtype != torch.bfloat16:
            raise ValueError("compact delta_output must be BF16")
    elif tuple(delta_output.shape) != (m, n) or delta_output.dtype != torch.bfloat16:
        raise ValueError("delta_output must be BF16 [M, N]")
    if delta_output.device != output.device:
        raise ValueError("delta_output must be on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    group_budget = int(group_budget)
    if group_budget < 1:
        raise ValueError("group_budget must be positive")
    direct_delta_chunk_limit = int(direct_delta_chunk_limit)
    if direct_delta_chunk_limit < 0:
        raise ValueError("direct_delta_chunk_limit must be non-negative")
    side_warps = int(side_warps)
    if side_warps < 1 or side_warps > 4:
        raise ValueError("side_warps must be in [1, 4]")
    side_mode_id = {
        "tail": 0,
        "idle_stage": 1,
        "extra_wg": 2,
        "extra_wg_idle": 3,
        "extra_wg_noprobe": 4,
        "extra_wg_write_noprobe": 5,
        "extra_wg_add_noprobe": 6,
        "extra_wg_rowadd_noprobe": 7,
        "extra_wg_metaadd_noprobe": 8,
        "extra_wg_rowchunk_add_noprobe": 9,
        "extra_wg_rowchunk_atomic_noprobe": 10,
        "extra_wg_smem_merge_noprobe": 11,
        "extra_wg_sharedacc_smem_noprobe": 12,
        "extra_wg_subacc32_smem_noprobe": 13,
        "extra_wg_local_delta_smem_noprobe": 14,
        "extra_wg_local_delta_hotsched_smem_noprobe": 15,
        "extra_wg_local_delta_hotsched_reg_smem_noprobe": 16,
        "extra_wg_local_delta_hotsched_db_smem_noprobe": 17,
        "extra_wg_local_delta_hotsched_colsplit_db_smem_noprobe": 18,
        "extra_wg_local_delta_hotsched_db_heavyrowadd_smem_noprobe": 19,
        "extra_wg_local_delta_hotsched_db_heavysmem_smem_noprobe": 20,
        "extra_wg_local_delta_hotsched_db_superhot_colsplit_smem_noprobe": 21,
        "extra_wg_local_delta_hotsched_db_superhot_atomic_smem_noprobe": 22,
        "extra_wg_local_delta_hotsched_db_heavypipe_smem_noprobe": 23,
        "extra_wg_local_delta_hotsched_db_sidesmemheavy_smem_noprobe": 24,
        "extra_wg_kmajor_local_delta_db_smem_noprobe": 25,
        "extra_wg_local_delta_stage_split_smem_noprobe": 26,
        "extra_wg_kmajor_local_delta_stage_split_smem_noprobe": 27,
        "extra_wg_kmajor_local_delta_stage_split_prestore_smem_noprobe": 28,
        "extra_wg_kmajor_local_delta_stage_split_side_merge_smem_noprobe": 29,
        "extra_wg_kmajor_local_delta_stage_split_all_lane_wait_smem_noprobe": 30,
    }.get(str(side_mode))
    if side_mode_id is None:
        raise ValueError("unsupported side_mode")
    if side_mode_id == 1 and side_warps != 2:
        raise ValueError("idle_stage side_mode currently requires side_warps=2")
    if side_mode_id == 3 and side_warps != 4:
        raise ValueError("extra_wg_idle side_mode currently requires side_warps=4")
    if side_mode_id in {18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29} and side_warps != 4:
        raise ValueError("this local-delta side_mode currently requires side_warps=4")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * side_warps:
        raise ValueError("write scheduler hybrid probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    call_args = (
        output,
        delta_output.contiguous(),
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        group_budget,
        direct_delta_write_mode,
        side_warps,
        side_mode_id,
        direct_delta_chunk_limit,
    )
    packed_call_args = None
    if packed_payload is not None:
        if (
            packed_payload.tile_offsets.device != output.device
            or packed_payload.tile_offsets.dtype != torch.int32
            or packed_payload.row_records.device != output.device
            or packed_payload.row_records.dtype != torch.int64
            or packed_payload.entry_records.device != output.device
            or packed_payload.entry_records.dtype != torch.int32
        ):
            raise ValueError("packed_payload tensors must be CUDA int32/int64 on output device")
        if (
            packed_payload.tile_offsets.dim() != 1
            or packed_payload.row_records.dim() != 1
            or packed_payload.entry_records.dim() != 1
            or not packed_payload.tile_offsets.is_contiguous()
            or not packed_payload.row_records.is_contiguous()
            or not packed_payload.entry_records.is_contiguous()
        ):
            raise ValueError("packed_payload tensors must be contiguous 1D tensors")
        packed_call_args = (
            packed_payload.tile_offsets,
            packed_payload.row_records,
            packed_payload.entry_records,
            int(packed_payload.payload_mode),
        )
    tile_meta_call_args = None
    if kmajor_tile_meta is not None:
        if (
            kmajor_tile_meta.group_starts.device != output.device
            or kmajor_tile_meta.group_starts.dtype != torch.int32
            or kmajor_tile_meta.group_counts.device != output.device
            or kmajor_tile_meta.group_counts.dtype != torch.int32
            or kmajor_tile_meta.group_meta.device != output.device
            or kmajor_tile_meta.group_meta.dtype != torch.int64
        ):
            raise ValueError(
                "kmajor_tile_meta starts/counts must be CUDA int32 and meta must be CUDA int64 on output device"
            )
        tile_count = (m + 127) // 128
        if (
            kmajor_tile_meta.group_starts.dim() != 1
            or kmajor_tile_meta.group_counts.dim() != 1
            or int(kmajor_tile_meta.group_starts.numel()) != tile_count
            or int(kmajor_tile_meta.group_counts.numel()) != tile_count
            or kmajor_tile_meta.group_meta.dim() != 1
            or int(kmajor_tile_meta.group_meta.numel()) != tile_count
            or not kmajor_tile_meta.group_starts.is_contiguous()
            or not kmajor_tile_meta.group_counts.is_contiguous()
            or not kmajor_tile_meta.group_meta.is_contiguous()
        ):
            raise ValueError("kmajor_tile_meta tensors must be contiguous [tile_m]")
        tile_meta_call_args = (
            kmajor_tile_meta.group_starts,
            kmajor_tile_meta.group_counts,
            kmajor_tile_meta.group_meta,
        )
    if phase_trace is not None:
        if phase_trace.device != output.device or phase_trace.dtype != torch.int64:
            raise ValueError("phase_trace must be CUDA int64 on the output device")
        if not phase_trace.is_contiguous() or phase_trace.dim() != 2:
            raise ValueError("phase_trace must be contiguous [ctas, slots]")
        trace_stride = int(phase_trace_stride or phase_trace.shape[1])
        trace_max_ctas = int(
            phase_trace.shape[0] if phase_trace_max_ctas is None else phase_trace_max_ctas
        )
        if trace_stride < 52 or trace_stride > int(phase_trace.shape[1]):
            raise ValueError("phase_trace_stride must be in [52, phase_trace.shape[1]]")
        if trace_max_ctas < 0 or trace_max_ctas > int(phase_trace.shape[0]):
            raise ValueError("phase_trace_max_ctas must be in [0, phase_trace.shape[0]]")
        if packed_payload is not None and kmajor_tile_meta is not None:
            return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed_tile_meta(
                *call_args,
                phase_trace,
                trace_max_ctas,
                trace_stride,
                *packed_call_args,
                *tile_meta_call_args,
            )
        if kmajor_tile_meta is not None:
            return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_tile_meta(
                *call_args,
                phase_trace,
                trace_max_ctas,
                trace_stride,
                *tile_meta_call_args,
            )
        if packed_payload is not None:
            return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed(
                *call_args,
                phase_trace,
                trace_max_ctas,
                trace_stride,
                *packed_call_args,
            )
        return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace(
            *call_args,
            phase_trace,
            trace_max_ctas,
            trace_stride,
        )
    if packed_payload is not None and kmajor_tile_meta is not None:
        return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed_tile_meta(
            *call_args,
            *packed_call_args,
            *tile_meta_call_args,
        )
    if packed_payload is not None:
        return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed(
            *call_args,
            *packed_call_args,
        )
    if kmajor_tile_meta is not None:
        return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_tile_meta(
            *call_args,
            *tile_meta_call_args,
        )
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid(*call_args)


def preallocated_nvfp4_tma_loadfma_write_incta_kmajor_atomic(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 8:
        raise ValueError("in-CTA kmajor atomic write probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_incta_kmajor_direct(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 8:
        raise ValueError("in-CTA kmajor direct write probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_direct(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_incta_kmajor_sharedacc(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    row_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 8:
        raise ValueError("in-CTA kmajor sharedacc write probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            row_payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_sharedacc(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        row_payload.row_offsets.contiguous(),
        row_payload.row_ks.contiguous(),
        row_payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_extrawg_kmajor_sharedacc(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    row_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n) or tuple(delta_output.shape) != (m, n):
        raise ValueError("output and delta_output shapes must be [M, N]")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    min_sink = ((m + 127) // 128) * ((n + 127) // 128) * sparse_warpgroups_abs * 4
    if int(probe_sink.numel()) < min_sink:
        raise ValueError("extra-WG kmajor sharedacc write probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = compact_active_rows_by_block_from_offsets(
            row_payload.row_offsets, m, bm=128
        )
    active_row_offsets = active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        row_payload.row_offsets.contiguous(),
        row_payload.row_ks.contiguous(),
        row_payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def preallocated_nvfp4_tma_loadfma_write_incta_kmajor_entry_direct(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    probe_sink: torch.Tensor,
    probe_counter: torch.Tensor,
    qx,
    qw,
    cold_payload: RowIndexedPayload,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    scale_tiles: TmaScaleTiles | None = None,
    cold_active_row_offsets: torch.Tensor | None = None,
    cold_active_rows: torch.Tensor | None = None,
    sparse_warpgroups: int = 1,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if delta_output.dim() != 2 or int(delta_output.shape[1]) != n:
        raise ValueError("delta_output must be [entries, N]")
    if int(delta_output.shape[0]) < int(kmajor_payload.entry_values.numel()):
        raise ValueError("delta_output must cover kmajor entries")
    if delta_output.dtype != torch.bfloat16 or delta_output.device != output.device:
        raise ValueError("delta_output must be BF16 CUDA on the output device")
    if probe_sink.device != output.device or probe_sink.dtype != torch.float32:
        raise ValueError("probe_sink must be CUDA float32 on the output device")
    if probe_counter.device != output.device or probe_counter.dtype != torch.int32:
        raise ValueError("probe_counter must be CUDA int32 on the output device")
    sparse_warpgroups_abs = abs(int(sparse_warpgroups))
    if sparse_warpgroups_abs < 1 or sparse_warpgroups_abs > 64:
        raise ValueError("sparse_warpgroups must be in [1, 64]")
    if int(probe_sink.numel()) < ((m + 127) // 128) * ((n + 127) // 128) * 8:
        raise ValueError("in-CTA kmajor entry-direct write probe_sink is too small")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if scale_tiles is None:
        scale_tiles = make_tma_scale_tiles(qx, qw, lineinfo=lineinfo)
    if cold_active_row_offsets is None or cold_active_rows is None:
        cold_active_row_offsets, cold_active_rows = compact_active_rows_by_block_from_offsets(
            cold_payload.row_offsets, m, bm=128
        )
    cold_active_row_offsets = cold_active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    cold_active_rows = cold_active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_entry_direct(
        output,
        delta_output,
        probe_sink.contiguous(),
        probe_counter.contiguous(),
        kmajor_payload.active_mblocks.contiguous(),
        a_data,
        scale_tiles.a_scale_tile.contiguous(),
        b_data,
        scale_tiles.b_scale_tile.contiguous(),
        a_amax,
        b_amax,
        cold_payload.row_offsets.contiguous(),
        cold_payload.row_ks.contiguous(),
        cold_payload.row_values.contiguous(),
        cold_active_row_offsets,
        cold_active_rows,
        kmajor_payload.group_offsets.contiguous(),
        kmajor_payload.group_ks.contiguous(),
        kmajor_payload.entry_offsets.contiguous(),
        kmajor_payload.entry_rows.contiguous(),
        kmajor_payload.entry_values.contiguous(),
        b_comp.contiguous(),
        m,
        k,
        n,
        int(sparse_warpgroups),
    )


def merge_entry_delta_active_rows(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    merge_row_offsets: torch.Tensor,
    merge_entry_indices: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    merge_row_offsets = merge_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    merge_entry_indices = merge_entry_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if int(merge_row_offsets.numel()) != int(active_rows.numel()) + 1:
        raise ValueError("merge_row_offsets length must be active_rows + 1")
    return ext.merge_entry_delta_active_rows(
        output,
        delta_entries.contiguous(),
        active_rows,
        merge_row_offsets,
        merge_entry_indices,
    )


def merge_entry_delta_active_rows_fastpath(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    merge_row_offsets: torch.Tensor,
    merge_entry_indices: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    merge_row_offsets = merge_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    merge_entry_indices = merge_entry_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if int(merge_row_offsets.numel()) != int(active_rows.numel()) + 1:
        raise ValueError("merge_row_offsets length must be active_rows + 1")
    return ext.merge_entry_delta_active_rows_fastpath(
        output,
        delta_entries.contiguous(),
        active_rows,
        merge_row_offsets,
        merge_entry_indices,
    )


def merge_entry_delta_active_rows_vec8(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    merge_row_offsets: torch.Tensor,
    merge_entry_indices: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    merge_row_offsets = merge_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    merge_entry_indices = merge_entry_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if int(merge_row_offsets.numel()) != int(active_rows.numel()) + 1:
        raise ValueError("merge_row_offsets length must be active_rows + 1")
    return ext.merge_entry_delta_active_rows_vec8(
        output,
        delta_entries.contiguous(),
        active_rows,
        merge_row_offsets,
        merge_entry_indices,
    )


def merge_entry_delta_active_rows_chunk_prefix_vec8(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    merge_row_offsets: torch.Tensor,
    merge_entry_indices: torch.Tensor,
    *,
    chunk_cols: int,
    chunks_per_row: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    if int(chunk_cols) <= 0 or int(chunk_cols) % 8 != 0:
        raise ValueError("chunk_cols must be a positive multiple of 8")
    if int(chunks_per_row) <= 0:
        raise ValueError("chunks_per_row must be positive")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    merge_row_offsets = merge_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    merge_entry_indices = merge_entry_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if int(merge_row_offsets.numel()) != int(active_rows.numel()) + 1:
        raise ValueError("merge_row_offsets length must be active_rows + 1")
    return ext.merge_entry_delta_active_rows_chunk_prefix_vec8(
        output,
        delta_entries.contiguous(),
        active_rows,
        merge_row_offsets,
        merge_entry_indices,
        int(chunk_cols),
        int(chunks_per_row),
    )


def sparse_active_row_value_payload_vec8_store(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_store(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_store_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_store_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def merge_full_delta_active_rows(
    output: torch.Tensor,
    delta_output: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_output.dim() != 2:
        raise ValueError("output and delta_output must be 2D")
    if output.dtype != torch.bfloat16 or delta_output.dtype != torch.bfloat16:
        raise ValueError("output and delta_output must be BF16")
    if output.device != delta_output.device:
        raise ValueError("output and delta_output must be on the same device")
    if tuple(output.shape) != tuple(delta_output.shape):
        raise ValueError("output and delta_output must have the same shape")
    if not output.is_contiguous() or not delta_output.is_contiguous():
        raise ValueError("output and delta_output must be contiguous")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.merge_full_delta_active_rows(
        output,
        delta_output,
        active_rows,
    )


def merge_compact_delta_active_rows(
    output: torch.Tensor,
    compact_delta: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or compact_delta.dim() != 2:
        raise ValueError("output and compact_delta must be 2D")
    if output.dtype != torch.bfloat16 or compact_delta.dtype != torch.bfloat16:
        raise ValueError("output and compact_delta must be BF16")
    if output.device != compact_delta.device:
        raise ValueError("output and compact_delta must be on the same device")
    if int(compact_delta.shape[0]) != int(active_rows.numel()):
        raise ValueError("compact_delta rows must match active_rows")
    if int(compact_delta.shape[1]) != int(output.shape[1]):
        raise ValueError("compact_delta N must match output N")
    if not output.is_contiguous() or not compact_delta.is_contiguous():
        raise ValueError("output and compact_delta must be contiguous")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.merge_compact_delta_active_rows(
        output,
        compact_delta,
        active_rows,
    )


def merge_two_compact_delta_active_rows(
    output: torch.Tensor,
    first_delta: torch.Tensor,
    first_rows: torch.Tensor,
    second_delta: torch.Tensor,
    second_rows: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or first_delta.dim() != 2 or second_delta.dim() != 2:
        raise ValueError("output and compact deltas must be 2D")
    if any(
        tensor.dtype != torch.bfloat16
        for tensor in (output, first_delta, second_delta)
    ):
        raise ValueError("output and compact deltas must be BF16")
    if any(
        tensor.device != output.device
        for tensor in (first_delta, second_delta)
    ):
        raise ValueError("output and compact deltas must be on the same device")
    if int(first_delta.shape[0]) != int(first_rows.numel()):
        raise ValueError("first_delta rows must match first_rows")
    if int(second_delta.shape[0]) != int(second_rows.numel()):
        raise ValueError("second_delta rows must match second_rows")
    if int(first_delta.shape[1]) != int(output.shape[1]):
        raise ValueError("first_delta N must match output N")
    if int(second_delta.shape[1]) != int(output.shape[1]):
        raise ValueError("second_delta N must match output N")
    if any(
        not tensor.is_contiguous()
        for tensor in (output, first_delta, second_delta)
    ):
        raise ValueError("output and compact deltas must be contiguous")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    first_rows = first_rows.to(device=output.device, dtype=torch.int32).contiguous()
    second_rows = second_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.merge_two_compact_delta_active_rows(
        output,
        first_delta,
        first_rows,
        second_delta,
        second_rows,
    )


def build_padded_light_heavy_rows(
    row_offsets: torch.Tensor,
    *,
    heavy_threshold: int,
    heavy_capacity: int,
    lineinfo: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if row_offsets.dim() != 1 or row_offsets.dtype != torch.int32 or not row_offsets.is_cuda:
        raise ValueError("row_offsets must be a CUDA int32 CSR offset tensor")
    rows = int(row_offsets.numel()) - 1
    if rows <= 0:
        raise ValueError("row_offsets must contain at least two entries")
    if int(heavy_threshold) <= 0:
        raise ValueError("heavy_threshold must be positive")
    if int(heavy_capacity) <= 0 or int(heavy_capacity) > rows:
        raise ValueError("heavy_capacity must be in (0, rows]")
    return tuple(
        ext.build_padded_light_heavy_rows(
            row_offsets.contiguous(),
            int(heavy_threshold),
            int(heavy_capacity),
        )
    )


def build_compact_dense_residual_active_rows(
    residual: torch.Tensor,
    row_payload: RowIndexedPayload,
    active_rows: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if residual.dim() != 2 or residual.dtype != torch.bfloat16 or not residual.is_cuda:
        raise ValueError("residual must be a CUDA BF16 [active_rows, K] tensor")
    if int(residual.shape[0]) != int(active_rows.numel()):
        raise ValueError("residual rows must match active_rows")
    if int(residual.shape[1]) != int(k):
        raise ValueError("residual K must match k")
    if int(k) % 8 != 0:
        raise ValueError("k must be divisible by 8")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous")
    active_rows = active_rows.to(device=residual.device, dtype=torch.int32).contiguous()
    return ext.build_compact_dense_residual_active_rows(
        residual,
        row_payload.row_values.contiguous(),
        row_payload.row_ks.contiguous(),
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace_skip_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    skip_per_row: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    """Add only the CSR suffix beginning at ``skip_per_row`` in each row."""
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if int(skip_per_row) < 0:
        raise ValueError("skip_per_row must be non-negative")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_skip_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
        int(skip_per_row),
    )


def sparse_packed_suffix12_vec8_inplace_vstore(
    output: torch.Tensor,
    packed_suffix_records: torch.Tensor,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    """Add a compact, zero-padded suffix of at most 12 records per row."""
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if (
        packed_suffix_records.dim() != 2
        or packed_suffix_records.dtype != torch.int32
        or int(packed_suffix_records.shape[1]) != 12
    ):
        raise ValueError(
            "packed_suffix_records must be int32 [active_rows, 12]"
        )
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if not 0 < int(weight_t_bf16.shape[0]) <= 8192:
        raise ValueError(
            "packed suffix wave encoding requires 0 < K <= 8192"
        )
    if int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 N must match output N")
    active_rows = active_rows.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    packed_suffix_records = packed_suffix_records.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    if int(packed_suffix_records.shape[0]) != int(active_rows.numel()):
        raise ValueError(
            "packed_suffix_records and active_rows must have the same row count"
        )
    return ext.sparse_packed_suffix12_vec8_inplace_vstore(
        output,
        packed_suffix_records,
        active_rows,
        weight_t_bf16.contiguous(),
    )


def sparse_active_row_value_payload_vec8_inplace_strict_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_strict_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_kmajor_epin_delta_store(
    output: torch.Tensor,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    active_row_offsets: torch.Tensor,
    active_rows: torch.Tensor,
    k: int,
    epin: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if int(epin) not in (32, 64):
        raise ValueError("epin must be one of 32, 64")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if b_comp.dim() != 2 or b_comp.dtype != torch.bfloat16 or not b_comp.is_cuda:
        raise ValueError("b_comp must be CUDA BF16 [K, N]")
    if int(b_comp.shape[0]) != int(k) or int(b_comp.shape[1]) != int(output.shape[1]):
        raise ValueError("b_comp shape must be [K, N]")
    active_row_offsets = active_row_offsets.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_kmajor_epin_delta_store(
        output,
        kmajor_payload.active_mblocks.to(device=output.device, dtype=torch.int32).contiguous(),
        active_row_offsets,
        active_rows,
        kmajor_payload.group_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.group_ks.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_rows.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_values.to(device=output.device, dtype=torch.bfloat16).contiguous(),
        b_comp.contiguous(),
        int(k),
        int(epin),
    )


def sparse_kmajor_epin64_delta_store(
    output: torch.Tensor,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    active_row_offsets: torch.Tensor,
    active_rows: torch.Tensor,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    return sparse_kmajor_epin_delta_store(
        output,
        kmajor_payload,
        b_comp,
        active_row_offsets=active_row_offsets,
        active_rows=active_rows,
        k=k,
        epin=64,
        lineinfo=lineinfo,
    )


def sparse_kmajor_epin64_direct_store(
    output: torch.Tensor,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if b_comp.dim() != 2 or b_comp.dtype != torch.bfloat16 or not b_comp.is_cuda:
        raise ValueError("b_comp must be CUDA BF16 [K, N]")
    if int(b_comp.shape[0]) != int(k) or int(b_comp.shape[1]) != int(output.shape[1]):
        raise ValueError("b_comp shape must be [K, N]")
    return ext.sparse_kmajor_epin64_direct_store(
        output,
        kmajor_payload.active_mblocks.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.group_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.group_ks.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_rows.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_values.to(device=output.device, dtype=torch.bfloat16).contiguous(),
        b_comp.contiguous(),
        int(k),
    )


def sparse_kmajor_serial_group_inplace(
    output: torch.Tensor,
    kmajor_payload: KMajorPayload,
    b_comp: torch.Tensor,
    *,
    k: int,
    bm: int = 128,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if int(output.shape[1]) % 64 != 0:
        raise ValueError("output N must be divisible by 64")
    if b_comp.dim() != 2 or b_comp.dtype != torch.bfloat16 or not b_comp.is_cuda:
        raise ValueError("b_comp must be CUDA BF16 [K, N]")
    if int(b_comp.shape[0]) != int(k) or int(b_comp.shape[1]) != int(output.shape[1]):
        raise ValueError("b_comp shape must be [K, N]")
    if int(bm) not in (32, 64, 128):
        raise ValueError("bm must be one of 32, 64, 128")
    return ext.sparse_kmajor_serial_group_inplace(
        output,
        kmajor_payload.active_mblocks.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.group_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.group_ks.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_rows.to(device=output.device, dtype=torch.int32).contiguous(),
        kmajor_payload.entry_values.to(device=output.device, dtype=torch.bfloat16).contiguous(),
        b_comp.contiguous(),
        int(k),
        int(bm),
    )


def sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace_fastpath(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_fastpath(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_value_payload_vec8_inplace_rowblock(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    if flat_indices is None:
        flat_indices = row_payload_flat_indices(row_payload, int(output.shape[0]), int(k))
    flat_indices = flat_indices.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_value_payload_vec8_inplace_rowblock(
        output,
        row_payload.row_values.contiguous(),
        weight_t_bf16.contiguous(),
        flat_indices,
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_col_value_payload_vec16_inplace(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if int(output.shape[1]) % 16 != 0:
        raise ValueError("output N must be divisible by 16")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if int(k) > 32767:
        raise ValueError("vec16 int16-column path requires K <= 32767")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    outlier_cols = row_payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.sparse_active_row_col_value_payload_vec16_inplace(
        output,
        row_payload.row_values.contiguous(),
        outlier_cols,
        weight_t_bf16.contiguous(),
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def split_hot_dense_padded_cold_rows(
    hot_dense: torch.Tensor,
    cold_values: torch.Tensor,
    cold_cols: torch.Tensor,
    cold_counts: torch.Tensor,
    overflow: torch.Tensor,
    row_payload: RowIndexedPayload,
    hot_lut: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    return tuple(
        ext.split_hot_dense_padded_cold_rows(
            hot_dense.contiguous(),
            cold_values.contiguous(),
            cold_cols.contiguous(),
            cold_counts.contiguous(),
            overflow.contiguous(),
            row_payload.row_values.contiguous(),
            row_payload.row_ks.to(device=hot_dense.device, dtype=torch.int16).contiguous(),
            row_payload.row_offsets.to(device=hot_dense.device, dtype=torch.int32).contiguous(),
            hot_lut.to(device=hot_dense.device, dtype=torch.int16).contiguous(),
        )
    )


def sparse_padded_cold_col_vec16_inplace(
    output: torch.Tensor,
    cold_values: torch.Tensor,
    cold_cols: torch.Tensor,
    cold_counts: torch.Tensor,
    row_payload: RowIndexedPayload,
    hot_lut: torch.Tensor,
    weight_t_bf16: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    return ext.sparse_padded_cold_col_vec16_inplace(
        output,
        cold_values.contiguous(),
        cold_cols.contiguous(),
        cold_counts.contiguous(),
        row_payload.row_values.contiguous(),
        row_payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous(),
        row_payload.row_offsets.to(device=output.device, dtype=torch.int32).contiguous(),
        hot_lut.to(device=output.device, dtype=torch.int16).contiguous(),
        weight_t_bf16.contiguous(),
    )


def sparse_active_row_col_value_payload_vec8_inplace_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if int(k) > 32767:
        raise ValueError("vec8 int16-column path requires K <= 32767")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    outlier_cols = row_payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.sparse_active_row_col_value_payload_vec8_inplace_vstore(
        output,
        row_payload.row_values.contiguous(),
        outlier_cols,
        weight_t_bf16.contiguous(),
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_col_value_payload_vec8_shmem_sum_then_add(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if int(k) > 32767:
        raise ValueError("shared-memory int16-column path requires K <= 32767")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    outlier_cols = row_payload.row_ks.to(device=output.device, dtype=torch.int16).contiguous()
    return ext.sparse_active_row_col_value_payload_vec8_shmem_sum_then_add(
        output,
        row_payload.row_values.contiguous(),
        outlier_cols,
        weight_t_bf16.contiguous(),
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def sparse_active_row_col16_value_payload_vec8_inplace_vstore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    outlier_cols_i16: torch.Tensor,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or output.dtype != torch.bfloat16 or not output.is_cuda:
        raise ValueError("output must be a CUDA BF16 [M, N] tensor")
    if int(output.shape[1]) % 8 != 0:
        raise ValueError("output N must be divisible by 8")
    if weight_t_bf16.dim() != 2 or weight_t_bf16.dtype != torch.bfloat16:
        raise ValueError("weight_t_bf16 must be BF16 [K, N]")
    if weight_t_bf16.device != output.device:
        raise ValueError("weight_t_bf16 must be on the output device")
    if int(weight_t_bf16.shape[0]) != int(k) or int(weight_t_bf16.shape[1]) != int(output.shape[1]):
        raise ValueError("weight_t_bf16 shape must be [K, N]")
    if int(k) > 32767:
        raise ValueError("vec8 int16-column path requires K <= 32767")
    if outlier_cols_i16.device != output.device or outlier_cols_i16.dtype != torch.int16:
        raise ValueError("outlier_cols_i16 must be CUDA int16")
    if int(outlier_cols_i16.numel()) != int(row_payload.row_values.numel()):
        raise ValueError("outlier_cols_i16 must have one entry per selected value")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.sparse_active_row_col_value_payload_vec8_inplace_vstore(
        output,
        row_payload.row_values.contiguous(),
        outlier_cols_i16.contiguous(),
        weight_t_bf16.contiguous(),
        row_payload.row_offsets.contiguous(),
        active_rows,
        int(k),
    )


def choose_sparse_exact_poststore_variant(
    output: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    active_row_fraction_threshold: float = 0.5,
) -> str:
    m = int(output.shape[0])
    active_fraction = float(int(active_rows.numel())) / float(max(m, 1))
    can_use_col_vec8 = int(output.shape[1]) % 8 == 0 and int(k) <= 32767
    if can_use_col_vec8 and active_fraction <= float(active_row_fraction_threshold):
        return "col_vec8_vstore"
    return "vstore"


def sparse_active_row_value_payload_vec8_inplace_best_poststore(
    output: torch.Tensor,
    row_payload: RowIndexedPayload,
    weight_t_bf16: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    k: int,
    flat_indices: torch.Tensor | None = None,
    variant: str = "auto",
    active_row_fraction_threshold: float = 0.5,
    lineinfo: bool = False,
) -> torch.Tensor:
    selected_variant = str(variant)
    if selected_variant == "auto":
        selected_variant = choose_sparse_exact_poststore_variant(
            output,
            active_rows,
            k=k,
            active_row_fraction_threshold=active_row_fraction_threshold,
        )
    if selected_variant == "col_vec8_vstore":
        return sparse_active_row_col_value_payload_vec8_inplace_vstore(
            output,
            row_payload,
            weight_t_bf16,
            active_rows,
            k=k,
            lineinfo=lineinfo,
        )
    if selected_variant == "b_evict_last_vstore":
        return sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore(
            output,
            row_payload,
            weight_t_bf16,
            active_rows,
            k=k,
            flat_indices=flat_indices,
            lineinfo=lineinfo,
        )
    if selected_variant == "vstore":
        return sparse_active_row_value_payload_vec8_inplace_vstore(
            output,
            row_payload,
            weight_t_bf16,
            active_rows,
            k=k,
            flat_indices=flat_indices,
            lineinfo=lineinfo,
        )
    if selected_variant == "strict_vstore":
        return sparse_active_row_value_payload_vec8_inplace_strict_vstore(
            output,
            row_payload,
            weight_t_bf16,
            active_rows,
            k=k,
            flat_indices=flat_indices,
            lineinfo=lineinfo,
        )
    raise ValueError(
        "variant must be 'auto', 'col_vec8_vstore', 'b_evict_last_vstore', "
        "'vstore', or 'strict_vstore'"
    )


def merge_single_entry_delta_active_rows(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    entry_indices: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    entry_indices = entry_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if int(active_rows.numel()) != int(entry_indices.numel()):
        raise ValueError("active_rows and entry_indices length mismatch")
    return ext.merge_single_entry_delta_active_rows(
        output,
        delta_entries.contiguous(),
        active_rows,
        entry_indices,
    )


def merge_double_entry_delta_active_rows(
    output: torch.Tensor,
    delta_entries: torch.Tensor,
    active_rows: torch.Tensor,
    entry0_indices: torch.Tensor,
    entry1_indices: torch.Tensor,
    *,
    lineinfo: bool = False,
) -> torch.Tensor:
    ext = _load_tma_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    if output.dim() != 2 or delta_entries.dim() != 2:
        raise ValueError("output and delta_entries must be 2D")
    if output.dtype != torch.bfloat16 or delta_entries.dtype != torch.bfloat16:
        raise ValueError("output and delta_entries must be BF16")
    if output.device != delta_entries.device:
        raise ValueError("output and delta_entries must be on the same device")
    if int(output.shape[1]) != int(delta_entries.shape[1]):
        raise ValueError("output and delta_entries must have the same N")
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    entry0_indices = entry0_indices.to(device=output.device, dtype=torch.int32).contiguous()
    entry1_indices = entry1_indices.to(device=output.device, dtype=torch.int32).contiguous()
    if (
        int(active_rows.numel()) != int(entry0_indices.numel())
        or int(active_rows.numel()) != int(entry1_indices.numel())
    ):
        raise ValueError("active_rows and entry indices length mismatch")
    return ext.merge_double_entry_delta_active_rows(
        output,
        delta_entries.contiguous(),
        active_rows,
        entry0_indices,
        entry1_indices,
    )


def preallocated_nvfp4_dense16_sparse_tail_add_active(
    output: torch.Tensor,
    qx,
    qw,
    payload: RowIndexedPayload,
    b_comp: torch.Tensor,
    *,
    dense_ntiles: int = 16,
    active_row_offsets: torch.Tensor | None = None,
    active_rows: torch.Tensor | None = None,
    lineinfo: bool = False,
) -> torch.Tensor:
    if payload.selected_count == 0:
        return preallocated_nvfp4_dense(output, qx, qw, lineinfo=lineinfo)
    if dense_ntiles not in (15, 16):
        raise ValueError("dense_ntiles must be 15 or 16")
    if b_comp.dim() != 2 or b_comp.dtype != torch.bfloat16:
        raise ValueError("b_comp must be BF16 [K, N]")
    ext = _load_extension(lineinfo)
    if ext is None:
        raise RuntimeError("CUDA extension is unavailable")
    a_data, a_scale, b_data, b_scale, a_amax, b_amax, m, k, n = _nvfp4_args(qx, qw)
    if tuple(output.shape) != (m, n):
        raise ValueError("output shape must be [M, N]")
    if tuple(b_comp.shape) != (k, n):
        raise ValueError("b_comp shape must be [K, N]")
    if active_row_offsets is None or active_rows is None:
        active_row_offsets, active_rows = balanced_active_rows_by_block_from_offsets(
            payload.row_offsets, m, bm=256
        )
    active_row_offsets = active_row_offsets.to(device=output.device, dtype=torch.int32).contiguous()
    active_rows = active_rows.to(device=output.device, dtype=torch.int32).contiguous()
    return ext.preallocated_nvfp4_dense16_sparse_tail_add_active(
        output,
        a_data,
        a_scale,
        b_data,
        b_scale,
        a_amax,
        b_amax,
        payload.row_offsets.contiguous(),
        payload.row_ks.contiguous(),
        payload.row_values.contiguous(),
        active_row_offsets,
        active_rows,
        b_comp.contiguous(),
        m,
        k,
        n,
        payload.r,
        payload.kb,
        payload.c,
        dense_ntiles,
    )
