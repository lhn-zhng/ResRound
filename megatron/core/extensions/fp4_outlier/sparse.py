"""Sparse correction helpers for exact input outlier compensation."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch

from .split import has_values
from .tensor import OutlierAwareNVFP4TensorRef


def logical_2d_shape(qresult: OutlierAwareNVFP4TensorRef) -> tuple[int, int]:
    return qresult.logical_2d_shape()


def prepare_dense_for_sparse_mm(
    dense: torch.Tensor,
    *,
    transpose: bool = False,
) -> torch.Tensor:
    dense_fp32 = dense if dense.dtype == torch.float32 else dense.to(torch.float32)
    if transpose:
        dense_fp32 = dense_fp32.t()
    return dense_fp32.contiguous()


def get_cached_sparse_tensor(
    qresult: OutlierAwareNVFP4TensorRef,
    *,
    transpose_sparse: bool,
) -> torch.Tensor:
    cache_attr = "_cached_sparse_coo_t" if transpose_sparse else "_cached_sparse_coo"
    cached = getattr(qresult, cache_attr, None)
    if cached is not None:
        return cached

    rows, cols = qresult.ensure_outlier_coo()
    values = qresult.valid_outlier_values()
    if rows is None or cols is None or not has_values(values):
        raise ValueError("Sparse tensor requested without outlier payload.")

    indices = torch.stack((rows.long(), cols.long()), dim=0)
    sparse = torch.sparse_coo_tensor(
        indices,
        values.to(torch.float32),
        logical_2d_shape(qresult),
        device=values.device,
    ).coalesce()
    setattr(qresult, "_cached_sparse_coo", sparse)
    if transpose_sparse:
        sparse_t = sparse.transpose(0, 1).coalesce()
        setattr(qresult, "_cached_sparse_coo_t", sparse_t)
        return sparse_t
    return sparse


def get_cached_sparse_csr_tensor(qresult: OutlierAwareNVFP4TensorRef) -> torch.Tensor:
    cached = getattr(qresult, "_cached_sparse_csr", None)
    if cached is not None:
        return cached

    values = qresult.valid_outlier_values()
    flat = getattr(qresult, "outlier_flat_indices", None)
    row_offsets = getattr(qresult, "outlier_row_offsets", None)
    if flat is None or row_offsets is None or not has_values(values):
        raise ValueError("CSR sparse tensor requested without flat-index outlier payload.")

    rows, cols = logical_2d_shape(qresult)
    col_indices = torch.remainder(
        flat.narrow(0, 0, values.numel()).to(torch.int64),
        int(cols),
    ).contiguous()
    crow_indices = row_offsets.to(torch.int64).contiguous()
    values_fp32 = values.to(torch.float32).contiguous()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        sparse = torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values_fp32,
            size=(int(rows), int(cols)),
            device=values.device,
        )
    setattr(qresult, "_cached_sparse_csr", sparse)
    return sparse


def _sparse_correction_backend() -> str:
    backend = os.getenv("FP4_OUTLIER_SPARSE_CORRECTION_BACKEND", "auto").strip().lower()
    if backend in {"", "auto", "csr", "coo"}:
        return "auto" if backend == "" else backend
    return "auto"


def _can_use_csr_correction(
    qresult: OutlierAwareNVFP4TensorRef,
    dense: torch.Tensor,
    *,
    transpose_sparse: bool,
    transpose_result: bool,
) -> bool:
    if transpose_sparse or transpose_result:
        return False
    if getattr(qresult, "outlier_flat_indices", None) is None:
        return False
    if getattr(qresult, "outlier_row_offsets", None) is None:
        return False
    rows, cols = logical_2d_shape(qresult)
    return dense.ndim == 2 and int(dense.shape[0]) == int(cols) and int(rows) >= 0


def compute_sparse_correction(
    qresult: OutlierAwareNVFP4TensorRef,
    dense: Optional[torch.Tensor],
    *,
    transpose_sparse: bool = False,
    transpose_dense: bool = False,
    transpose_result: bool = False,
) -> Optional[torch.Tensor]:
    if dense is None or not has_values(qresult.valid_outlier_values()):
        return None

    dense_prepared = prepare_dense_for_sparse_mm(dense, transpose=transpose_dense)
    backend = _sparse_correction_backend()
    if backend in {"auto", "csr"} and _can_use_csr_correction(
        qresult,
        dense_prepared,
        transpose_sparse=transpose_sparse,
        transpose_result=transpose_result,
    ):
        return torch.sparse.mm(get_cached_sparse_csr_tensor(qresult), dense_prepared)
    if backend == "csr":
        raise ValueError("CSR sparse correction backend is not compatible with these operands.")

    correction = torch.sparse.mm(
        get_cached_sparse_tensor(qresult, transpose_sparse=transpose_sparse),
        dense_prepared,
    )
    return correction.t() if transpose_result else correction
