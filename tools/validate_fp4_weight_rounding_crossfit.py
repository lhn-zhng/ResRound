#!/usr/bin/env python3
"""Held-out validation for FP4 weight-rounding cross-fit audit.

The benchmark uses real GPT-2 345M checkpoint weights, real embedding-derived
layer-0 QKV inputs, and token IDs from the indexed OpenWebText dataset. It
compares a fixed-size in-sample R2 proposal with a proposal/audit payload on an
independent later token batch. It does not mutate the checkpoint or any running
training process.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.extensions.fp4_outlier.config import configure_from_transformer_config
from megatron.core.extensions.fp4_outlier.factory import nvfp4_outlier_quantizer_factory
from megatron.core.extensions.fp4_outlier.storage import make_primary_nvfp4_storage
from megatron.core.extensions.fp4_outlier.weight_rounding import (
    adjacent_codes_and_values,
    enforce_crossfit_audit,
    sample_weight_rounding_inputs,
    select_local_covariance_codes_from_error,
    split_weight_rounding_crossfit_inputs,
    unpack_fp4_codes,
)


@dataclass
class TrialResult:
    baseline_energy: float
    r2_energy: float
    crossfit_energy: float
    r2_regressed_rows: int
    crossfit_regressed_rows: int
    rows: int
    r2_updates: int
    proposed_updates: int
    retained_updates: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT
            / "checkpoints/fp4_paper_repro/gpt2_345m_openwebtext_4gpu/"
            "weight_only/iter_0017167"
        ),
    )
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "datasets/openwebtext_gpt2/bpe_openwebtext"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--proposal-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--proposal-tokens", type=int, default=1024)
    parser.add_argument("--selection-tokens", type=int, default=1536)
    parser.add_argument("--audit-tokens", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rounds-per-group", type=int, default=2)
    parser.add_argument("--offdiag-shrink", type=float, default=0.5)
    parser.add_argument("--max-regression-fraction", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=0)
    return parser.parse_args()


def configure_recipe() -> None:
    configure_from_transformer_config(
        SimpleNamespace(
            fp4_outlier_ratio=0.0,
            fp4_outlier_selection_method="normal_threshold",
            fp4_outlier_enable_fprop=True,
            fp4_outlier_enable_fast_fprop=False,
            fp4_outlier_enable_dgrad=False,
            fp4_outlier_enable_wgrad=False,
            fp4_outlier_enable_nvfp4_a1_a2_all_gather=False,
            fp4_outlier_enable_weight_rounding=True,
            fp4_outlier_main_quantizer_rht=False,
            fp4_outlier_input_stochastic_rounding=False,
        )
    )


def load_checkpoint_tensors(checkpoint: str) -> dict[str, torch.Tensor]:
    state = {
        "embedding.word_embeddings.weight": torch.empty(
            50304, 1024, dtype=torch.bfloat16
        ),
        "embedding.position_embeddings.weight": torch.empty(
            1024, 1024, dtype=torch.bfloat16
        ),
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight": torch.empty(
            24, 1024, dtype=torch.bfloat16
        ),
        "decoder.layers.self_attention.linear_qkv.layer_norm_bias": torch.empty(
            24, 1024, dtype=torch.bfloat16
        ),
        "decoder.layers.self_attention.linear_qkv.weight": torch.empty(
            24, 3072, 1024, dtype=torch.bfloat16
        ),
    }
    dcp.load(state, checkpoint_id=checkpoint)
    return state


def load_token_ids(data_path: str, count: int) -> torch.Tensor:
    dataset = IndexedDataset(data_path, multimodal=False, mmap=True)
    pieces = []
    collected = 0
    document = 0
    while collected < count:
        piece = torch.from_numpy(dataset[document].copy()).to(torch.int64)
        pieces.append(piece)
        collected += int(piece.numel())
        document += 1
    return torch.cat(pieces)[:count]


def make_layer0_input(
    token_ids: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    word_embedding: torch.Tensor,
    position_embedding: torch.Tensor,
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
) -> torch.Tensor:
    expected = batch_size * sequence_length
    if token_ids.numel() != expected:
        raise ValueError(f"Expected {expected} token IDs, got {token_ids.numel()}.")
    token_ids = token_ids.reshape(batch_size, sequence_length).t().contiguous()
    positions = torch.arange(sequence_length, device=token_ids.device)
    positions = positions[:, None].expand(sequence_length, batch_size)
    hidden = word_embedding[token_ids] + position_embedding[positions]
    hidden = F.layer_norm(
        hidden.float(),
        (hidden.shape[-1],),
        layer_norm_weight.float(),
        layer_norm_bias.float(),
        1.0e-5,
    )
    return hidden.to(torch.bfloat16).reshape(-1, hidden.shape[-1])


def quantize_input(activation: torch.Tensor) -> torch.Tensor:
    qresult = nvfp4_outlier_quantizer_factory("linear_input")(activation)
    storage = make_primary_nvfp4_storage(
        qresult,
        use_rowwise=True,
        use_columnwise=False,
    )
    return storage.dequantize(dtype=torch.float32)


def row_energy(
    x: torch.Tensor,
    weight_error: torch.Tensor,
    *,
    token_chunk: int = 1024,
) -> torch.Tensor:
    result = torch.zeros(
        weight_error.shape[0],
        device=x.device,
        dtype=torch.float64,
    )
    weight_t = weight_error.float().t().contiguous()
    for start in range(0, x.shape[0], token_chunk):
        residual = x[start : start + token_chunk].float() @ weight_t
        result += residual.square().sum(dim=0, dtype=torch.float64)
    return result


def measurable_regressed_rows(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
) -> int:
    tolerance = 64.0 * torch.finfo(torch.float32).eps * torch.maximum(
        candidate.abs(), baseline.abs()
    ).clamp_min(1.0)
    return int((candidate > baseline + tolerance).sum().item())


@torch.no_grad()
def run_trial(
    train_input: torch.Tensor,
    test_input: torch.Tensor,
    *,
    error: torch.Tensor,
    delta: torch.Tensor,
    nearest_codes: torch.Tensor,
    adjacent_codes: torch.Tensor,
    args: argparse.Namespace,
) -> TrialResult:
    r2_sampled = sample_weight_rounding_inputs(
        train_input,
        args.proposal_tokens,
        stratified=True,
        stratified_batch_size=args.proposal_batch_size,
    )

    _, r2_selected = select_local_covariance_codes_from_error(
        r2_sampled,
        error,
        delta,
        nearest_codes,
        adjacent_codes,
        group_size=args.group_size,
        rounds_per_group=args.rounds_per_group,
        offdiag_shrink=args.offdiag_shrink,
    )

    sampled = sample_weight_rounding_inputs(
        train_input,
        args.selection_tokens,
        stratified=True,
        stratified_batch_size=args.proposal_batch_size,
    )
    proposal, audit = split_weight_rounding_crossfit_inputs(
        sampled,
        args.audit_tokens,
        stratified_batch_size=args.proposal_batch_size,
    )
    _, proposed_selected = select_local_covariance_codes_from_error(
        proposal,
        error,
        delta,
        nearest_codes,
        adjacent_codes,
        group_size=args.group_size,
        rounds_per_group=args.rounds_per_group,
        offdiag_shrink=args.offdiag_shrink,
    )
    crossfit_selected = enforce_crossfit_audit(
        audit,
        error,
        delta,
        proposed_selected,
        group_size=args.group_size,
        max_updates_per_group=args.rounds_per_group,
        max_regression_fraction=args.max_regression_fraction,
    )

    baseline_row_energy = row_energy(test_input, error)
    r2_row_energy = row_energy(
        test_input,
        error + torch.where(r2_selected, delta, torch.zeros_like(delta)),
    )
    crossfit_row_energy = row_energy(
        test_input,
        error + torch.where(crossfit_selected, delta, torch.zeros_like(delta)),
    )
    return TrialResult(
        baseline_energy=float(baseline_row_energy.sum().item()),
        r2_energy=float(r2_row_energy.sum().item()),
        crossfit_energy=float(crossfit_row_energy.sum().item()),
        r2_regressed_rows=measurable_regressed_rows(
            r2_row_energy, baseline_row_energy
        ),
        crossfit_regressed_rows=measurable_regressed_rows(
            crossfit_row_energy, baseline_row_energy
        ),
        rows=int(error.shape[0]),
        r2_updates=int(r2_selected.sum().item()),
        proposed_updates=int(proposed_selected.sum().item()),
        retained_updates=int(crossfit_selected.sum().item()),
    )


def relative_percent(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def main() -> None:
    args = parse_args()
    if args.selection_tokens != args.proposal_tokens + args.audit_tokens:
        raise ValueError(
            "selection_tokens must equal proposal_tokens + audit_tokens: "
            f"{args.selection_tokens} != "
            f"{args.proposal_tokens} + {args.audit_tokens}."
        )
    configure_recipe()
    device = torch.device(args.device)
    state = load_checkpoint_tensors(args.checkpoint)

    word_embedding = state["embedding.word_embeddings.weight"].to(device)
    position_embedding = state["embedding.position_embeddings.weight"].to(device)
    layer_norm_weight = state[
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight"
    ][args.layer].to(device)
    layer_norm_bias = state[
        "decoder.layers.self_attention.linear_qkv.layer_norm_bias"
    ][args.layer].to(device)
    weight = state["decoder.layers.self_attention.linear_qkv.weight"][args.layer].to(
        device
    )
    del state

    weight_qresult = nvfp4_outlier_quantizer_factory("linear_weight")(weight)
    weight_storage = make_primary_nvfp4_storage(
        weight_qresult,
        use_rowwise=True,
        use_columnwise=False,
    )
    nearest = weight_storage.dequantize(dtype=torch.float32)
    nearest_codes = unpack_fp4_codes(weight_qresult.data)
    adjacent_codes, adjacent = adjacent_codes_and_values(
        weight.float(), nearest, nearest_codes
    )
    error = nearest - weight.float()
    delta = adjacent - nearest

    tokens_per_trial = (
        args.proposal_batch_size + args.test_batch_size
    ) * args.sequence_length
    token_ids = load_token_ids(args.data_path, args.trials * tokens_per_trial)

    results = []
    for trial in range(args.trials):
        trial_tokens = token_ids[
            trial * tokens_per_trial : (trial + 1) * tokens_per_trial
        ].to(device)
        proposal_count = args.proposal_batch_size * args.sequence_length
        train_activation = make_layer0_input(
            trial_tokens[:proposal_count],
            batch_size=args.proposal_batch_size,
            sequence_length=args.sequence_length,
            word_embedding=word_embedding,
            position_embedding=position_embedding,
            layer_norm_weight=layer_norm_weight,
            layer_norm_bias=layer_norm_bias,
        )
        test_activation = make_layer0_input(
            trial_tokens[proposal_count:],
            batch_size=args.test_batch_size,
            sequence_length=args.sequence_length,
            word_embedding=word_embedding,
            position_embedding=position_embedding,
            layer_norm_weight=layer_norm_weight,
            layer_norm_bias=layer_norm_bias,
        )
        train_input = quantize_input(train_activation)
        test_input = quantize_input(test_activation)
        result = run_trial(
            train_input,
            test_input,
            error=error,
            delta=delta,
            nearest_codes=nearest_codes,
            adjacent_codes=adjacent_codes,
            args=args,
        )
        results.append(result)
        print(
            f"trial={trial} "
            f"r2_test_delta={relative_percent(result.r2_energy, result.baseline_energy):+.6f}% "
            f"crossfit_test_delta={relative_percent(result.crossfit_energy, result.baseline_energy):+.6f}% "
            f"regressed_rows={result.r2_regressed_rows}/{result.rows}->"
            f"{result.crossfit_regressed_rows}/{result.rows} "
            f"updates={result.r2_updates} proposal={result.proposed_updates} "
            f"retained={result.retained_updates}"
        )
        del train_activation, test_activation, train_input, test_input

    baseline_total = sum(item.baseline_energy for item in results)
    r2_total = sum(item.r2_energy for item in results)
    crossfit_total = sum(item.crossfit_energy for item in results)
    print(
        "summary "
        f"trials={len(results)} "
        f"r2_test_delta={relative_percent(r2_total, baseline_total):+.6f}% "
        f"crossfit_test_delta={relative_percent(crossfit_total, baseline_total):+.6f}% "
        f"regressed_rows={sum(x.r2_regressed_rows for x in results)}->"
        f"{sum(x.crossfit_regressed_rows for x in results)} "
        f"retention={sum(x.retained_updates for x in results) / max(1, sum(x.proposed_updates for x in results)):.6f}"
    )


if __name__ == "__main__":
    main()
