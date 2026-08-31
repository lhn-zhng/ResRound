# ELCR: Ephemeral Local-Covariance Rounding

ELCR is a FPROP-only weight-code correction for the custom
`A = A1 + A2` input-outlier recipe. It keeps the existing computation

```text
Q(A1) @ Q(W)^T + A2 @ W^T
```

and only changes a sparse subset of the packed E2M1 codes consumed by the
first term. The native NVFP4 global scale, 16-value microscales, payload
shape, and Transformer Engine GEMM are unchanged. Master weights and the
weight variants saved for DGRAD/WGRAD are never modified.

## Motivation

The matched 6k operand-oracle experiment identifies the FPROP weight as the
strongest individual BF16 operand:

| Recipe | Full validation loss | Fraction of TE-to-BF16 gap closed |
|---|---:|---:|
| TE NVFP4 | 3.530029 | 0% |
| FPROP input BF16 | 3.518931 | 32.2% |
| FPROP weight BF16 | 3.513549 | 47.8% |
| BF16 | 3.495570 | 100% |

This motivates improving the weight representation without undoing the
already successful input split.

## Local objective

Let `X = Q(A1)` be a strided sample of the already quantized dense-main
activation, and let `e = Q(W) - W` be one weight row's error. For an
input-channel group `b`,

```text
G_b = X_b^T X_b .
```

Each weight code may either remain at its native round-to-nearest E2M1
endpoint or move to the other adjacent endpoint. If that move changes the
dequantized value by `delta_j`, its exact block-objective change is

```text
Delta_j = 2 delta_j (G_b e_b)_j + delta_j^2 (G_b)_{jj}.
```

ELCR greedily takes a configured small number of negative-`Delta` moves per
row and group. For multiple moves, it updates `G e` by the exact recurrence

```text
G(e + delta_k unit_k) = Ge + delta_k G[:, k].
```

The production R2 configuration uses:

```text
group_size       = 128
rounds_per_group = 2
selection_tokens = 1024
offdiag_shrink   = 0.5
```

The off-diagonal shrink is a finite-sample stabilizer, not a claimed
contribution.

## Exact joint guard

Independent block decisions can interact through covariance between
different groups. ELCR therefore evaluates all proposed moves for a complete
output row jointly and accepts them only when

```text
||X(e + d)||_2^2 < ||Xe||_2^2
```

with an FP32 roundoff margin. This gives a strict sampled-error descent
guarantee even though selection uses block-local covariance.

The guard is evaluated in fixed 1024-row chunks. Compared with the original
256-row active-index implementation, the optimized version removes a
device-to-host synchronization and reduces real FC1 guard GEMM launches from
22 to 6. On the concurrent four-GPU 100M workload this reduced the candidate
median step time from 3.112 s to 2.972 s.

## Lifetime and distributed behavior

- A rounded payload is built from the first microbatch after a quantized
  weight generation changes.
- The packed payload is reused for the remaining microbatches in that
  generation.
- The state is rank-local and generation-local; there is no DP
  synchronization, calibration set, BF16 teacher, optimizer state, or
  trainable rounding parameter.
- Only matrices with `rows >= columns` are enabled by default. This covers
  QKV, attention output projection, and FC1 while skipping the weak FC2 case.

## Measured evidence before the final 6k run

Offline evaluation on 32 captured real FPROP linears:

| Configuration | Output-error improvement | Win rate |
|---|---:|---:|
| group64 / R1 / 2048 tokens | 1.54% | 32/32 |
| group128 / R2 / 2048 tokens | 4.06% | 32/32 |
| group128 / R2 / 1024 tokens / shrink 0.5 | 3.23% | 32/32 |
| group128 / R2 / 512 tokens | 1.28% | 24/32 |

The 1024-token point is the measured cost/accuracy knee. Its only
production-enabled modules (QKV/O/FC1) improve by approximately 4.03%.

Strict four-GPU 100-step results:

| Recipe | Train loss | Interval validation | Full validation |
|---|---:|---:|---:|
| input-only | 6.818800 | 6.817369 | 6.807122 |
| input + ELCR R1 | 6.814485 | 6.807443 | 6.799005 |
| input + ELCR R2/2048 | 6.808703 | 6.799510 | 6.790892 |
| input + ELCR R2/1024/shrink | 6.801651 | 6.798705 | 6.790239 |

The optimized R2/1024 implementation has a measured median step overhead of
4.83% over input-only under the same concurrent workload. The final matched
6k comparison is intentionally left as a required acceptance test rather
than inferred from this short run.

## Reproduction

```bash
cd /path/to/ResRound
VARIANT=local_cov_r2_s1024 \
TRAIN_ITERS=6000 \
WANDB_MODE=online \
bash examples/fp4_paper_repro/train_quartet_c4_100m_local_cov_compare.sh
```

The relevant public parameters are:

```text
--fp4-outlier-enable-weight-rounding
--fp4-outlier-weight-rounding-group-size
--fp4-outlier-weight-rounding-rounds-per-group
--fp4-outlier-weight-rounding-selection-tokens
--fp4-outlier-weight-rounding-offdiag-shrink
--[no-]fp4-outlier-weight-rounding-expansion-only
--[no-]fp4-outlier-weight-rounding-reuse-generation-payload
```

## Novelty boundary

The defensible claim is deliberately narrow:

> A training-time, generation-local, pass-isolated native-NVFP4 code
> correction that uses the current already-quantized `A1` covariance, keeps
> scales and the TE GEMM fixed, preserves master/backward weights, and
> composes with a sparse `A1 + A2` FPROP input decomposition without teacher,
> calibration, trainable state, or distributed synchronization.

Do **not** claim that activation-aware rounding, adjacent-endpoint rounding,
or covariance-weighted reconstruction is independently new. The closest
work differs as follows:

- [GPTQ](https://arxiv.org/abs/2210.17323) and
  [AWP](https://arxiv.org/abs/2506.10205) are offline PTQ/reconstruction
  methods that persistently quantize a deployed model.
- [FAAR](https://arxiv.org/abs/2603.22370) learns NVFP4 rounding through a
  fine-tuning/alignment procedure.
- [TTQ](https://arxiv.org/abs/2603.19296) performs online calibration for
  test-time inference quantization.
- [Quartet II](https://arxiv.org/abs/2601.22813) targets unbiased gradient
  estimation across quantized training passes.
- [Four Over Six](https://arxiv.org/abs/2512.02010) changes adaptive block
  scaling rather than making an ephemeral FPROP-only code payload at fixed
  scales.
