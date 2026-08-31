# GPT-2 345M OpenWebText Notes

## Model

Use the GPT-2 Medium scale as the "around 300M" model:

- layers: 24
- hidden size: 1024
- attention heads: 16
- FFN hidden size: 4096
- sequence length: 1024
- vocabulary: GPT-2 BPE, 50257 raw vocab, padded by Megatron to 50304 with `--make-vocab-size-divisible-by 128`

This is usually referred to as 345M or 355M depending on counting convention.

## Dataset Choice

The GPT-2 paper trained on WebText. Since WebText itself was not released,
OpenWebText is the standard open reproduction for GPT-2-style runs.

Example Megatron indexed prefix:

```bash
/path/to/openwebtext_prefix
```

Required files:

```bash
/path/to/openwebtext_prefix.bin
/path/to/openwebtext_prefix.idx
```

## Scripts

Training:

```bash
cd /path/to/ResRound
RUN_VARIANTS=bf16,te_nvfp4,ours0p1 \
bash examples/fp4_paper_repro/train_gpt2_345m_openwebtext.sh
```

Dry run:

```bash
cd /path/to/ResRound
DRY_RUN=1 bash examples/fp4_paper_repro/train_gpt2_345m_openwebtext.sh
```

Data check or preprocessing:

```bash
cd /path/to/ResRound
bash examples/fp4_paper_repro/prepare_gpt2_openwebtext_data.sh
```

To preprocess a raw JSONL:

```bash
RAW_JSONL=/path/to/openwebtext.jsonl \
OUTPUT_PREFIX=/path/to/out/bpe_openwebtext \
bash examples/fp4_paper_repro/prepare_gpt2_openwebtext_data.sh
```

## Quartet-II Nanochat 1.9B

Quartet-II reports Nanochat pretraining on FineWeb-Edu with 20 tokens per
parameter. For the 1.9B model this is 38B tokens.

The old local reproduction script matches this budget:

```bash
bash examples/fp4_paper_repro/train_quartet_nanochat_1p9b.sh
```

with:

```bash
TRAIN_TOKENS=38000000000
```

The fprop-input branch currently does not expose the Quartet-II CLI flags, so
the GPT-2 fprop-input launcher only includes BF16, TE NVFP4, and ours 0.1%.
