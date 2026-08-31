#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from megatron.core.datasets import indexed_dataset  # noqa: E402


def parse_int(value: str) -> int:
    return int(value.replace("_", ""))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def remove_matching(paths):
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def shard_prefix(shards_dir: Path, prefix_name: str, shard_id: int) -> Path:
    return shards_dir / f"{prefix_name}_shard_{shard_id:05d}"


def cleanup_inprogress(shards_dir: Path, prefix_name: str) -> None:
    remove_matching(shards_dir.glob(f"{prefix_name}_shard_*.inprogress.bin"))
    remove_matching(shards_dir.glob(f"{prefix_name}_shard_*.inprogress.idx"))


def write_data_path_files(output_root: Path, prefixes) -> None:
    shards_dir = output_root / "shards"
    lines = [str(prefix) for prefix in prefixes]
    (shards_dir / "data_path_prefixes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (shards_dir / "data_path_args.txt").write_text(" ".join(lines) + "\n", encoding="utf-8")


def load_streaming_dataset(args, skip_docs: int):
    os.environ.setdefault("HF_HOME", str(args.hf_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.hf_cache_dir / "datasets"))
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    if skip_docs:
        dataset = dataset.skip(skip_docs)
    return dataset


class ShardWriter:
    def __init__(self, args, dtype):
        self.args = args
        self.dtype = dtype
        self.builder = None
        self.shard_id = args.start_shard_id
        self.shard_tokens = 0
        self.shard_docs = 0
        self.current_tmp_prefix = None
        self.current_final_prefix = None

    def open(self):
        final_prefix = shard_prefix(self.args.shards_dir, self.args.prefix_name, self.shard_id)
        tmp_prefix = final_prefix.with_name(final_prefix.name + ".inprogress")
        self.current_tmp_prefix = tmp_prefix
        self.current_final_prefix = final_prefix
        self.builder = indexed_dataset.IndexedDatasetBuilder(
            indexed_dataset.get_bin_path(str(tmp_prefix)),
            dtype=self.dtype,
        )
        self.shard_tokens = 0
        self.shard_docs = 0

    def add_document(self, ids):
        if self.builder is None:
            self.open()
        self.builder.add_document(ids, [len(ids)])
        self.shard_tokens += len(ids)
        self.shard_docs += 1

    def finalize(self):
        if self.builder is None:
            return None
        tmp_prefix = self.current_tmp_prefix
        final_prefix = self.current_final_prefix
        self.builder.finalize(indexed_dataset.get_idx_path(str(tmp_prefix)))
        os.replace(indexed_dataset.get_bin_path(str(tmp_prefix)), indexed_dataset.get_bin_path(str(final_prefix)))
        os.replace(indexed_dataset.get_idx_path(str(tmp_prefix)), indexed_dataset.get_idx_path(str(final_prefix)))
        result = {
            "prefix": str(final_prefix),
            "tokens": self.shard_tokens,
            "docs": self.shard_docs,
            "shard_id": self.shard_id,
        }
        self.builder = None
        self.current_tmp_prefix = None
        self.current_final_prefix = None
        self.shard_id += 1
        self.shard_tokens = 0
        self.shard_docs = 0
        return result

    def close_incomplete(self):
        if self.builder is not None:
            self.builder.data_file.close()
            self.builder = None


def get_args():
    parser = argparse.ArgumentParser(description="Stream FineWeb-Edu into Megatron indexed shards.")
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "datasets/fineweb_edu_llama3_38b"
    )
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-100BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--tokenizer-model", default=str(REPO_ROOT / "tokenizers/Llama-3.1-8B-Instruct")
    )
    parser.add_argument("--target-tokens", type=parse_int, default=38_000_000_000)
    parser.add_argument("--shard-tokens", type=parse_int, default=500_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--prefix-name", default="fineweb_edu_sample100BT_llama3_text_document")
    parser.add_argument("--append-eod", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--log-interval-docs", type=int, default=10_000)
    parser.add_argument("--log-interval-seconds", type=float, default=60.0)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.shards_dir = args.output_root / "shards"
    args.logs_dir = args.output_root / "logs"
    args.hf_cache_dir = (args.hf_cache_dir or (args.output_root / "hf_cache")).resolve()
    args.manifest_path = args.output_root / "manifest.json"
    args.shards_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.hf_cache_dir.mkdir(parents=True, exist_ok=True)
    return args


def initialize_state(args):
    if args.overwrite:
        remove_matching(args.shards_dir.glob(f"{args.prefix_name}_shard_*.bin"))
        remove_matching(args.shards_dir.glob(f"{args.prefix_name}_shard_*.idx"))
        cleanup_inprogress(args.shards_dir, args.prefix_name)
        remove_matching([args.manifest_path])

    if args.resume and args.manifest_path.exists():
        manifest = read_json(args.manifest_path)
        prefixes = [Path(p) for p in manifest.get("completed_prefixes", [])]
        args.start_shard_id = int(manifest.get("next_shard_id", len(prefixes)))
        return {
            "completed_docs": int(manifest.get("completed_docs", 0)),
            "completed_tokens": int(manifest.get("completed_tokens", 0)),
            "completed_prefixes": prefixes,
            "completed_shards": manifest.get("completed_shards", []),
        }

    existing = list(args.shards_dir.glob(f"{args.prefix_name}_shard_*.bin")) + list(
        args.shards_dir.glob(f"{args.prefix_name}_shard_*.idx")
    )
    if existing:
        raise RuntimeError(
            f"Existing shards found under {args.shards_dir}. Use --resume or --overwrite."
        )

    args.start_shard_id = 0
    cleanup_inprogress(args.shards_dir, args.prefix_name)
    return {
        "completed_docs": 0,
        "completed_tokens": 0,
        "completed_prefixes": [],
        "completed_shards": [],
    }


def save_manifest(args, state, status):
    payload = {
        "status": status,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "tokenizer_model": args.tokenizer_model,
        "target_tokens": args.target_tokens,
        "shard_tokens": args.shard_tokens,
        "completed_tokens": state["completed_tokens"],
        "completed_docs": state["completed_docs"],
        "completed_prefixes": [str(p) for p in state["completed_prefixes"]],
        "completed_shards": state["completed_shards"],
        "next_shard_id": args.start_shard_id,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(args.manifest_path, payload)
    write_data_path_files(args.output_root, state["completed_prefixes"])


def log_progress(start_time, last_log_time, state, writer):
    now = time.time()
    elapsed = max(now - start_time, 1e-6)
    recent_elapsed = max(now - last_log_time, 1e-6)
    total_tokens = state["completed_tokens"] + writer.shard_tokens
    total_docs = state["completed_docs"] + writer.shard_docs
    print(
        "progress "
        f"docs={total_docs} "
        f"tokens={total_tokens} "
        f"tokens_B={total_tokens / 1e9:.3f} "
        f"current_shard_tokens={writer.shard_tokens} "
        f"avg_tokens_per_s={total_tokens / elapsed:.1f} "
        f"elapsed_h={elapsed / 3600:.3f} "
        f"since_last_s={recent_elapsed:.1f}",
        flush=True,
    )
    return now


def main():
    args = get_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    state = initialize_state(args)
    cleanup_inprogress(args.shards_dir, args.prefix_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    eos_id = tokenizer.eos_token_id
    if args.append_eod and eos_id is None:
        raise RuntimeError("Tokenizer does not define eos_token_id; disable --append-eod or choose another tokenizer.")

    dtype = indexed_dataset.DType.optimal_dtype(len(tokenizer))
    if dtype is not np.int32:
        print(f"Using dtype={dtype}; tokenizer vocab={len(tokenizer)}", flush=True)
    else:
        print(f"Using dtype=np.int32; tokenizer vocab={len(tokenizer)}", flush=True)

    save_manifest(args, state, "running")
    dataset = load_streaming_dataset(args, state["completed_docs"])
    writer = ShardWriter(args, dtype)

    start_time = time.time()
    last_log_time = start_time
    local_seen_docs = 0
    batch = []

    def process_batch(texts):
        nonlocal last_log_time
        encodings = tokenizer(
            texts,
            add_special_tokens=args.add_special_tokens,
            truncation=False,
            padding=False,
        )["input_ids"]
        stop = False
        for ids in encodings:
            if not ids:
                continue
            if args.append_eod:
                ids = list(ids)
                ids.append(eos_id)
            writer.add_document(ids)
            state["completed_docs"] += 1
            local_total_tokens = state["completed_tokens"] + writer.shard_tokens

            if writer.shard_tokens >= args.shard_tokens or local_total_tokens >= args.target_tokens:
                shard = writer.finalize()
                if shard is not None:
                    state["completed_tokens"] += shard["tokens"]
                    state["completed_prefixes"].append(Path(shard["prefix"]))
                    state["completed_shards"].append(shard)
                    args.start_shard_id = writer.shard_id
                    save_manifest(args, state, "running")
                if state["completed_tokens"] >= args.target_tokens:
                    stop = True
                    break

            now = time.time()
            if (
                state["completed_docs"] % args.log_interval_docs == 0
                or now - last_log_time >= args.log_interval_seconds
            ):
                last_log_time = log_progress(start_time, last_log_time, state, writer)
        return stop

    try:
        for row in dataset:
            if args.max_docs is not None and local_seen_docs >= args.max_docs:
                break
            text = row.get(args.text_field)
            local_seen_docs += 1
            if not isinstance(text, str) or not text.strip():
                continue
            batch.append(text)
            if len(batch) >= args.batch_size:
                if process_batch(batch):
                    batch = []
                    break
                batch = []

        if batch:
            process_batch(batch)

        shard = writer.finalize()
        if shard is not None:
            state["completed_tokens"] += shard["tokens"]
            state["completed_prefixes"].append(Path(shard["prefix"]))
            state["completed_shards"].append(shard)
            args.start_shard_id = writer.shard_id

        status = "complete" if state["completed_tokens"] >= args.target_tokens else "stopped"
        save_manifest(args, state, status)
        log_progress(start_time, last_log_time, state, writer)
        print(f"status={status}", flush=True)
        print(f"manifest={args.manifest_path}", flush=True)
        print(f"data_path_args={args.shards_dir / 'data_path_args.txt'}", flush=True)
    except BaseException:
        writer.close_incomplete()
        cleanup_inprogress(args.shards_dir, args.prefix_name)
        save_manifest(args, state, "interrupted")
        raise


if __name__ == "__main__":
    main()
