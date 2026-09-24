"""Fresh-process gray-box probability extraction for frozen LLMPrint prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def first_token_id(tokenizer, word: str) -> tuple[int, list[int]]:
    ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise ValueError(f"target word produced no token: {word!r}")
    return int(ids[0]), [int(value) for value in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fingerprint-dir", type=Path, required=True)
    parser.add_argument("--fingerprint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probability sequence: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal LLMPrint evaluation")

    manifest = json.loads(args.fingerprint_manifest.read_text(encoding="utf-8"))
    if manifest.get("validated_record_count") != 200 or manifest.get("integrity_status") != "PASS":
        raise RuntimeError("fingerprint manifest has not passed 200/200 integrity")
    expected = manifest["artifacts"]
    if len(expected) != 200:
        raise RuntimeError("fingerprint manifest must contain exactly 200 artifacts")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    load_seconds = time.perf_counter() - load_started
    records = []
    started = time.perf_counter()
    for index, entry in enumerate(expected):
        pair_id = f"llmprint-pair-{index:03d}"
        if entry["pair_id"] != pair_id:
            raise RuntimeError(f"manifest order mismatch at {index}")
        artifact = args.fingerprint_dir / f"{pair_id}.json"
        if sha256(artifact) != entry["sha256"]:
            raise RuntimeError(f"immutable fingerprint SHA256 mismatch: {pair_id}")
        fingerprint = json.loads(artifact.read_text(encoding="utf-8"))
        positive_id, positive_tokens = first_token_id(tokenizer, fingerprint["w_plus"])
        negative_id, negative_tokens = first_token_id(tokenizer, fingerprint["w_minus"])
        encoded = tokenizer(
            fingerprint["complete_optimized_prompt"],
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"].to("cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=encoded).logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
        positive_logit = float(logits[positive_id]) if positive_id < logits.numel() else float("-inf")
        negative_logit = float(logits[negative_id]) if negative_id < logits.numel() else float("-inf")
        positive_probability = float(probabilities[positive_id]) if positive_id < probabilities.numel() else 0.0
        negative_probability = float(probabilities[negative_id]) if negative_id < probabilities.numel() else 0.0
        records.append({
            "fingerprint_index": index,
            "pair_id": pair_id,
            "w_plus": fingerprint["w_plus"],
            "w_minus": fingerprint["w_minus"],
            "w_plus_token_id": positive_id,
            "w_minus_token_id": negative_id,
            "w_plus_all_token_ids": positive_tokens,
            "w_minus_all_token_ids": negative_tokens,
            "w_plus_logit": positive_logit,
            "w_minus_logit": negative_logit,
            "w_plus_probability": positive_probability,
            "w_minus_probability": negative_probability,
            "paper_bit": int(positive_logit >= negative_logit),
            "release_filtered_eligibility": min(positive_probability, negative_probability) > 1e-3,
            "release_bit": int(positive_logit > negative_logit),
            "margin": positive_logit - negative_logit,
        })
        del encoded, logits, probabilities

    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "wmkd.llmprint-probability-sequence.v1",
        "status": "COMPLETED",
        "model_id": args.model_id,
        "revision": args.revision,
        "local_path": str(args.model),
        "fresh_process_pid": os.getpid(),
        "fresh_reload": True,
        "trust_remote_code": False,
        "dtype": str(dtype),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "vocab_size": config.get("vocab_size"),
        "config_sha256": sha256(config_path),
        "fingerprint_manifest_sha256": sha256(args.fingerprint_manifest),
        "record_count": len(records),
        "model_load_seconds": load_seconds,
        "evaluation_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "error": None,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
