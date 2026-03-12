#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from run_text_generation import (
    encode_request,
    load_tokenizer,
    resolve_main_binary,
    write_token_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local C++ forward logits against Hugging Face logits."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="")
    parser.add_argument("--main-binary", type=Path, default=PROJECT_ROOT / "main")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def resolve_vocab_size(config) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)

    text_config = getattr(config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)

    raise ValueError("unable to resolve vocab size from Hugging Face config")


def run_cpp_forward(
    main_binary: Path, model_dir: Path, token_ids: list[int], pad_token_id: int, vocab_size: int
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="qwen_compare_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        token_path = tmpdir_path / "prompt_tokens.bin"
        logits_path = tmpdir_path / "logits.bin"
        write_token_batch(token_path, [token_ids], pad_token_id)

        cmd = [
            str(resolve_main_binary(main_binary)),
            "-m",
            str(model_dir),
            "--token-input",
            str(token_path),
            "--logits-output",
            str(logits_path),
        ]
        subprocess.run(cmd, check=True)

        logits = np.fromfile(logits_path, dtype=np.float32)
        expected = len(token_ids) * vocab_size
        if logits.size != expected:
            raise ValueError(
                f"unexpected logits size: got {logits.size}, expected {expected}"
            )
        return logits.reshape(1, len(token_ids), vocab_size)


def run_hf_forward(model_dir: Path, token_ids: list[int]) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required. Install them in the project venv."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float().cpu().numpy()
    return logits


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def print_topk(title: str, tokenizer, logits: np.ndarray, k: int) -> None:
    values = logits[-1]
    top_ids = np.argsort(values)[-k:][::-1]
    print(title)
    for rank, token_id in enumerate(top_ids, start=1):
        print(
            f"  {rank}. id={int(token_id):6d} logit={values[token_id]:10.6f} "
            f"text={token_text(tokenizer, int(token_id))!r}"
        )


def main() -> int:
    args = build_parser().parse_args()
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install it in the project venv."
        ) from exc

    tokenizer = load_tokenizer(args.model_dir)
    config = AutoConfig.from_pretrained(str(args.model_dir), local_files_only=True)
    vocab_size = resolve_vocab_size(config)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    token_ids = encode_request(
        tokenizer, args.prompt, args.system, args.enable_thinking
    )
    if not token_ids:
        raise ValueError("tokenized prompt is empty")

    cpp_logits = run_cpp_forward(
        args.main_binary, args.model_dir, token_ids, pad_token_id, vocab_size
    )
    hf_logits = run_hf_forward(args.model_dir, token_ids)

    if cpp_logits.shape != hf_logits.shape:
        raise ValueError(
            f"logits shape mismatch: cpp={cpp_logits.shape}, hf={hf_logits.shape}"
        )

    diff = cpp_logits - hf_logits
    last_cpp = cpp_logits[0, -1]
    last_hf = hf_logits[0, -1]
    cpp_next = int(np.argmax(last_cpp))
    hf_next = int(np.argmax(last_hf))

    print(f"Prompt: {args.prompt}")
    print(f"Token count: {len(token_ids)}")
    print(f"Logits shape: {cpp_logits.shape}")
    print(f"Overall max abs diff: {np.max(np.abs(diff)):.6f}")
    print(f"Overall mean abs diff: {np.mean(np.abs(diff)):.6f}")
    print(f"Overall RMSE: {np.sqrt(np.mean(diff * diff)):.6f}")
    print(f"Last-token max abs diff: {np.max(np.abs(last_cpp - last_hf)):.6f}")
    print(f"Last-token mean abs diff: {np.mean(np.abs(last_cpp - last_hf)):.6f}")
    print(
        "Next-token argmax: "
        f"cpp={cpp_next}({token_text(tokenizer, cpp_next)!r}) "
        f"hf={hf_next}({token_text(tokenizer, hf_next)!r})"
    )
    print_topk("C++ top-k:", tokenizer, cpp_logits[0], args.top_k)
    print_topk("HF top-k:", tokenizer, hf_logits[0], args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
