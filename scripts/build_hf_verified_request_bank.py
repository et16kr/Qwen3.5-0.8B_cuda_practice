#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT.parent / "images" / "Qwen3.5-0.8B"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hf_verified_request_bank.txt"

GLOBAL_BANNED_PATTERNS = (
    "i cannot",
    "i can't",
    "as an ai",
    "not enough information",
    "cannot browse",
    "can't browse",
)


@dataclass(frozen=True)
class Topic:
    phrase: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairTopic:
    left: str
    right: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptSpec:
    text: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bank of HF-verified Qwen prompts and save them as one prompt per line."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260312)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def explanation_topics() -> list[Topic]:
    return [
        Topic("Qwen3.5 full attention blocks", ("full attention", "query", "key", "value")),
        Topic("Qwen3.5 linear attention blocks", ("linear attention", "state", "token", "head")),
        Topic("gated DeltaNet style state updates", ("deltanet", "state", "update", "gate")),
        Topic("partial MRoPE in Qwen3.5", ("rope", "rotary", "partial", "position")),
        Topic("Q/K RMSNorm in full attention", ("rmsnorm", "query", "key", "normalization")),
        Topic("grouped-query attention in Qwen3.5", ("grouped", "query", "key-value", "head")),
        Topic("attention output gates", ("gate", "attention", "output", "sigmoid")),
        Topic("depthwise causal convolution in linear attention", ("depthwise", "causal", "convolution", "linear attention")),
        Topic("the tied embedding and LM head path", ("embedding", "lm head", "tied", "logit")),
        Topic("padding handling in batched decoding", ("padding", "batch", "sequence", "mask")),
        Topic("greedy next-token decoding", ("greedy", "token", "decode", "argmax")),
        Topic("activation buffers in a CUDA inference stack", ("activation", "buffer", "tensor", "memory")),
        Topic("Tensor shape changes through split-head and merge-head steps", ("shape", "head", "split", "merge")),
        Topic("sigmoid gating in Qwen3.5 blocks", ("sigmoid", "gate", "block", "hidden")),
        Topic("SiLU and gated MLP behavior", ("silu", "mlp", "gate", "activation")),
        Topic("causal masking inside full attention", ("causal", "mask", "attention", "future token")),
        Topic("token batch length handling", ("length", "token batch", "padding", "sequence")),
        Topic("CPU reference validation for CUDA kernels", ("cpu", "reference", "cuda", "validation")),
        Topic("BF16 checkpoint loading into float buffers", ("bf16", "float", "checkpoint", "loader")),
        Topic("numerical stability in softmax", ("softmax", "stability", "max", "overflow")),
    ]


def comparison_topics() -> list[PairTopic]:
    return [
        PairTopic("full attention", "linear attention", ("full attention", "linear attention", "tradeoff", "layer")),
        PairTopic("Q/K RMSNorm", "standard RMSNorm", ("rmsnorm", "query", "key", "difference")),
        PairTopic("grouped-query attention", "full multi-head attention", ("grouped", "multi-head", "memory", "head")),
        PairTopic("depthwise causal convolution", "dense temporal mixing", ("depthwise", "causal", "convolution", "mixing")),
        PairTopic("CPU validation", "throughput benchmarking", ("cpu", "validation", "throughput", "benchmark")),
        PairTopic("prompt length", "generation length", ("prompt", "generation", "length", "context")),
        PairTopic("bfloat16 checkpoints", "float32 accumulations", ("bfloat16", "float32", "precision", "accumulation")),
        PairTopic("split-head layout", "merged hidden-state layout", ("split", "merge", "layout", "tensor")),
        PairTopic("causal masks", "padding masks", ("causal", "padding", "mask", "attention")),
        PairTopic("gated DeltaNet state updates", "softmax attention context", ("deltanet", "attention", "state", "context")),
    ]


def debug_topics() -> list[Topic]:
    return [
        Topic("NaNs after ScaleMaskSoftmax", ("nan", "softmax", "scale", "mask")),
        Topic("wrong sequence lengths in padded batches", ("length", "padding", "batch", "sequence")),
        Topic("shape mismatches after SplitHeads", ("shape", "splitheads", "tensor", "head")),
        Topic("broken partial MRoPE indexing", ("rope", "position", "index", "rotary")),
        Topic("incorrect grouped attention scores", ("grouped", "attention", "score", "head")),
        Topic("mismatched CPU and GPU outputs", ("cpu", "gpu", "mismatch", "reference")),
        Topic("wrong lm head projection output", ("lm head", "logit", "projection", "vocabulary")),
        Topic("incorrect sigmoid gate behavior", ("sigmoid", "gate", "output", "attention")),
        Topic("bad depthwise convolution kernels", ("depthwise", "convolution", "kernel", "causal")),
        Topic("host-device copy mistakes in Tensor::to_gpu", ("host", "device", "copy", "tensor")),
    ]


def component_topics() -> list[Topic]:
    return [
        Topic("EmbeddingLookup_gpu", ("embedding", "lookup", "token", "output")),
        Topic("QwenRMSNorm_gpu", ("rmsnorm", "normalize", "hidden", "weight")),
        Topic("Linear_gpu", ("linear", "matrix", "row", "output")),
        Topic("ApplyPartialMRoPE_gpu", ("rope", "rotary", "query", "key")),
        Topic("AttentionScoresGrouped_gpu", ("attention", "score", "query", "key")),
        Topic("ScaleMaskSoftmax_gpu", ("softmax", "mask", "score", "probability")),
        Topic("AttentionContextGrouped_gpu", ("context", "probability", "value", "head")),
        Topic("DepthwiseConv1dCausal_gpu", ("depthwise", "causal", "convolution", "channel")),
        Topic("PrepareLinearDecay_gpu", ("linear decay", "state", "head", "time")),
        Topic("DeltaStateScan_gpu", ("delta", "state", "scan", "update")),
        Topic("MergeHeads_gpu", ("merge", "head", "hidden", "layout")),
        Topic("LMHead_gpu", ("lm head", "logit", "vocabulary", "embedding")),
    ]


def make_explanation_prompts(topic: Topic) -> list[PromptSpec]:
    templates = (
        "Explain {phrase} in plain language for someone implementing Qwen3.5 inference.",
        "Give a concise explanation of {phrase} in the Qwen3.5 text path.",
        "Describe the role of {phrase} during autoregressive decoding in Qwen3.5.",
        "Why does a CUDA engineer need to understand {phrase} when debugging Qwen3.5?",
        "Summarize {phrase} without equations, but keep the tensor flow concrete.",
        "Write a short note about {phrase} for a GPU inference code review.",
        "How would you explain {phrase} to a new engineer reading model.cu and layer.cu?",
        "What usually goes wrong around {phrase} in a hand-written CUDA inference stack?",
        "Give a debugging checklist for {phrase} in a text-only CUDA bring-up.",
        "Explain {phrase} with emphasis on tensor shapes and buffer ownership.",
        "Why does {phrase} matter when validating GPU output against a CPU reference?",
        "Describe {phrase} as if you were reviewing a student CUDA kernel patch.",
    )
    return [
        PromptSpec(
            text=template.format(phrase=topic.phrase),
            must_have=topic.must_have,
            banned=topic.banned,
        )
        for template in templates
    ]


def make_comparison_prompts(topic: PairTopic) -> list[PromptSpec]:
    templates = (
        "Compare {left} and {right} in the context of Qwen3.5 inference.",
        "What is the practical difference between {left} and {right} for CUDA bring-up?",
        "Explain when {left} matters more than {right} in a Qwen3.5 text-only stack.",
        "Contrast {left} with {right} for debugging correctness and performance.",
        "Describe the tradeoff between {left} and {right} for an engineer optimizing inference.",
        "How would you explain {left} versus {right} to someone validating a CPU reference path?",
    )
    return [
        PromptSpec(
            text=template.format(left=topic.left, right=topic.right),
            must_have=topic.must_have,
            banned=topic.banned,
        )
        for template in templates
    ]


def build_prompt_specs() -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    for topic in explanation_topics():
        prompts.extend(make_explanation_prompts(topic))
    for topic in comparison_topics():
        prompts.extend(make_comparison_prompts(topic))
    for topic in debug_topics():
        prompts.extend(make_explanation_prompts(topic))
    for topic in component_topics():
        prompts.extend(make_explanation_prompts(topic))
    deduped: list[PromptSpec] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.text not in seen:
            deduped.append(prompt)
            seen.add(prompt.text)
    return deduped


def contains_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return True
    matches = sum(1 for keyword in keywords if keyword in text)
    required = 1 if len(keywords) <= 2 else 2
    return matches >= required


def should_keep_response(prompt: PromptSpec, response: str) -> bool:
    normalized = normalize(response)
    if len(normalized.split()) < 4:
        return False
    if any(pattern in normalized for pattern in GLOBAL_BANNED_PATTERNS):
        return False
    if any(pattern in normalized for pattern in prompt.banned):
        return False
    return contains_keywords(normalized, prompt.must_have)


def choose_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def render_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            kwargs["enable_thinking"] = True
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
    return prompt


def generate_batch(
    model,
    tokenizer,
    prompts: list[PromptSpec],
    device,
    max_new_tokens: int,
    enable_thinking: bool,
) -> list[str]:
    import torch

    rendered = [render_prompt(tokenizer, prompt.text, enable_thinking) for prompt in prompts]
    encoded = tokenizer(
        rendered,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)


def load_model_stack(model_dir: Path, device):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required. Install them in the project venv."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def main() -> int:
    args = build_parser().parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    specs = build_prompt_specs()
    if args.count > len(specs):
        raise ValueError(
            f"requested {args.count} prompts, but only {len(specs)} candidates are available"
        )

    rng = random.Random(args.seed)
    rng.shuffle(specs)

    device = choose_device(args.device)
    tokenizer, model = load_model_stack(args.model_dir, device)

    kept: list[str] = []
    checked = 0
    while specs and len(kept) < args.count:
        batch = specs[: args.batch_size]
        specs = specs[args.batch_size :]
        responses = generate_batch(
            model,
            tokenizer,
            batch,
            device,
            args.max_new_tokens,
            args.enable_thinking,
        )
        checked += len(batch)
        for prompt, response in zip(batch, responses):
            if should_keep_response(prompt, response):
                kept.append(prompt.text)
                if len(kept) == args.count:
                    break

    if len(kept) < args.count:
        raise RuntimeError(
            f"only verified {len(kept)} prompts after checking {checked} candidates"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"saved {len(kept)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
