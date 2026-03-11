# Qwen3.5-0.8B CUDA Practice

`Qwen3.5-0.8B` 텍스트 경로를 대상으로 한 CUDA 연습용 inference 프로젝트입니다.

- 목적: `Qwen3.5` 구조 학습과 CUDA 커널 제작 연습
- 범위: `model.language_model.*`만 지원
- 비지원: `vision`, `video`, `audio`, `mtp`, cache 기반 generation
- 실행 전략: CPU reference forward를 먼저 만들고 `*_gpu()`를 순차적으로 교체

## 구조

- `src/main.cpp`: C++ 진입점
- `src/model.cu`: Qwen 전용 파라미터 로드와 전체 forward
- `src/layer.cu`: CPU reference 연산과 GPU TODO 함수
- `src/config.cpp`: `text_config` 파싱
- `src/safetensors_loader.cpp`: `model.safetensors.index.json` 기반 가중치 로더
- `src/generation.cpp`: greedy generation 루프
- `scripts/run_text_generation.py`: 로컬 tokenizer/chat template wrapper

## 현재 구현 범위

- `layer_types` 기반으로 `linear_attention`와 `full_attention`를 분기합니다.
- `full_attention`는 Q/K RMSNorm, partial RoPE, GQA, sigmoid gate를 반영합니다.
- `linear_attention`는 gated DeltaNet 계열 CPU reference를 구현합니다.
- 최종 logits는 tied embedding으로 계산합니다.

## CUDA 연습 포인트

- `EmbeddingLookup_gpu`
- `QwenRMSNorm_gpu`
- `QwenRMSNormGated_gpu`
- `Linear_gpu`
- `SplitHeads_gpu`
- `ApplyPartialMRoPE_gpu`
- `AttentionScoresGrouped_gpu`
- `ScaleMaskSoftmax_gpu`
- `AttentionContextGrouped_gpu`
- `DepthwiseConv1dCausal_gpu`
- `PrepareLinearDecay_gpu`
- `DeltaStateScan_gpu`
- `MergeHeads_gpu`
- `ResidualAdd_gpu`
- `SiLU_gpu`
- `Sigmoid_gpu`
- `ElementwiseMul_gpu`
- `LMHead_gpu`

## 준비

모델 디렉터리에는 최소한 아래 파일이 있어야 합니다.

- `config.json`
- `model.safetensors.index.json`
- `model.safetensors-00001-of-00001.safetensors`
- tokenizer 관련 파일

## 빌드

```bash
make
```

## 실행

일반적으로는 Python wrapper를 사용합니다.

```bash
MODEL_DIR=/path/to/Qwen3.5-0.8B \
INPUT_TXT=./data/requests.txt \
OUTPUT_TXT=./data/responses.txt \
make run
```

또는:

```bash
python3 ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --input ./data/requests.txt \
  --output ./data/responses.txt
```

단건 요청:

```bash
python3 ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Qwen3.5의 full attention 구조를 설명해 주세요."
```

thinking template를 켜려면:

```bash
python3 ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Explain gated DeltaNet briefly." \
  --enable-thinking
```

forward-only:

```bash
./main -m /path/to/Qwen3.5-0.8B \
  --token-input ./data/prompts.bin \
  --logits-output ./data/logits.bin
```

## 주의

- 이 프로젝트는 `Qwen3.5-0.8B` 전용입니다.
- 빌드/실행 검증은 Ubuntu 환경을 기준으로 후속 점검이 필요합니다.
- text-only 경로만 구현하므로 multimodal 입력은 지원하지 않습니다.
