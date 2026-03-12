# Qwen3.5-0.8B CUDA Practice

CUDA 연습용 Qwen3.5-0.8B inference 프로젝트입니다.

- CPU 기준 forward를 먼저 제공하고, 학생이 CUDA kernel로 바꿔 가며 학습하는 형태를 목표로 합니다.
- Qwen3.5 text path 구조를 반영했습니다: `full_attention`, `linear_attention`, Q/K RMSNorm, partial MRoPE, grouped-query attention, gated MLP.
- 기본 실행 진입점은 `scripts/run_text_generation.py`입니다.
- `main`은 Python wrapper가 내부적으로 호출하는 tokenized generation 실행기입니다.
- 입력 텍스트 파일은 `1줄 = 1개 요청` 형식입니다.
- 필요하면 pretokenized input 파일을 넣어 `forward -> logits` 모드로도 실행할 수 있습니다.

## 모델 다운로드

- Hugging Face: `https://huggingface.co/Qwen/Qwen3.5-0.8B`

기본 사용 경로인 Python wrapper는 로컬 tokenizer의 chat template를 그대로 사용합니다.

## 준비 사항

- CUDA Toolkit과 `nvcc`
- C++17 지원 컴파일러
- Python 3
- Python 패키지
  - 기본 wrapper: `transformers` main, `tokenizers`, `jinja2`
  - HF 비교/요청 뱅크: `torch`, `numpy`
- 다운로드한 모델 디렉터리

예시 `.venv`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -U \
  "transformers @ git+https://github.com/huggingface/transformers.git@main" \
  tokenizers \
  jinja2 \
  torch \
  numpy
```

GPU 비교군으로는 CUDA-enabled `torch`가 필요합니다. Hugging Face GPU 출력을
기준값으로 비교하려면, CUDA가 활성화된 PyTorch 환경을 사용하면 됩니다.

모델 디렉터리에는 최소한 아래 파일들이 있어야 합니다.

- `config.json`
- `model.safetensors.index.json`
- `model.safetensors-00001-of-00001.safetensors`
- tokenizer 관련 파일들 (`tokenizer.json`, `tokenizer_config.json` 등)

## 프로젝트 구성

- `scripts/run_text_generation.py`: 기본 실행 진입점. text 입력 tokenize, `main` 호출, 결과 decode
- `scripts/compare_hf_logits.py`: 로컬 C++ forward logits와 Hugging Face logits 비교
- `scripts/build_hf_verified_request_bank.py`: HF 모델로 응답을 확인해 요청 뱅크 생성
- `scripts/sample_requests.py`: 요청 뱅크에서 원하는 개수만 랜덤 샘플링
- `src/main.cpp`: Python wrapper가 호출하는 tokenized generation/forward-only 엔트리 포인트
- `src/app_options.cpp`: CLI 파싱
- `src/generation.cpp`: batch generation 루프
- `src/model.cu`: Qwen 파라미터 로딩과 전체 forward
- `src/layer.cu`: CPU 기준 연산 구현과 GPU dispatch wrapper
- `src/config.cpp`: `config.json`과 `tokenizer_config.json` 로딩
- `data/`: 예시 요청 텍스트 파일

## 입력 파일 형식

`main`이 읽는 token batch 파일 형식:

- `int32 B`
- `int32 T`
- `int32 lengths[B]`
- `int32 token_ids[B*T]`

생성 결과 token batch 파일 형식:

- `int32 B`
- `int32 T_generated_max`
- `int32 lengths[B]`
- `int32 token_ids[B*T_generated_max]`

## 빌드

```bash
make
```

## 실행

일반적으로는 `main`을 직접 실행하지 않고 Python wrapper를 사용합니다.

### 1. 줄 단위 batch text generation

```bash
PYTHON_BIN=.venv/bin/python \
MODEL_DIR=/path/to/Qwen3.5-0.8B \
INPUT_TXT=./data/requests.txt \
OUTPUT_TXT=./data/responses.txt \
make run
```

또는:

```bash
.venv/bin/python ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --input ./data/requests.txt \
  --output ./data/responses.txt
```

출력 파일은 각 요청과 그에 대응하는 답변을 delimiter와 함께 저장합니다.

### 2. 한 번만 요청하기

```bash
.venv/bin/python ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Explain the role of Q/K RMSNorm in a Qwen3.5 full-attention block."
```

thinking template를 켜려면:

```bash
.venv/bin/python ./scripts/run_text_generation.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Explain gated DeltaNet briefly." \
  --enable-thinking
```

### 3. `main` 직접 실행

Python wrapper가 내부적으로 token file을 만들어 `main`을 호출합니다. 아래는 디버깅이나 실험용 직접 실행 예시입니다.

```bash
./main -m /path/to/Qwen3.5-0.8B \
  --token-input ./data/prompts.bin \
  --token-output ./data/generated_tokens.bin
```

### 4. forward-only / logits 저장

```bash
./main -m /path/to/Qwen3.5-0.8B \
  --token-input /path/to/your_tokens.bin \
  --logits-output ./data/logits.bin \
  -v
```

저장소에는 샘플 `bin` 파일을 포함하지 않습니다. 필요하면 Python wrapper가 만드는 token batch를 재사용하거나, 같은 포맷으로 직접 만들면 됩니다.

### 5. Hugging Face logits 비교

```bash
.venv/bin/python ./scripts/compare_hf_logits.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Explain Qwen3.5 linear attention briefly."
```

CUDA-enabled PyTorch 환경이 있으면 그 환경으로 비교하는 것이 좋습니다. 예:

```bash
.venv/bin/python ./scripts/compare_hf_logits.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Explain Qwen3.5 linear attention briefly."
```

### 6. Hugging Face 기준 요청 뱅크 생성

```bash
.venv/bin/python ./scripts/build_hf_verified_request_bank.py \
  --model-dir /path/to/Qwen3.5-0.8B \
  --count 256 \
  --output ./data/hf_verified_request_bank.txt
```

생성된 요청 뱅크에서 원하는 개수만 무작위로 추출하려면:

```bash
.venv/bin/python ./scripts/sample_requests.py \
  --input ./data/hf_verified_request_bank.txt \
  --output ./data/requests.txt \
  --count 64 \
  --seed 20260312
```

## 현재 상태

- `qwen_forward()`는 현재 `*_gpu()` dispatch 경로를 사용합니다.
- 다만 각 `*_gpu()` 함수는 아직 CPU 기준 결과를 먼저 내는 wrapper이며, 학생이 CUDA kernel로 교체하는 구조입니다.
- Python wrapper가 각 줄을 독립 요청으로 tokenize한 뒤, `main`은 tokenized batch에 대해 greedy decoding을 수행합니다.
- generation은 tokenizer metadata를 읽어 `pad/eos/bos`를 Qwen tokenizer와 일치시키도록 맞췄습니다.
- weight loader는 `F32`, `F16`, `BF16` safetensors를 읽어 float로 변환합니다.
- CPU reference는 `q_proj` query/gate 분리와 linear-attention depthwise causal conv 방향을 수정한 뒤, Hugging Face GPU 기준과 다시 맞췄습니다.

## 검증 메모

- 비교 기준: CUDA-enabled PyTorch 환경의 Hugging Face GPU 실행
- 모델: Hugging Face `Qwen/Qwen3.5-0.8B`
- forward logits 비교:
  - `overall_max_abs ~= 1.18e-4`
  - `overall_mean_abs ~= 7.44e-6`
  - last-token argmax와 top-k가 Hugging Face GPU와 일치
- greedy generation 비교:
  - 짧은 프롬프트에서 C++ CPU 출력과 Hugging Face GPU 출력이 같은 prefix를 생성하는 것을 확인
- `data/requests.txt`:
  - Hugging Face GPU에서 정상적인 영어 응답을 시작하는 요청만 골라 `1024`개로 구성

## 남은 연습 포인트

- `EmbeddingLookup_gpu`
- `QwenRMSNorm_gpu`
- `QwenRMSNormGated_gpu`
- `MaskPaddingHiddenStates_gpu`
- `Linear_gpu`
- `SplitHeads_gpu`
- `SplitLinearQKV_gpu`
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

## 주의

- 현재 구현은 학습용 CPU reference 중심입니다. 긴 컨텍스트에서는 느립니다.
- cache 기반 generation은 아직 구현하지 않았습니다. 따라서 generation은 매 스텝 전체 prompt를 다시 계산합니다.
- text-only 경로만 구현하므로 multimodal 입력은 지원하지 않습니다.
- `make run`은 `MODEL_DIR`과 적절한 Python 환경이 준비되어 있어야 합니다.
