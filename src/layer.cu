#include "layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"

namespace {

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least 2 dimensions");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

float softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return std::exp(x);
  }
  return static_cast<float>(std::log1p(std::exp(x)));
}

void build_inv_freq(const QwenConfig &config, std::vector<float> *inv_freq) {
  const size_t rotary_dim = config.rotary_dim();
  CHECK_ERROR((rotary_dim % 2) == 0, "rotary_dim must be even");
  inv_freq->assign(rotary_dim / 2, 0.0f);
  for (size_t idx = 0; idx < inv_freq->size(); ++idx) {
    const float exponent = (2.0f * static_cast<float>(idx)) /
                           static_cast<float>(rotary_dim);
    (*inv_freq)[idx] = 1.0f / std::pow(config.rope_theta, exponent);
  }
}

void apply_rope_tensor(Tensor *tensor, const QwenConfig &config) {
  const size_t B = tensor->shape[0];
  const size_t H = tensor->shape[1];
  const size_t T = tensor->shape[2];
  const size_t D = tensor->shape[3];
  const size_t rotary_dim = config.rotary_dim();
  CHECK_ERROR(rotary_dim <= D, "rotary_dim exceeds head dimension");
  const size_t half_rotary = rotary_dim / 2;

  std::vector<float> inv_freq;
  build_inv_freq(config, &inv_freq);

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t t = 0; t < T; ++t) {
        float *ptr = tensor->buf + ((b * H + h) * T + t) * D;
        for (size_t i = 0; i < half_rotary; ++i) {
          const float angle = static_cast<float>(t) * inv_freq[i];
          const float c = std::cos(angle);
          const float s = std::sin(angle);
          const float x0 = ptr[i];
          const float x1 = ptr[i + half_rotary];
          ptr[i] = x0 * c - x1 * s;
          ptr[i + half_rotary] = x1 * c + x0 * s;
        }
      }
    }
  }
}

}  // namespace

void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  CHECK_ERROR(embedding->ndim == 2, "Embedding tensor must be rank 2");
  CHECK_ERROR(output->shape[0] == tokens->B && output->shape[1] == tokens->T,
              "Embedding output shape mismatch");
  CHECK_ERROR(output->shape[2] == embedding->shape[1],
              "Embedding hidden size mismatch");

  const size_t hidden = embedding->shape[1];
  const size_t vocab_size = embedding->shape[0];

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < tokens->B; ++b) {
    for (size_t t = 0; t < tokens->T; ++t) {
      const int32_t token_id = tokens->buf[b * tokens->T + t];
      CHECK_ERROR(token_id >= 0 && token_id < static_cast<int32_t>(vocab_size),
                  "Token id %d out of range", token_id);
      const float *src = embedding->buf + static_cast<size_t>(token_id) * hidden;
      float *dst = output->buf + (b * tokens->T + t) * hidden;
      std::memcpy(dst, src, hidden * sizeof(float));
    }
  }
}

void EmbeddingLookup_gpu(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  EmbeddingLookup(tokens, embedding, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void QwenRMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);
  CHECK_ERROR(weight->ndim == 1 && weight->shape[0] == cols,
              "QwenRMSNorm weight shape mismatch");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "QwenRMSNorm output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * cols;
    float *out = output->buf + row * cols;

    float mean_sq = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      mean_sq += in[col] * in[col];
    }
    mean_sq /= static_cast<float>(cols);
    const float scale = 1.0f / std::sqrt(mean_sq + eps);

    for (size_t col = 0; col < cols; ++col) {
      out[col] = (in[col] * scale) * (1.0f + weight->buf[col]);
    }
  }
}

void QwenRMSNorm_gpu(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  QwenRMSNorm(input, weight, output, eps);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void QwenRMSNormGated(Tensor *input, Tensor *gate, Tensor *weight, Tensor *output,
                      float eps) {
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);
  CHECK_ERROR(gate->num_elem() == input->num_elem(),
              "QwenRMSNormGated gate shape mismatch");
  CHECK_ERROR(weight->ndim == 1 && weight->shape[0] == cols,
              "QwenRMSNormGated weight shape mismatch");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "QwenRMSNormGated output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * cols;
    const float *gate_row = gate->buf + row * cols;
    float *out = output->buf + row * cols;

    float mean_sq = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      mean_sq += in[col] * in[col];
    }
    mean_sq /= static_cast<float>(cols);
    const float scale = 1.0f / std::sqrt(mean_sq + eps);

    for (size_t col = 0; col < cols; ++col) {
      const float g = gate_row[col];
      const float silu_g = g / (1.0f + std::exp(-g));
      out[col] = (in[col] * scale) * weight->buf[col] * silu_g;
    }
  }
}

void QwenRMSNormGated_gpu(Tensor *input, Tensor *gate, Tensor *weight,
                          Tensor *output, float eps) {
  QwenRMSNormGated(input, gate, weight, output, eps);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void MaskPaddingHiddenStates(Tensor *input, Tensor *output, const TokenBatch *tokens) {
  CHECK_ERROR(input->ndim == 3, "MaskPaddingHiddenStates expects rank-3 input");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "MaskPaddingHiddenStates shape mismatch");
  CHECK_ERROR(tokens != nullptr && tokens->lengths != nullptr,
              "MaskPaddingHiddenStates requires token lengths");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t H = input->shape[2];

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      float *dst = output->buf + (b * T + t) * H;
      const float *src = input->buf + (b * T + t) * H;
      if (t >= static_cast<size_t>(tokens->lengths[b])) {
        std::memset(dst, 0, H * sizeof(float));
      } else if (dst != src) {
        std::memcpy(dst, src, H * sizeof(float));
      }
    }
  }
}

void MaskPaddingHiddenStates_gpu(Tensor *input, Tensor *output,
                                 const TokenBatch *tokens) {
  MaskPaddingHiddenStates(input, output, tokens);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void Linear(Tensor *input, Tensor *weight, Tensor *output) {
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  CHECK_ERROR(weight->ndim == 2 && weight->shape[1] == in_dim,
              "Linear weight shape mismatch");
  const size_t out_dim = weight->shape[0];
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * in_dim;
    float *out = output->buf + row * out_dim;
    for (size_t col = 0; col < out_dim; ++col) {
      const float *w = weight->buf + col * in_dim;
      float sum = 0.0f;
      for (size_t k = 0; k < in_dim; ++k) {
        sum += in[k] * w[k];
      }
      out[col] = sum;
    }
  }
}

void Linear_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  Linear(input, weight, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitTensorLastDim(Tensor *input, size_t left_size, Tensor *left, Tensor *right) {
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);
  CHECK_ERROR(left_size <= cols, "SplitTensorLastDim split exceeds last dimension");
  const size_t right_size = cols - left_size;
  CHECK_ERROR(left->num_elem() == rows * left_size,
              "SplitTensorLastDim left shape mismatch");
  CHECK_ERROR(right->num_elem() == rows * right_size,
              "SplitTensorLastDim right shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *src = input->buf + row * cols;
    float *dst_left = left->buf + row * left_size;
    float *dst_right = right->buf + row * right_size;
    std::memcpy(dst_left, src, left_size * sizeof(float));
    std::memcpy(dst_right, src + left_size, right_size * sizeof(float));
  }
}

void SplitTensorLastDim_gpu(Tensor *input, size_t left_size, Tensor *left,
                            Tensor *right) {
  SplitTensorLastDim(input, left_size, left, right);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitHeads(Tensor *input, Tensor *output, size_t num_heads, size_t head_dim) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 4, "SplitHeads rank mismatch");
  CHECK_ERROR(input->shape[0] == output->shape[0] &&
                  input->shape[1] == output->shape[2],
              "SplitHeads batch/sequence mismatch");
  CHECK_ERROR(input->shape[2] == num_heads * head_dim,
              "SplitHeads hidden mismatch");
  CHECK_ERROR(output->shape[1] == num_heads && output->shape[3] == head_dim,
              "SplitHeads output head shape mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < num_heads; ++h) {
        const size_t src_base = (b * T + t) * (num_heads * head_dim) + h * head_dim;
        const size_t dst_base = ((b * num_heads + h) * T + t) * head_dim;
        std::memcpy(output->buf + dst_base, input->buf + src_base,
                    head_dim * sizeof(float));
      }
    }
  }
}

void SplitHeads_gpu(Tensor *input, Tensor *output, size_t num_heads,
                    size_t head_dim) {
  SplitHeads(input, output, num_heads, head_dim);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitHeadScalars(Tensor *input, Tensor *output) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 3, "SplitHeadScalars rank mismatch");
  CHECK_ERROR(input->shape[0] == output->shape[0] &&
                  input->shape[1] == output->shape[2] &&
                  input->shape[2] == output->shape[1],
              "SplitHeadScalars shape mismatch");
  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t H = input->shape[2];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        output->buf[(b * H + h) * T + t] = input->buf[(b * T + t) * H + h];
      }
    }
  }
}

void SplitHeadScalars_gpu(Tensor *input, Tensor *output) {
  SplitHeadScalars(input, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitLinearQKV(Tensor *input, Tensor *query, Tensor *key, Tensor *value,
                    size_t num_k_heads, size_t head_k_dim, size_t num_v_heads,
                    size_t head_v_dim) {
  CHECK_ERROR(input->ndim == 3, "SplitLinearQKV expects rank-3 input");
  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t qk_hidden = num_k_heads * head_k_dim;
  const size_t v_hidden = num_v_heads * head_v_dim;
  CHECK_ERROR(input->shape[2] == qk_hidden * 2 + v_hidden,
              "SplitLinearQKV hidden mismatch");

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      const float *src = input->buf + (b * T + t) * input->shape[2];
      for (size_t h = 0; h < num_k_heads; ++h) {
        std::memcpy(query->buf + ((b * num_k_heads + h) * T + t) * head_k_dim,
                    src + h * head_k_dim, head_k_dim * sizeof(float));
        std::memcpy(key->buf + ((b * num_k_heads + h) * T + t) * head_k_dim,
                    src + qk_hidden + h * head_k_dim,
                    head_k_dim * sizeof(float));
      }
      for (size_t h = 0; h < num_v_heads; ++h) {
        std::memcpy(value->buf + ((b * num_v_heads + h) * T + t) * head_v_dim,
                    src + qk_hidden * 2 + h * head_v_dim,
                    head_v_dim * sizeof(float));
      }
    }
  }
}

void SplitLinearQKV_gpu(Tensor *input, Tensor *query, Tensor *key, Tensor *value,
                        size_t num_k_heads, size_t head_k_dim,
                        size_t num_v_heads, size_t head_v_dim) {
  SplitLinearQKV(input, query, key, value, num_k_heads, head_k_dim, num_v_heads,
                 head_v_dim);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ApplyPartialMRoPE(Tensor *q, Tensor *k, const QwenConfig &config) {
  CHECK_ERROR(q->ndim == 4 && k->ndim == 4, "ApplyPartialMRoPE expects rank-4");
  CHECK_ERROR(q->shape[2] == k->shape[2] && q->shape[3] == k->shape[3],
              "ApplyPartialMRoPE sequence/head mismatch");
  apply_rope_tensor(q, config);
  apply_rope_tensor(k, config);
}

void ApplyPartialMRoPE_gpu(Tensor *q, Tensor *k, const QwenConfig &config) {
  ApplyPartialMRoPE(q, k, config);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionScoresGrouped(Tensor *q, Tensor *k, Tensor *scores,
                            size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = q->shape[0];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];
  const size_t heads_per_group = num_q_heads / num_kv_heads;

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < num_q_heads; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t kv_head = h / heads_per_group;
        const size_t q_base = ((b * num_q_heads + h) * T + tq) * D;
        const size_t score_base = ((b * num_q_heads + h) * T + tq) * T;
        for (size_t tk = 0; tk < T; ++tk) {
          const size_t k_base = ((b * num_kv_heads + kv_head) * T + tk) * D;
          float sum = 0.0f;
          for (size_t d = 0; d < D; ++d) {
            sum += q->buf[q_base + d] * k->buf[k_base + d];
          }
          scores->buf[score_base + tk] = sum;
        }
      }
    }
  }
}

void AttentionScoresGrouped_gpu(Tensor *q, Tensor *k, Tensor *scores,
                                size_t num_q_heads, size_t num_kv_heads) {
  AttentionScoresGrouped(q, k, scores, num_q_heads, num_kv_heads);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens) {
  const size_t B = scores->shape[0];
  const size_t H = scores->shape[1];
  const size_t T = scores->shape[2];
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t valid_t =
            (tokens != nullptr && tokens->lengths != nullptr)
                ? static_cast<size_t>(tokens->lengths[b])
                : T;
        const size_t row_base = ((b * H + h) * T + tq) * T;
        if (tq >= valid_t) {
          for (size_t tk = 0; tk < T; ++tk) {
            probs->buf[row_base + tk] = 0.0f;
          }
          continue;
        }

        const size_t row_end = std::min(tq, valid_t - 1);
        float row_max = -1.0e30f;
        for (size_t tk = 0; tk <= row_end; ++tk) {
          row_max = std::max(row_max, scores->buf[row_base + tk] * scale);
        }

        float sum = 0.0f;
        for (size_t tk = 0; tk < T; ++tk) {
          if (tk > row_end || tk >= valid_t) {
            probs->buf[row_base + tk] = 0.0f;
            continue;
          }
          const float e = std::exp(scores->buf[row_base + tk] * scale - row_max);
          probs->buf[row_base + tk] = e;
          sum += e;
        }
        for (size_t tk = 0; tk <= row_end; ++tk) {
          probs->buf[row_base + tk] /= sum;
        }
      }
    }
  }
}

void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs, size_t head_dim,
                          const TokenBatch *tokens) {
  ScaleMaskSoftmax(scores, probs, head_dim, tokens);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context,
                             size_t num_q_heads, size_t num_kv_heads) {
  CHECK_ERROR(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");

  const size_t B = probs->shape[0];
  const size_t T = probs->shape[2];
  const size_t D = v->shape[3];
  const size_t heads_per_group = num_q_heads / num_kv_heads;

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < num_q_heads; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t kv_head = h / heads_per_group;
        const size_t prob_base = ((b * num_q_heads + h) * T + tq) * T;
        const size_t out_base = ((b * num_q_heads + h) * T + tq) * D;
        for (size_t d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (size_t tk = 0; tk < T; ++tk) {
            const size_t v_base = ((b * num_kv_heads + kv_head) * T + tk) * D;
            sum += probs->buf[prob_base + tk] * v->buf[v_base + d];
          }
          context->buf[out_base + d] = sum;
        }
      }
    }
  }
}

void AttentionContextGrouped_gpu(Tensor *probs, Tensor *v, Tensor *context,
                                 size_t num_q_heads, size_t num_kv_heads) {
  AttentionContextGrouped(probs, v, context, num_q_heads, num_kv_heads);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void DepthwiseConv1dCausal(Tensor *input, Tensor *weight, Tensor *output,
                           size_t kernel_size) {
  CHECK_ERROR(input->ndim == 3 && weight->ndim == 3 && output->ndim == 3,
              "DepthwiseConv1dCausal rank mismatch");
  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t C = input->shape[2];
  CHECK_ERROR(weight->shape[0] == C && weight->shape[1] == 1 &&
                  weight->shape[2] == kernel_size,
              "DepthwiseConv1dCausal weight shape mismatch");
  CHECK_ERROR(output->shape[0] == B && output->shape[1] == T && output->shape[2] == C,
              "DepthwiseConv1dCausal output shape mismatch");

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (size_t k = 0; k < kernel_size; ++k) {
          if (t < k) {
            continue;
          }
          const size_t src_t = t - k;
          const float x = input->buf[(b * T + src_t) * C + c];
          const float w = weight->buf[c * kernel_size + k];
          sum += x * w;
        }
        output->buf[(b * T + t) * C + c] = sum / (1.0f + std::exp(-sum));
      }
    }
  }
}

void DepthwiseConv1dCausal_gpu(Tensor *input, Tensor *weight, Tensor *output,
                               size_t kernel_size) {
  DepthwiseConv1dCausal(input, weight, output, kernel_size);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void PrepareLinearDecay(Tensor *a, Tensor *a_log, Tensor *dt_bias, Tensor *output) {
  CHECK_ERROR(a->ndim == 3 && output->ndim == 3, "PrepareLinearDecay expects rank-3");
  CHECK_ERROR(a->shape[0] == output->shape[0] && a->shape[1] == output->shape[1] &&
                  a->shape[2] == output->shape[2],
              "PrepareLinearDecay shape mismatch");
  CHECK_ERROR(a_log->ndim == 1 && dt_bias->ndim == 1 &&
                  a_log->shape[0] == a->shape[1] && dt_bias->shape[0] == a->shape[1],
              "PrepareLinearDecay parameter shape mismatch");
  const size_t B = a->shape[0];
  const size_t H = a->shape[1];
  const size_t T = a->shape[2];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t t = 0; t < T; ++t) {
        const float x = a->buf[(b * H + h) * T + t] + dt_bias->buf[h];
        output->buf[(b * H + h) * T + t] =
            -std::exp(a_log->buf[h]) * softplus(x);
      }
    }
  }
}

void PrepareLinearDecay_gpu(Tensor *a, Tensor *a_log, Tensor *dt_bias,
                            Tensor *output) {
  PrepareLinearDecay(a, a_log, dt_bias, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void DeltaStateScan(Tensor *query, Tensor *key, Tensor *value, Tensor *g,
                    Tensor *beta, Tensor *output, const TokenBatch *tokens) {
  CHECK_ERROR(query->ndim == 4 && key->ndim == 4 && value->ndim == 4,
              "DeltaStateScan expects rank-4 q/k/v");
  CHECK_ERROR(g->ndim == 3 && beta->ndim == 3 && output->ndim == 4,
              "DeltaStateScan g/beta/output rank mismatch");

  const size_t B = query->shape[0];
  const size_t Hk = query->shape[1];
  const size_t T = query->shape[2];
  const size_t Dk = query->shape[3];
  const size_t Hv = value->shape[1];
  const size_t Dv = value->shape[3];
  CHECK_ERROR(key->shape[0] == B && key->shape[1] == Hk && key->shape[2] == T &&
                  key->shape[3] == Dk,
              "DeltaStateScan key shape mismatch");
  CHECK_ERROR(value->shape[0] == B && value->shape[2] == T,
              "DeltaStateScan value batch/sequence mismatch");
  CHECK_ERROR(g->shape[0] == B && g->shape[1] == Hv && g->shape[2] == T,
              "DeltaStateScan g shape mismatch");
  CHECK_ERROR(beta->shape[0] == B && beta->shape[1] == Hv && beta->shape[2] == T,
              "DeltaStateScan beta shape mismatch");
  CHECK_ERROR(output->shape[0] == B && output->shape[1] == Hv &&
                  output->shape[2] == T && output->shape[3] == Dv,
              "DeltaStateScan output shape mismatch");
  CHECK_ERROR(Hv % Hk == 0, "DeltaStateScan requires Hv divisible by Hk");

  const size_t heads_per_group = Hv / Hk;
  const float scale = 1.0f / std::sqrt(static_cast<float>(Dk));
  std::vector<float> state(B * Hv * Dk * Dv, 0.0f);

  for (size_t b = 0; b < B; ++b) {
    const size_t valid_t =
        (tokens != nullptr && tokens->lengths != nullptr)
            ? static_cast<size_t>(tokens->lengths[b])
            : T;
    for (size_t h = 0; h < Hv; ++h) {
      const size_t hk = h / heads_per_group;
      std::vector<float> q_norm(Dk, 0.0f);
      std::vector<float> k_norm(Dk, 0.0f);
      std::vector<float> delta(Dv, 0.0f);
      float *state_ptr = state.data() + ((b * Hv + h) * Dk * Dv);

      for (size_t t = 0; t < T; ++t) {
        float *out = output->buf + ((b * Hv + h) * T + t) * Dv;
        if (t >= valid_t) {
          std::memset(out, 0, Dv * sizeof(float));
          continue;
        }

        const float *q_ptr = query->buf + ((b * Hk + hk) * T + t) * Dk;
        const float *k_ptr = key->buf + ((b * Hk + hk) * T + t) * Dk;
        const float *v_ptr = value->buf + ((b * Hv + h) * T + t) * Dv;

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (size_t d = 0; d < Dk; ++d) {
          q_sum += q_ptr[d] * q_ptr[d];
          k_sum += k_ptr[d] * k_ptr[d];
        }
        const float q_inv = 1.0f / std::sqrt(q_sum + 1.0e-6f);
        const float k_inv = 1.0f / std::sqrt(k_sum + 1.0e-6f);
        for (size_t d = 0; d < Dk; ++d) {
          q_norm[d] = q_ptr[d] * q_inv * scale;
          k_norm[d] = k_ptr[d] * k_inv;
        }

        const float g_exp = std::exp(g->buf[(b * Hv + h) * T + t]);
        const float beta_t = beta->buf[(b * Hv + h) * T + t];

        for (size_t dk = 0; dk < Dk; ++dk) {
          for (size_t dv = 0; dv < Dv; ++dv) {
            state_ptr[dk * Dv + dv] *= g_exp;
          }
        }

        for (size_t dv = 0; dv < Dv; ++dv) {
          float kv_mem = 0.0f;
          for (size_t dk = 0; dk < Dk; ++dk) {
            kv_mem += state_ptr[dk * Dv + dv] * k_norm[dk];
          }
          delta[dv] = (v_ptr[dv] - kv_mem) * beta_t;
        }

        for (size_t dk = 0; dk < Dk; ++dk) {
          for (size_t dv = 0; dv < Dv; ++dv) {
            state_ptr[dk * Dv + dv] += k_norm[dk] * delta[dv];
          }
        }

        for (size_t dv = 0; dv < Dv; ++dv) {
          float sum = 0.0f;
          for (size_t dk = 0; dk < Dk; ++dk) {
            sum += state_ptr[dk * Dv + dv] * q_norm[dk];
          }
          out[dv] = sum;
        }
      }
    }
  }
}

void DeltaStateScan_gpu(Tensor *query, Tensor *key, Tensor *value, Tensor *g,
                        Tensor *beta, Tensor *output, const TokenBatch *tokens) {
  DeltaStateScan(query, key, value, g, beta, output, tokens);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void MergeHeads(Tensor *context, Tensor *merged) {
  const size_t B = context->shape[0];
  const size_t H = context->shape[1];
  const size_t T = context->shape[2];
  const size_t D = context->shape[3];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        const size_t src_base = ((b * H + h) * T + t) * D;
        const size_t dst_base = (b * T + t) * (H * D) + h * D;
        std::memcpy(merged->buf + dst_base, context->buf + src_base,
                    D * sizeof(float));
      }
    }
  }
}

void MergeHeads_gpu(Tensor *context, Tensor *merged) {
  MergeHeads(context, merged);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i] + addend->buf[i];
  }
}

void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output) {
  ResidualAdd(input, addend, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void SiLU(Tensor *inout) {
#pragma omp parallel for
  for (size_t i = 0; i < inout->num_elem(); ++i) {
    const float x = inout->buf[i];
    inout->buf[i] = x / (1.0f + std::exp(-x));
  }
}

void SiLU_gpu(Tensor *inout) {
  SiLU(inout);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void Sigmoid(Tensor *inout) {
#pragma omp parallel for
  for (size_t i = 0; i < inout->num_elem(); ++i) {
    const float x = inout->buf[i];
    inout->buf[i] = 1.0f / (1.0f + std::exp(-x));
  }
}

void Sigmoid_gpu(Tensor *inout) {
  Sigmoid(inout);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ElementwiseMul(Tensor *lhs, Tensor *rhs, Tensor *output) {
  CHECK_ERROR(lhs->num_elem() == rhs->num_elem() &&
                  lhs->num_elem() == output->num_elem(),
              "ElementwiseMul shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < lhs->num_elem(); ++i) {
    output->buf[i] = lhs->buf[i] * rhs->buf[i];
  }
}

void ElementwiseMul_gpu(Tensor *lhs, Tensor *rhs, Tensor *output) {
  ElementwiseMul(lhs, rhs, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void LMHead(Tensor *input, Tensor *weight, Tensor *output) {
  const size_t rows = flat_rows(input);
  const size_t hidden = last_dim(input);
  CHECK_ERROR(weight->ndim == 2 && weight->shape[1] == hidden,
              "LMHead weight shape mismatch");
  CHECK_ERROR(output->num_elem() == rows * weight->shape[0],
              "LMHead output shape mismatch");

  const size_t vocab_size = weight->shape[0];

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * hidden;
    float *out = output->buf + row * vocab_size;
    for (size_t vocab = 0; vocab < vocab_size; ++vocab) {
      const float *w = weight->buf + vocab * hidden;
      float sum = 0.0f;
      for (size_t c = 0; c < hidden; ++c) {
        sum += in[c] * w[c];
      }
      out[vocab] = sum;
    }
  }
}

void LMHead_gpu(Tensor *input, Tensor *weight, Tensor *output) {
  LMHead(input, weight, output);
  CHECK_CUDA(cudaDeviceSynchronize());
}
