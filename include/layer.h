#pragma once

#include "qwen_config.h"
#include "tensor.h"

void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output);
void EmbeddingLookup_gpu(TokenBatch *tokens, Tensor *embedding, Tensor *output);

void QwenRMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps);
void QwenRMSNorm_gpu(Tensor *input, Tensor *weight, Tensor *output, float eps);

void QwenRMSNormGated(Tensor *input, Tensor *gate, Tensor *weight, Tensor *output,
                      float eps);
void QwenRMSNormGated_gpu(Tensor *input, Tensor *gate, Tensor *weight, Tensor *output,
                          float eps);

void MaskPaddingHiddenStates(Tensor *input, Tensor *output, const TokenBatch *tokens);
void MaskPaddingHiddenStates_gpu(Tensor *input, Tensor *output,
                                 const TokenBatch *tokens);

void Linear(Tensor *input, Tensor *weight, Tensor *output);
void Linear_gpu(Tensor *input, Tensor *weight, Tensor *output);

void SplitTensorLastDim(Tensor *input, size_t left_size, Tensor *left, Tensor *right);
void SplitTensorLastDim_gpu(Tensor *input, size_t left_size, Tensor *left,
                            Tensor *right);

void SplitHeads(Tensor *input, Tensor *output, size_t num_heads, size_t head_dim);
void SplitHeads_gpu(Tensor *input, Tensor *output, size_t num_heads,
                    size_t head_dim);

void SplitHeadScalars(Tensor *input, Tensor *output);
void SplitHeadScalars_gpu(Tensor *input, Tensor *output);

void SplitLinearQKV(Tensor *input, Tensor *query, Tensor *key, Tensor *value,
                    size_t num_k_heads, size_t head_k_dim, size_t num_v_heads,
                    size_t head_v_dim);
void SplitLinearQKV_gpu(Tensor *input, Tensor *query, Tensor *key, Tensor *value,
                        size_t num_k_heads, size_t head_k_dim,
                        size_t num_v_heads, size_t head_v_dim);

void ApplyPartialMRoPE(Tensor *q, Tensor *k, const QwenConfig &config);
void ApplyPartialMRoPE_gpu(Tensor *q, Tensor *k, const QwenConfig &config);

void AttentionScoresGrouped(Tensor *q, Tensor *k, Tensor *scores,
                            size_t num_q_heads, size_t num_kv_heads);
void AttentionScoresGrouped_gpu(Tensor *q, Tensor *k, Tensor *scores,
                                size_t num_q_heads, size_t num_kv_heads);

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens);
void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs, size_t head_dim,
                          const TokenBatch *tokens);

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context,
                             size_t num_q_heads, size_t num_kv_heads);
void AttentionContextGrouped_gpu(Tensor *probs, Tensor *v, Tensor *context,
                                 size_t num_q_heads, size_t num_kv_heads);

void DepthwiseConv1dCausal(Tensor *input, Tensor *weight, Tensor *output,
                           size_t kernel_size);
void DepthwiseConv1dCausal_gpu(Tensor *input, Tensor *weight, Tensor *output,
                               size_t kernel_size);

void PrepareLinearDecay(Tensor *a, Tensor *a_log, Tensor *dt_bias, Tensor *output);
void PrepareLinearDecay_gpu(Tensor *a, Tensor *a_log, Tensor *dt_bias,
                            Tensor *output);

void DeltaStateScan(Tensor *query, Tensor *key, Tensor *value, Tensor *g,
                    Tensor *beta, Tensor *output, const TokenBatch *tokens);
void DeltaStateScan_gpu(Tensor *query, Tensor *key, Tensor *value, Tensor *g,
                        Tensor *beta, Tensor *output, const TokenBatch *tokens);

void MergeHeads(Tensor *context, Tensor *merged);
void MergeHeads_gpu(Tensor *context, Tensor *merged);

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output);
void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output);

void SiLU(Tensor *inout);
void SiLU_gpu(Tensor *inout);

void Sigmoid(Tensor *inout);
void Sigmoid_gpu(Tensor *inout);

void ElementwiseMul(Tensor *lhs, Tensor *rhs, Tensor *output);
void ElementwiseMul_gpu(Tensor *lhs, Tensor *rhs, Tensor *output);

void LMHead(Tensor *input, Tensor *weight, Tensor *output);
void LMHead_gpu(Tensor *input, Tensor *weight, Tensor *output);
