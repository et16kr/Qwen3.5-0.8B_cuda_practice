#include "model.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

struct LayerWeights {
  QwenLayerType type = QwenLayerType::kLinearAttention;

  Parameter *input_norm_weight = nullptr;
  Parameter *post_attn_norm_weight = nullptr;
  Parameter *mlp_gate_weight = nullptr;
  Parameter *mlp_up_weight = nullptr;
  Parameter *mlp_down_weight = nullptr;

  Parameter *q_proj_weight = nullptr;
  Parameter *k_proj_weight = nullptr;
  Parameter *v_proj_weight = nullptr;
  Parameter *o_proj_weight = nullptr;
  Parameter *q_norm_weight = nullptr;
  Parameter *k_norm_weight = nullptr;

  Parameter *linear_in_proj_qkv = nullptr;
  Parameter *linear_conv1d_weight = nullptr;
  Parameter *linear_in_proj_a = nullptr;
  Parameter *linear_in_proj_b = nullptr;
  Parameter *linear_in_proj_z = nullptr;
  Parameter *linear_a_log = nullptr;
  Parameter *linear_dt_bias = nullptr;
  Parameter *linear_norm_weight = nullptr;
  Parameter *linear_out_proj = nullptr;
};

QwenConfig config_;

Parameter *tok_embeddings = nullptr;
std::vector<LayerWeights> layers_;
Parameter *final_norm_weight = nullptr;

Activation *x = nullptr;
Activation *residual = nullptr;
Activation *norm_buf = nullptr;
Activation *masked_hidden = nullptr;

Activation *q_proj_raw = nullptr;
Activation *q_flat = nullptr;
Activation *att_gate = nullptr;
Activation *k_flat = nullptr;
Activation *v_flat = nullptr;
Activation *q_heads = nullptr;
Activation *k_heads = nullptr;
Activation *v_heads = nullptr;
Activation *att_scores = nullptr;
Activation *att_probs = nullptr;
Activation *context = nullptr;
Activation *merged = nullptr;
Activation *attn_out = nullptr;

Activation *linear_qkv = nullptr;
Activation *linear_conv = nullptr;
Activation *linear_query = nullptr;
Activation *linear_key = nullptr;
Activation *linear_value = nullptr;
Activation *linear_z_flat = nullptr;
Activation *linear_z = nullptr;
Activation *linear_a_proj = nullptr;
Activation *linear_b_proj = nullptr;
Activation *linear_a = nullptr;
Activation *linear_beta = nullptr;
Activation *linear_g = nullptr;
Activation *linear_context = nullptr;
Activation *linear_out_heads = nullptr;
Activation *linear_merged = nullptr;

Activation *gate_buf = nullptr;
Activation *up_buf = nullptr;
Activation *gated_buf = nullptr;
Activation *mlp_out = nullptr;
Activation *final_norm = nullptr;

size_t current_batch = 0;
size_t current_seq = 0;
TokenBatch *current_tokens = nullptr;

void delete_tensor(Tensor *&tensor) {
  if (tensor != nullptr) {
    delete tensor;
    tensor = nullptr;
  }
}

void delete_layer_weights(LayerWeights *layer) {
  delete_tensor(layer->input_norm_weight);
  delete_tensor(layer->post_attn_norm_weight);
  delete_tensor(layer->mlp_gate_weight);
  delete_tensor(layer->mlp_up_weight);
  delete_tensor(layer->mlp_down_weight);

  delete_tensor(layer->q_proj_weight);
  delete_tensor(layer->k_proj_weight);
  delete_tensor(layer->v_proj_weight);
  delete_tensor(layer->o_proj_weight);
  delete_tensor(layer->q_norm_weight);
  delete_tensor(layer->k_norm_weight);

  delete_tensor(layer->linear_in_proj_qkv);
  delete_tensor(layer->linear_conv1d_weight);
  delete_tensor(layer->linear_in_proj_a);
  delete_tensor(layer->linear_in_proj_b);
  delete_tensor(layer->linear_in_proj_z);
  delete_tensor(layer->linear_a_log);
  delete_tensor(layer->linear_dt_bias);
  delete_tensor(layer->linear_norm_weight);
  delete_tensor(layer->linear_out_proj);
}

void load_layer_parameters(SafetensorsLoader *loader, size_t layer_idx) {
  LayerWeights layer;
  layer.type = config_.layer_types[layer_idx];

  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer_idx) + ".";
  const size_t hidden = config_.hidden_size;
  const size_t intermediate = config_.intermediate_size;

  layer.input_norm_weight =
      loader->load_parameter((prefix + "input_layernorm.weight").c_str(), {hidden});
  layer.post_attn_norm_weight = loader->load_parameter(
      (prefix + "post_attention_layernorm.weight").c_str(), {hidden});
  layer.mlp_gate_weight =
      loader->load_parameter((prefix + "mlp.gate_proj.weight").c_str(),
                             {intermediate, hidden});
  layer.mlp_up_weight =
      loader->load_parameter((prefix + "mlp.up_proj.weight").c_str(),
                             {intermediate, hidden});
  layer.mlp_down_weight =
      loader->load_parameter((prefix + "mlp.down_proj.weight").c_str(),
                             {hidden, intermediate});

  if (layer.type == QwenLayerType::kFullAttention) {
    layer.q_proj_weight = loader->load_parameter(
        (prefix + "self_attn.q_proj.weight").c_str(),
        {config_.full_q_hidden() * 2, hidden});
    layer.k_proj_weight = loader->load_parameter(
        (prefix + "self_attn.k_proj.weight").c_str(),
        {config_.full_kv_hidden(), hidden});
    layer.v_proj_weight = loader->load_parameter(
        (prefix + "self_attn.v_proj.weight").c_str(),
        {config_.full_kv_hidden(), hidden});
    layer.o_proj_weight = loader->load_parameter(
        (prefix + "self_attn.o_proj.weight").c_str(),
        {hidden, config_.full_q_hidden()});
    layer.q_norm_weight = loader->load_parameter(
        (prefix + "self_attn.q_norm.weight").c_str(), {config_.head_dim});
    layer.k_norm_weight = loader->load_parameter(
        (prefix + "self_attn.k_norm.weight").c_str(), {config_.head_dim});
  } else {
    layer.linear_in_proj_qkv = loader->load_parameter(
        (prefix + "linear_attn.in_proj_qkv.weight").c_str(),
        {config_.linear_key_hidden() * 2 + config_.linear_value_hidden(), hidden});
    layer.linear_conv1d_weight = loader->load_parameter(
        (prefix + "linear_attn.conv1d.weight").c_str(),
        {config_.linear_key_hidden() * 2 + config_.linear_value_hidden(), 1,
         config_.linear_conv_kernel_dim});
    layer.linear_in_proj_a = loader->load_parameter(
        (prefix + "linear_attn.in_proj_a.weight").c_str(),
        {config_.linear_num_value_heads, hidden});
    layer.linear_in_proj_b = loader->load_parameter(
        (prefix + "linear_attn.in_proj_b.weight").c_str(),
        {config_.linear_num_value_heads, hidden});
    layer.linear_in_proj_z = loader->load_parameter(
        (prefix + "linear_attn.in_proj_z.weight").c_str(),
        {config_.linear_value_hidden(), hidden});
    layer.linear_a_log = loader->load_parameter(
        (prefix + "linear_attn.A_log").c_str(), {config_.linear_num_value_heads});
    layer.linear_dt_bias = loader->load_parameter(
        (prefix + "linear_attn.dt_bias").c_str(), {config_.linear_num_value_heads});
    layer.linear_norm_weight = loader->load_parameter(
        (prefix + "linear_attn.norm.weight").c_str(), {config_.linear_value_head_dim});
    layer.linear_out_proj = loader->load_parameter(
        (prefix + "linear_attn.out_proj.weight").c_str(),
        {hidden, config_.linear_value_hidden()});
  }

  layers_.push_back(layer);
}

void load_parameters(const char *model_dir) {
  char config_path[512];
  std::snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

  config_ = load_qwen_config(config_path);
  SafetensorsLoader loader(model_dir);

  tok_embeddings = loader.load_parameter("model.language_model.embed_tokens.weight",
                                         {config_.vocab_size, config_.hidden_size});
  for (size_t layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
    load_layer_parameters(&loader, layer_idx);
  }
  final_norm_weight = loader.load_parameter("model.language_model.norm.weight",
                                            {config_.hidden_size});
}

void free_parameters() {
  delete_tensor(tok_embeddings);
  for (LayerWeights &layer : layers_) {
    delete_layer_weights(&layer);
  }
  layers_.clear();
  delete_tensor(final_norm_weight);
}

void apply_mlp(const LayerWeights &layer) {
  QwenRMSNorm(x, layer.post_attn_norm_weight, norm_buf, config_.rms_norm_eps);
  Linear(norm_buf, layer.mlp_gate_weight, gate_buf);
  Linear(norm_buf, layer.mlp_up_weight, up_buf);
  SiLU(gate_buf);
  ElementwiseMul(gate_buf, up_buf, gated_buf);
  Linear(gated_buf, layer.mlp_down_weight, mlp_out);
  ResidualAdd(x, mlp_out, residual);
  std::swap(x, residual);
}

void full_attention_block(const LayerWeights &layer) {
  QwenRMSNorm(x, layer.input_norm_weight, norm_buf, config_.rms_norm_eps);

  Linear(norm_buf, layer.q_proj_weight, q_proj_raw);
  SplitTensorLastDim(q_proj_raw, config_.full_q_hidden(), q_flat, att_gate);
  Linear(norm_buf, layer.k_proj_weight, k_flat);
  Linear(norm_buf, layer.v_proj_weight, v_flat);

  SplitHeads(q_flat, q_heads, config_.num_attention_heads, config_.head_dim);
  SplitHeads(k_flat, k_heads, config_.num_key_value_heads, config_.head_dim);
  SplitHeads(v_flat, v_heads, config_.num_key_value_heads, config_.head_dim);

  QwenRMSNorm(q_heads, layer.q_norm_weight, q_heads, config_.rms_norm_eps);
  QwenRMSNorm(k_heads, layer.k_norm_weight, k_heads, config_.rms_norm_eps);
  ApplyPartialMRoPE(q_heads, k_heads, config_);

  AttentionScoresGrouped(q_heads, k_heads, att_scores, config_.num_attention_heads,
                         config_.num_key_value_heads);
  ScaleMaskSoftmax(att_scores, att_probs, config_.head_dim, current_tokens);
  AttentionContextGrouped(att_probs, v_heads, context, config_.num_attention_heads,
                          config_.num_key_value_heads);
  MergeHeads(context, merged);

  Sigmoid(att_gate);
  ElementwiseMul(merged, att_gate, q_flat);
  Linear(q_flat, layer.o_proj_weight, attn_out);

  ResidualAdd(x, attn_out, residual);
  std::swap(x, residual);

  apply_mlp(layer);
}

void linear_attention_block(const LayerWeights &layer) {
  QwenRMSNorm(x, layer.input_norm_weight, norm_buf, config_.rms_norm_eps);
  MaskPaddingHiddenStates(norm_buf, masked_hidden, current_tokens);

  Linear(masked_hidden, layer.linear_in_proj_qkv, linear_qkv);
  DepthwiseConv1dCausal(linear_qkv, layer.linear_conv1d_weight, linear_conv,
                        config_.linear_conv_kernel_dim);
  SplitLinearQKV(linear_conv, linear_query, linear_key, linear_value,
                 config_.linear_num_key_heads, config_.linear_key_head_dim,
                 config_.linear_num_value_heads, config_.linear_value_head_dim);

  Linear(masked_hidden, layer.linear_in_proj_z, linear_z_flat);
  SplitHeads(linear_z_flat, linear_z, config_.linear_num_value_heads,
             config_.linear_value_head_dim);

  Linear(masked_hidden, layer.linear_in_proj_a, linear_a_proj);
  Linear(masked_hidden, layer.linear_in_proj_b, linear_b_proj);
  SplitHeadScalars(linear_a_proj, linear_a);
  SplitHeadScalars(linear_b_proj, linear_beta);

  PrepareLinearDecay(linear_a, layer.linear_a_log, layer.linear_dt_bias, linear_g);
  Sigmoid(linear_beta);
  DeltaStateScan(linear_query, linear_key, linear_value, linear_g, linear_beta,
                 linear_context, current_tokens);

  QwenRMSNormGated(linear_context, linear_z, layer.linear_norm_weight,
                   linear_out_heads, config_.rms_norm_eps);
  MergeHeads(linear_out_heads, linear_merged);
  Linear(linear_merged, layer.linear_out_proj, attn_out);

  ResidualAdd(x, attn_out, residual);
  std::swap(x, residual);

  apply_mlp(layer);
}

void transformer_block(size_t layer_idx) {
  const LayerWeights &layer = layers_[layer_idx];
  if (layer.type == QwenLayerType::kFullAttention) {
    full_attention_block(layer);
  } else {
    linear_attention_block(layer);
  }
}

void qwen_forward_cpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  EmbeddingLookup(tokens, tok_embeddings, x);
  for (size_t layer_idx = 0; layer_idx < config_.num_hidden_layers; ++layer_idx) {
    transformer_block(layer_idx);
  }
  QwenRMSNorm(x, final_norm_weight, final_norm, config_.rms_norm_eps);
  LMHead(final_norm, tok_embeddings, logits);
}

}  // namespace

TokenBatch load_tokens(const char *path) {
  FILE *f = std::fopen(path, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open token file %s", path);

  int32_t B = 0;
  int32_t T = 0;
  CHECK_ERROR(std::fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(std::fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);
  CHECK_ERROR(B > 0 && T > 0, "Invalid token shape in %s", path);

  CHECK_ERROR(std::fseek(f, 0, SEEK_END) == 0, "Failed to seek %s", path);
  long file_size = std::ftell(f);
  CHECK_ERROR(file_size >= 0, "Failed to stat %s", path);
  std::rewind(f);
  CHECK_ERROR(std::fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(std::fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);

  TokenBatch batch(static_cast<size_t>(B), static_cast<size_t>(T));
  const size_t tokens_bytes = static_cast<size_t>(B) * static_cast<size_t>(T) *
                              sizeof(int32_t);
  const size_t lengths_bytes = static_cast<size_t>(B) * sizeof(int32_t);
  const size_t header_bytes = 2 * sizeof(int32_t);
  const size_t file_bytes = static_cast<size_t>(file_size);

  bool has_lengths = false;
  if (file_bytes == header_bytes + tokens_bytes) {
    has_lengths = false;
  } else if (file_bytes == header_bytes + lengths_bytes + tokens_bytes) {
    has_lengths = true;
  } else {
    CHECK_ERROR(false, "Unsupported token file size for %s", path);
  }

  if (has_lengths) {
    CHECK_ERROR(std::fread(batch.lengths, sizeof(int32_t), static_cast<size_t>(B), f) ==
                    static_cast<size_t>(B),
                "Failed to read lengths from %s", path);
  } else {
    for (size_t b = 0; b < batch.B; ++b) {
      batch.lengths[b] = static_cast<int32_t>(batch.T);
    }
  }

  const size_t expected = static_cast<size_t>(B) * static_cast<size_t>(T);
  CHECK_ERROR(std::fread(batch.buf, sizeof(int32_t), expected, f) == expected,
              "Failed to read token ids from %s", path);
  const int trailing = std::fgetc(f);
  std::fclose(f);
  CHECK_ERROR(trailing == EOF, "Unexpected trailing bytes in token file %s", path);

  for (size_t b = 0; b < batch.B; ++b) {
    CHECK_ERROR(batch.lengths[b] > 0 &&
                    batch.lengths[b] <= static_cast<int32_t>(batch.T),
                "Invalid sequence length %d in %s", batch.lengths[b], path);
  }
  batch.to_gpu();
  return batch;
}

void initialize_model(const char *model_dir) { load_parameters(model_dir); }

void alloc_activations(size_t batch_size, size_t seq_len) {
  CHECK_ERROR(batch_size > 0 && seq_len > 0, "Activation shape must be positive");
  CHECK_ERROR(seq_len <= config_.max_position_embeddings,
              "Sequence length %zu exceeds max_position_embeddings %zu", seq_len,
              config_.max_position_embeddings);

  free_activations();

  current_batch = batch_size;
  current_seq = seq_len;

  const size_t hidden = config_.hidden_size;
  const size_t intermediate = config_.intermediate_size;
  const size_t full_q_hidden = config_.full_q_hidden();
  const size_t full_kv_hidden = config_.full_kv_hidden();
  const size_t linear_qkv_hidden =
      config_.linear_key_hidden() * 2 + config_.linear_value_hidden();

  x = new Activation({batch_size, seq_len, hidden});
  residual = new Activation({batch_size, seq_len, hidden});
  norm_buf = new Activation({batch_size, seq_len, hidden});
  masked_hidden = new Activation({batch_size, seq_len, hidden});

  q_proj_raw = new Activation({batch_size, seq_len, full_q_hidden * 2});
  q_flat = new Activation({batch_size, seq_len, full_q_hidden});
  att_gate = new Activation({batch_size, seq_len, full_q_hidden});
  k_flat = new Activation({batch_size, seq_len, full_kv_hidden});
  v_flat = new Activation({batch_size, seq_len, full_kv_hidden});
  q_heads = new Activation(
      {batch_size, config_.num_attention_heads, seq_len, config_.head_dim});
  k_heads = new Activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim});
  v_heads = new Activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim});
  att_scores =
      new Activation({batch_size, config_.num_attention_heads, seq_len, seq_len});
  att_probs =
      new Activation({batch_size, config_.num_attention_heads, seq_len, seq_len});
  context = new Activation(
      {batch_size, config_.num_attention_heads, seq_len, config_.head_dim});
  merged = new Activation({batch_size, seq_len, full_q_hidden});
  attn_out = new Activation({batch_size, seq_len, hidden});

  linear_qkv = new Activation({batch_size, seq_len, linear_qkv_hidden});
  linear_conv = new Activation({batch_size, seq_len, linear_qkv_hidden});
  linear_query = new Activation({batch_size, config_.linear_num_key_heads, seq_len,
                                 config_.linear_key_head_dim});
  linear_key = new Activation({batch_size, config_.linear_num_key_heads, seq_len,
                               config_.linear_key_head_dim});
  linear_value = new Activation({batch_size, config_.linear_num_value_heads, seq_len,
                                 config_.linear_value_head_dim});
  linear_z_flat = new Activation({batch_size, seq_len, config_.linear_value_hidden()});
  linear_z = new Activation({batch_size, config_.linear_num_value_heads, seq_len,
                             config_.linear_value_head_dim});
  linear_a_proj =
      new Activation({batch_size, seq_len, config_.linear_num_value_heads});
  linear_b_proj =
      new Activation({batch_size, seq_len, config_.linear_num_value_heads});
  linear_a =
      new Activation({batch_size, config_.linear_num_value_heads, seq_len});
  linear_beta =
      new Activation({batch_size, config_.linear_num_value_heads, seq_len});
  linear_g =
      new Activation({batch_size, config_.linear_num_value_heads, seq_len});
  linear_context = new Activation(
      {batch_size, config_.linear_num_value_heads, seq_len,
       config_.linear_value_head_dim});
  linear_out_heads = new Activation(
      {batch_size, config_.linear_num_value_heads, seq_len,
       config_.linear_value_head_dim});
  linear_merged =
      new Activation({batch_size, seq_len, config_.linear_value_hidden()});

  gate_buf = new Activation({batch_size, seq_len, intermediate});
  up_buf = new Activation({batch_size, seq_len, intermediate});
  gated_buf = new Activation({batch_size, seq_len, intermediate});
  mlp_out = new Activation({batch_size, seq_len, hidden});
  final_norm = new Activation({batch_size, seq_len, hidden});
}

void qwen_forward(TokenBatch *tokens, Tensor *logits) {
  current_tokens = tokens;
  qwen_forward_cpu(tokens, logits);
  current_tokens = nullptr;

  // TODO(student): Replace the CPU path with GPU kernels layer by layer.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void validate_against_cpu(TokenBatch *tokens, Tensor *logits_gpu) {
  Tensor reference({tokens->B, tokens->T, config_.vocab_size});
  current_tokens = tokens;
  qwen_forward_cpu(tokens, &reference);
  current_tokens = nullptr;

  int diff = validate_buffer(logits_gpu->buf, reference.buf, reference.num_elem(),
                             1.0e-3f, 1.0e-3f);
  if (diff < 0) {
    std::printf("Validation: PASSED\n");
    return;
  }

  std::printf("Validation: FAILED\n");
  std::printf("First mismatch at index %d: output=%f reference=%f\n", diff,
              logits_gpu->buf[diff], reference.buf[diff]);
  EXIT(EXIT_FAILURE);
}

void finalize_model() { free_parameters(); }

void free_activations() {
  delete_tensor(x);
  delete_tensor(residual);
  delete_tensor(norm_buf);
  delete_tensor(masked_hidden);
  delete_tensor(q_proj_raw);
  delete_tensor(q_flat);
  delete_tensor(att_gate);
  delete_tensor(k_flat);
  delete_tensor(v_flat);
  delete_tensor(q_heads);
  delete_tensor(k_heads);
  delete_tensor(v_heads);
  delete_tensor(att_scores);
  delete_tensor(att_probs);
  delete_tensor(context);
  delete_tensor(merged);
  delete_tensor(attn_out);
  delete_tensor(linear_qkv);
  delete_tensor(linear_conv);
  delete_tensor(linear_query);
  delete_tensor(linear_key);
  delete_tensor(linear_value);
  delete_tensor(linear_z_flat);
  delete_tensor(linear_z);
  delete_tensor(linear_a_proj);
  delete_tensor(linear_b_proj);
  delete_tensor(linear_a);
  delete_tensor(linear_beta);
  delete_tensor(linear_g);
  delete_tensor(linear_context);
  delete_tensor(linear_out_heads);
  delete_tensor(linear_merged);
  delete_tensor(gate_buf);
  delete_tensor(up_buf);
  delete_tensor(gated_buf);
  delete_tensor(mlp_out);
  delete_tensor(final_norm);
  current_batch = 0;
  current_seq = 0;
}

const QwenConfig &model_config() { return config_; }
