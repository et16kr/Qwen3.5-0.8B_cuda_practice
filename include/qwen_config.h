#pragma once

#include <cstddef>
#include <string>
#include <vector>

enum class QwenLayerType {
  kLinearAttention,
  kFullAttention,
};

struct QwenConfig {
  size_t vocab_size = 0;
  size_t hidden_size = 0;
  size_t intermediate_size = 0;
  size_t num_hidden_layers = 0;
  size_t num_attention_heads = 0;
  size_t num_key_value_heads = 0;
  size_t head_dim = 0;
  size_t linear_num_key_heads = 0;
  size_t linear_num_value_heads = 0;
  size_t linear_key_head_dim = 0;
  size_t linear_value_head_dim = 0;
  size_t linear_conv_kernel_dim = 0;
  size_t max_position_embeddings = 0;
  float rms_norm_eps = 1.0e-6f;
  float rope_theta = 10000000.0f;
  float partial_rotary_factor = 1.0f;
  bool mrope_interleaved = false;
  bool tie_word_embeddings = true;
  int bos_token_id = -1;
  int pad_token_id = -1;
  std::vector<int> eos_token_ids;
  std::vector<int> mrope_section;
  std::vector<QwenLayerType> layer_types;

  size_t full_q_hidden() const { return num_attention_heads * head_dim; }
  size_t full_kv_hidden() const { return num_key_value_heads * head_dim; }
  size_t linear_key_hidden() const {
    return linear_num_key_heads * linear_key_head_dim;
  }
  size_t linear_value_hidden() const {
    return linear_num_value_heads * linear_value_head_dim;
  }
  size_t full_gate_hidden() const { return full_q_hidden(); }
  size_t rotary_dim() const;

  bool is_eos(int token_id) const;
  int primary_eos_token_id() const;
};

QwenConfig load_qwen_config(const char *path);
