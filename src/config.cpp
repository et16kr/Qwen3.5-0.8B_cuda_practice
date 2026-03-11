#include "qwen_config.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

template <typename T>
T read_required(const nlohmann::json &obj, const char *key) {
  if (!obj.contains(key)) {
    throw std::runtime_error(std::string("missing config key: ") + key);
  }
  return obj.at(key).get<T>();
}

template <typename T>
T read_optional(const nlohmann::json &obj, const char *key, const T &fallback) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return fallback;
  }
  return obj.at(key).get<T>();
}

std::vector<int> read_eos_ids(const nlohmann::json &obj) {
  if (!obj.contains("eos_token_id")) {
    return {};
  }
  const nlohmann::json &value = obj.at("eos_token_id");
  if (value.is_array()) {
    return value.get<std::vector<int>>();
  }
  return {value.get<int>()};
}

QwenLayerType parse_layer_type(const std::string &name) {
  if (name == "linear_attention") {
    return QwenLayerType::kLinearAttention;
  }
  if (name == "full_attention") {
    return QwenLayerType::kFullAttention;
  }
  throw std::runtime_error("unsupported layer type: " + name);
}

}  // namespace

size_t QwenConfig::rotary_dim() const {
  size_t dim = static_cast<size_t>(static_cast<float>(head_dim) * partial_rotary_factor);
  dim = std::min(dim, head_dim);
  if ((dim % 2) != 0) {
    --dim;
  }
  return dim;
}

bool QwenConfig::is_eos(int token_id) const {
  for (int eos_id : eos_token_ids) {
    if (eos_id == token_id) {
      return true;
    }
  }
  return false;
}

int QwenConfig::primary_eos_token_id() const {
  return eos_token_ids.empty() ? -1 : eos_token_ids.front();
}

QwenConfig load_qwen_config(const char *path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(std::string("failed to open config: ") + path);
  }

  nlohmann::json json;
  input >> json;
  if (!json.contains("text_config") || !json.at("text_config").is_object()) {
    throw std::runtime_error("config.json is missing text_config");
  }
  const nlohmann::json &text = json.at("text_config");

  QwenConfig cfg;
  cfg.vocab_size = read_required<size_t>(text, "vocab_size");
  cfg.hidden_size = read_required<size_t>(text, "hidden_size");
  cfg.intermediate_size = read_required<size_t>(text, "intermediate_size");
  cfg.num_hidden_layers = read_required<size_t>(text, "num_hidden_layers");
  cfg.num_attention_heads = read_required<size_t>(text, "num_attention_heads");
  cfg.num_key_value_heads = read_required<size_t>(text, "num_key_value_heads");
  cfg.head_dim = read_required<size_t>(text, "head_dim");
  cfg.linear_num_key_heads = read_required<size_t>(text, "linear_num_key_heads");
  cfg.linear_num_value_heads =
      read_required<size_t>(text, "linear_num_value_heads");
  cfg.linear_key_head_dim = read_required<size_t>(text, "linear_key_head_dim");
  cfg.linear_value_head_dim =
      read_required<size_t>(text, "linear_value_head_dim");
  cfg.linear_conv_kernel_dim =
      read_required<size_t>(text, "linear_conv_kernel_dim");
  cfg.max_position_embeddings =
      read_required<size_t>(text, "max_position_embeddings");
  cfg.rms_norm_eps = read_optional<float>(text, "rms_norm_eps", 1.0e-6f);
  cfg.tie_word_embeddings =
      read_optional<bool>(text, "tie_word_embeddings", true);
  cfg.eos_token_ids = read_eos_ids(text);
  cfg.pad_token_id = read_optional<int>(json, "pad_token_id", -1);
  cfg.bos_token_id = read_optional<int>(json, "bos_token_id", -1);

  if (text.contains("layer_types")) {
    for (const nlohmann::json &item : text.at("layer_types")) {
      cfg.layer_types.push_back(parse_layer_type(item.get<std::string>()));
    }
  }
  if (cfg.layer_types.empty()) {
    throw std::runtime_error("text_config.layer_types must not be empty");
  }

  if (text.contains("rope_parameters") && text.at("rope_parameters").is_object()) {
    const nlohmann::json &rope = text.at("rope_parameters");
    cfg.rope_theta = read_optional<float>(rope, "rope_theta", cfg.rope_theta);
    cfg.partial_rotary_factor =
        read_optional<float>(rope, "partial_rotary_factor", 1.0f);
    cfg.mrope_interleaved =
        read_optional<bool>(rope, "mrope_interleaved", false);
    cfg.mrope_section = read_optional<std::vector<int>>(rope, "mrope_section", {});
  }

  if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.intermediate_size == 0 ||
      cfg.num_hidden_layers == 0 || cfg.num_attention_heads == 0 ||
      cfg.num_key_value_heads == 0 || cfg.head_dim == 0 ||
      cfg.linear_num_key_heads == 0 || cfg.linear_num_value_heads == 0 ||
      cfg.linear_key_head_dim == 0 || cfg.linear_value_head_dim == 0 ||
      cfg.linear_conv_kernel_dim == 0 || cfg.max_position_embeddings == 0) {
    throw std::runtime_error("config contains zero-sized dimensions");
  }
  if (cfg.layer_types.size() != cfg.num_hidden_layers) {
    throw std::runtime_error("layer_types size must equal num_hidden_layers");
  }
  if ((cfg.head_dim % 2) != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }
  if ((cfg.linear_key_head_dim % 2) != 0) {
    throw std::runtime_error("linear_key_head_dim must be even");
  }
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
    throw std::runtime_error(
        "num_attention_heads must be divisible by num_key_value_heads");
  }
  return cfg;
}
