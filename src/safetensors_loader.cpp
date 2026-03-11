#include "safetensors_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "util.h"

namespace {

float fp16_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x03ff;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -14;
      uint32_t m = mant;
      while ((m & 0x0400) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03ff;
      bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

}  // namespace

SafetensorsLoader::SafetensorsLoader(const char *model_dir) : model_dir_(model_dir) {
  load_index(model_dir);
}

SafetensorsLoader::~SafetensorsLoader() {
  for (auto &entry : shard_files_) {
    if (entry.second != nullptr) {
      std::fclose(entry.second);
    }
  }
}

void SafetensorsLoader::load_index(const char *model_dir) {
  const std::string index_path =
      std::string(model_dir) + "/model.safetensors.index.json";
  std::ifstream input(index_path);
  CHECK_ERROR(input.good(), "Failed to open %s", index_path.c_str());

  nlohmann::json json;
  input >> json;
  CHECK_ERROR(json.contains("weight_map") && json.at("weight_map").is_object(),
              "weight_map missing from %s", index_path.c_str());

  std::unordered_set<std::string> shards;
  for (auto it = json.at("weight_map").begin(); it != json.at("weight_map").end();
       ++it) {
    const std::string name = it.key();
    const std::string shard = it.value().get<std::string>();
    tensors_[name].shard_name = shard;
    shards.insert(shard);
  }

  for (const std::string &shard : shards) {
    load_shard_header(shard);
  }
}

void SafetensorsLoader::load_shard_header(const std::string &shard_name) {
  const std::string full_path = model_dir_ + "/" + shard_name;
  FILE *fp = std::fopen(full_path.c_str(), "rb");
  CHECK_ERROR(fp != nullptr, "Failed to open model shard %s", full_path.c_str());

  uint64_t header_len = 0;
  CHECK_ERROR(std::fread(&header_len, sizeof(uint64_t), 1, fp) == 1,
              "Failed to read header length from %s", full_path.c_str());
  std::vector<char> header_buf(static_cast<size_t>(header_len) + 1, '\0');
  CHECK_ERROR(std::fread(header_buf.data(), 1, static_cast<size_t>(header_len), fp) ==
                  static_cast<size_t>(header_len),
              "Failed to read safetensors header from %s", full_path.c_str());
  const size_t data_base = sizeof(uint64_t) + static_cast<size_t>(header_len);

  nlohmann::json header = nlohmann::json::parse(header_buf.data());
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") {
      continue;
    }
    if (!it.value().is_object() || !tensors_.count(it.key())) {
      continue;
    }
    TensorInfo &info = tensors_.at(it.key());
    info.dtype = it.value().at("dtype").get<std::string>();
    info.shape = it.value().at("shape").get<std::vector<size_t>>();
    const std::vector<size_t> offsets =
        it.value().at("data_offsets").get<std::vector<size_t>>();
    CHECK_ERROR(offsets.size() == 2, "Invalid data_offsets for tensor %s",
                it.key().c_str());
    info.begin = offsets[0];
    info.end = offsets[1];
    info.data_base = data_base;
  }

  shard_files_[shard_name] = fp;
}

FILE *SafetensorsLoader::open_shard(const std::string &shard_name) const {
  auto it = shard_files_.find(shard_name);
  CHECK_ERROR(it != shard_files_.end() && it->second != nullptr,
              "Unknown shard %s", shard_name.c_str());
  return it->second;
}

Parameter *SafetensorsLoader::load_parameter(
    const char *name, const std::vector<size_t> &expected_shape) const {
  auto it = tensors_.find(name);
  CHECK_ERROR(it != tensors_.end(), "Missing tensor %s in safetensors index",
              name);
  const TensorInfo &info = it->second;
  CHECK_ERROR(info.shape == expected_shape, "Tensor %s shape mismatch", name);
  CHECK_ERROR(info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16",
              "Tensor %s has unsupported dtype %s", name, info.dtype.c_str());

  size_t numel = 1;
  for (size_t dim : info.shape) {
    numel *= dim;
  }
  size_t elem_size = (info.dtype == "F32") ? sizeof(float) : sizeof(uint16_t);
  CHECK_ERROR((info.end - info.begin) == numel * elem_size,
              "Tensor %s byte size mismatch", name);

  Parameter *param = new Parameter(expected_shape);
  FILE *fp = open_shard(info.shard_name);
  CHECK_ERROR(std::fseek(fp, static_cast<long>(info.data_base + info.begin), SEEK_SET) ==
                  0,
              "Failed to seek tensor %s", name);

  if (info.dtype == "F32") {
    CHECK_ERROR(std::fread(param->buf, sizeof(float), numel, fp) == numel,
                "Failed to read tensor %s", name);
  } else {
    std::vector<uint16_t> tmp(numel);
    CHECK_ERROR(std::fread(tmp.data(), sizeof(uint16_t), numel, fp) == numel,
                "Failed to read tensor %s", name);
    for (size_t i = 0; i < numel; ++i) {
      if (info.dtype == "BF16") {
        uint32_t bits = static_cast<uint32_t>(tmp[i]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(float));
        param->buf[i] = value;
      } else {
        param->buf[i] = fp16_to_float(tmp[i]);
      }
    }
  }

  param->to_gpu();
  return param;
}

bool SafetensorsLoader::has_tensor(const char *name) const {
  return tensors_.find(name) != tensors_.end();
}
