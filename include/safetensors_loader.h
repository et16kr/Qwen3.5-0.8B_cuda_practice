#pragma once

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

class SafetensorsLoader {
 public:
  explicit SafetensorsLoader(const char *model_dir);
  ~SafetensorsLoader();

  Parameter *load_parameter(const char *name,
                            const std::vector<size_t> &expected_shape) const;
  bool has_tensor(const char *name) const;

 private:
  struct TensorInfo {
    std::string shard_name;
    std::string dtype;
    std::vector<size_t> shape;
    size_t begin = 0;
    size_t end = 0;
    size_t data_base = 0;
  };

  std::string model_dir_;
  std::unordered_map<std::string, TensorInfo> tensors_;
  std::unordered_map<std::string, FILE *> shard_files_;

  void load_index(const char *model_dir);
  void load_shard_header(const std::string &shard_name);
  FILE *open_shard(const std::string &shard_name) const;
};
