/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/utils/singleton.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

#include "oneapi/tbb.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"

namespace paddle {
namespace inference {
namespace openvino {
/*
 * OpenVINO Engine.
 *
 * There are two alternative ways to use it, one is to build from a paddle
 * protobuf model, another way is to manually construct the network.
 */

static inline ov::element::Type PhiType2OVType(phi::DataType type) {
  static const std::map<phi::DataType, ov::element::Type> type_map{
      {phi::DataType::BOOL, ov::element::boolean},
      {phi::DataType::INT16, ov::element::i16},
      {phi::DataType::INT32, ov::element::i32},
      {phi::DataType::INT64, ov::element::i64},
      {phi::DataType::FLOAT16, ov::element::f16},
      {phi::DataType::FLOAT32, ov::element::f32},
      {phi::DataType::FLOAT64, ov::element::f64},
      {phi::DataType::UINT8, ov::element::u8},
      {phi::DataType::INT8, ov::element::i8},
      {phi::DataType::BFLOAT16, ov::element::bf16}};
  auto it = type_map.find(type);
  PADDLE_ENFORCE_EQ(
      it != type_map.end(),
      true,
      common::errors::InvalidArgument(
          "phi::DataType[%s] not supported convert to ov::element::Type .",
          type));
  return it->second;
}

static inline phi::DataType OVType2PhiType(ov::element::Type type) {
  static const std::map<ov::element::Type, phi::DataType> type_map{
      {ov::element::boolean, phi::DataType::BOOL},
      {ov::element::i16, phi::DataType::INT16},
      {ov::element::i32, phi::DataType::INT32},
      {ov::element::i64, phi::DataType::INT64},
      {ov::element::f16, phi::DataType::FLOAT16},
      {ov::element::f32, phi::DataType::FLOAT32},
      {ov::element::f64, phi::DataType::FLOAT64},
      {ov::element::u8, phi::DataType::UINT8},
      {ov::element::i8, phi::DataType::INT8},
      {ov::element::bf16, phi::DataType::BFLOAT16}};
  auto it = type_map.find(type);
  PADDLE_ENFORCE_EQ(
      it != type_map.end(),
      true,
      common::errors::InvalidArgument(
          "phi::DataType[%s] not supported convert to ov::element::Type .",
          type));
  return it->second;
}

class OpenVINOEngine {
 public:
  /*
   * Construction parameters of OpenVINOEngine.
   */
  struct ConstructionParams {
    std::string model_program_path;
    std::string model_params_path;
    std::string model_opt_cache_dir;
    int cpu_math_library_num_threads;
    std::string inference_precision;
  };

  explicit OpenVINOEngine(const ConstructionParams& params) : params_(params) {}

  ~OpenVINOEngine() {}

  void BuildEngine();
  void BindingInput(const std::string& input_name,
                    ov::element::Type ov_type,
                    const std::vector<size_t> data_shape,
                    T* data,
                    int64_t data_num);
  ov::Model* model() { return model_.get(); }
  ov::CompiledModel compiled_model() { return complied_model_; }
  ov::InferRequest infer_request() { return infer_request_; }
  ov::Shape GetOuputShapeByName(const std::string& name);
  phi::DataType GetOuputTypeByName(const std::string& name);
  void CopyOuputDataByName(const std::string& output_name, void* pd_data);

 private:
  //
  // Construction parameters.
  //
  ConstructionParams params_;

  ov::Core core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel complied_model_;
  ov::InferRequest infer_request_;
  TensorName2IndexMap input_map_;
  TensorName2IndexMap output_map_;
  std::mutex mutex_;

 public:
  thread_local static int predictor_id_per_thread;

 private:
  std::string paddle_frontend_name_{"paddle"};
};  // class OpenVINOEngine

class OVEngineManager {
 public:
  OVEngineManager() {}

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.empty();
  }

  bool Has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engines_.count(name) == 0) return false;
    return engines_.at(name).get() != nullptr;
  }

  OpenVINOEngine* Get(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engines_.at(name).get();
  }

  OpenVINOEngine* Create(const std::string& name,
                         const OpenVINOEngine::ConstructionParams& params) {
    auto engine = std::make_unique<OpenVINOEngine>(params);
    std::lock_guard<std::mutex> lock(mutex_);
    engines_[name].reset(engine.release());
    return engines_[name].get();
  }

  void DeleteAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& item : engines_) {
      item.second.reset(nullptr);
    }
    engines_.clear();
  }

  void DeleteKey(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = engines_.find(key);
    if (iter != engines_.end()) {
      iter->second.reset(nullptr);
      engines_.erase(iter);
    }
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<OpenVINOEngine>> engines_;
};

}  // namespace openvino
}  // namespace inference
}  // namespace paddle
