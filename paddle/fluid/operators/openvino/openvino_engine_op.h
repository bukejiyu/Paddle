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

#ifdef PADDLE_WITH_OPENVINO
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/errors.h"
// #include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/openvino/engine.h"
#include "paddle/fluid/inference/openvino/helper.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace inference {
namespace openvino {}  // namespace openvino
template <typename T>
struct Singleton;
}  // namespace inference
}  // namespace paddle

namespace paddle {

namespace operators {

using inference::Singleton;
using inference::openvino::OpenVINOEngine;

class OpenVINOEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> runtime_input_names_;
  std::string engine_key_ = "openvino";
  std::string model_opt_cache_dir_;
  std::string model_program_path_;
  std::string model_params_path_;
  std::string inference_precision_;
  int cpu_math_library_num_threads_;
  mutable OpenVINOEngine *ov_engine_{nullptr};

 public:
  OpenVINOEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    output_names_ = Outputs("Ys");
    engine_key_ = Attr<std::string>("engine_key");
    model_opt_cache_dir_ = Attr<std::string>("model_opt_cache_dir");
    model_program_path_ = Attr<std::string>("model_program_path");
    model_params_path_ = Attr<std::string>("model_params_path");
    inference_precision_ = Attr<std::string>("inference_precision");
    cpu_math_library_num_threads_ = Attr<std::string>("inference_num_threads");
    for (auto &x : input_names_) {
      runtime_input_names_.emplace_back(x);
    }
    bool has_engine =
        inference::Singleton<inference::openvino::OVEngineManager>::Global()
            .Has(engine_key_);
    if (has_engine) {
      ov_engine_ =
          inference::Singleton<inference::openvino::OVEngineManager>::Global()
              .Get(engine_key_);
    }
  }

 protected:
  void RunNativeImpl(const framework::Scope &scope,
                     const phi::Place &dev_place) const {
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto &current_scope = scope.NewScope();
    auto ctx = executor.Prepare(*program, block->ID());
    executor.RunPreparedContext(ctx.get(), &current_scope, false, true, true);
  }

  void RunOpenvino(const framework::Scope &scope,
                   const phi::Place &dev_place,
                   OpenVINOEngine *engine) const {
    for (auto x : runtime_input_names_) {
      auto &t = inference::analysis::GetFromScope<phi::DenseTensor>(scope, x);
      auto t_shape = common::vectorize<size_t>(t.dims());
      if (t_shape.empty()) {
        PADDLE_ENFORCE_EQ(
            t.numel(),
            1UL,
            common::errors::PreconditionNotMet(
                "This tensor must have one element, but got %ld.", t.numel()));
        t_shape.push_back(1);
      }
      if (t.dtype() == phi::DataType::FLOAT32) {
        engine->BindingInput(x,
                             inference::openvino::PhiType2OVType(t.dtype()),
                             t_shape,
                             t.data<float>(),
                             t.numel());
      } else if (t.dtype() == phi::DataType::INT32) {
        engine->BindingInput(x,
                             inference::openvino::PhiType2OVType(t.dtype()),
                             t_shape,
                             t.data<int32_t>(),
                             t.numel());
      } else if (t.dtype() == phi::DataType::FLOAT64) {
        engine->BindingInput(x,
                             inference::openvino::PhiType2OVType(t.dtype()),
                             t_shape,
                             t.data<double>(),
                             t.numel());
      } else if (t.dtype() == phi::DataType::BOOL) {
        engine->BindingInput(x,
                             inference::openvino::PhiType2OVType(t.dtype()),
                             t_shape,
                             t.data<bool>(),
                             t.numel());
      } else {
        PADDLE_THROW(
            common::errors::Fatal("The OV Engine OP only support "
                                  "float/int32_t/float64/bool input."));
      }
    }
    VLOG(1) << "start openvino execute ";
    engine->Execute();
    VLOG(1) << "end openvino execute!";
    for (size_t i = 0; i < Outputs("Ys").size(); i++) {
      std::vector<int> ddim;
      auto output_shape = engine->GetOuputShapeByName(output_names_[i]);
      auto ov_type = engine->GetOuputTypeByName(output_names_[i]);
      for (size_t j = 0; j < output_shape.size(); j++) {
        ddim.push_back(output_shape[j]);
      }
      auto y = Outputs("Ys")[i];
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(
          fluid_v,
          common::errors::NotFound(
              "Output variable %s is not found in Openvino subgraph.", y));
      auto *fluid_t = fluid_v->GetMutable<phi::DenseTensor>();
      fluid_t->Resize(common::make_ddim(ddim));
      engine->CopyOuputDataByName(output_names_[i],
                                  fluid_t->mutable_data(dev_place, ov_type));
    }
  }

  OpenVINOEngine *GetEngine(const framework::Scope &scope,
                            const phi::Place &dev_place) const {
    if (!ov_engine_) {
      OpenVINOEngine::ConstructionParams params;
      params.model_program_path = model_program_path_;
      params.model_params_path = model_params_path_;
      params.model_opt_cache_dir = model_opt_cache_dir_;
      params.infernce_precision = inference_precision_;
      params.cpu_math_library_num_threads = cpu_math_library_num_threads_;
      ov_engine_ =
          inference::Singleton<inference::openvino::OVEngineManager>::Global()
              .Create(engine_key_, params);
      ov_engine_->BuildEngine();
    }
    PADDLE_ENFORCE_NOT_NULL(
        ov_engine_,
        common::errors::Fatal(
            "The pointer to openvino engine should not be null."));
    return ov_engine_;
  }
  void RunImpl(const framework::Scope &scope,
               const phi::Place &dev_place) const override {
    auto *ov_engine = GetEngine(scope, dev_place);
    RunOpenvino(scope, dev_place, ov_engine);
  }
};

}  // namespace operators
}  // namespace paddle

#endif
