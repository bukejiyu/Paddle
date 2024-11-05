/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/openvino/engine.h"
#include <glog/logging.h>
#include <string>

namespace paddle::inference::openvino {
bool OpenVINOEngine::IsModelStatic() {
  bool isStatic = true;
  for (auto&& it : compiled_model().inputs()) {
    if (isStatic && it.get_partial_shape().is_dynamic()) {
      isStatic = !isStatic;
    }
  }
  return isStatic;
}

template <typename T>
void OpenVINOEngine::BindingInput(const std::string& input_name,
                                  ov::element::Type ov_type,
                                  const std::vector<size_t> data_shape,
                                  T* data,
                                  int64_t data_num) {
  auto model_input = complied_model_.input(input_name);
  PADDLE_ENFORCE_EQ(model_input.get_element_type() == ov_type,
                    true,
                    common::errors::PreconditionNotMet(
                        "runtime input element type is not same with model"));

  if (IsModelStatic()) {
    PADDLE_ENFORCE_EQ(
        model_input.get_partial_shape().compatible(
            ov::PartialShape(data_shape)),
        true,
        common::errors::PreconditionNotMet(
            "model is static but runtime input shape is not same with "
            "model!"));
  }

  try {
    auto requestTensor = infer_request_.get_tensor(input_name);
    requestTensor.set_shape(data_shape);
    auto input_shape = requestTensor.get_shape();
    std::memcpy(
        requestTensor.data(), static_cast<void*>(data), data_num * sizeof(T));
    infer_request_.set_tensor(input_name, requestTensor);
  } catch (const std::exception& exp) {
    LOG(ERROR) << exp.what();
  }
}

ov::Shape OpenVINOEngine::GetOuputShapeByName(const std::string& output_name) {
  return infer_request_.get_tensor(output_name).get_shape();
}

phi::DataType OpenVINOEngine::GetOuputTypeByName(
    const std::string& output_name) {
  return OVType2PhiType(
      infer_request_.get_tensor(output_name).get_element_type());
}

void OpenVINOEngine::CopyOuputDataByName(const std::string& output_name,
                                         void* pd_data) {
  auto ov_tensor = infer_request_.get_tensor(output_name);
  std::memcpy(pd_data, ov_tensor.data(), ov_tensor.get_byte_size());
}

void OpenVINOEngine::Execute() {
  try {
    infer_request_.infer();
  } catch (const std::exception& exp) {
    LOG(ERROR) << exp.what();
  }
}

void OpenVINOEngine::BuildEngine() {
  std::string model_path = params_.model_program_path;
  std::string param_path = params_.model_params_path;

  ov::frontend::FrontEndManager fem;
  ov::frontend::FrontEnd::Ptr frontEnd;
  ov::frontend::InputModel::Ptr inputModel;
  bool flag_success{true};

  frontEnd = fem.load_by_framework(paddle_frontend_name_);
  PADDLE_ENFORCE_NOT_NULL(frontEnd,
                          common::errors::Unavailable(
                              "Cannot find the front-end paddle in OpenVINO."));
  inputModel = frontEnd->load(model_path);
  std::shared_ptr<ov::Model> model;
  model = frontEnd->convert_partially(inputModel);
  for (auto& node : model->get_ordered_ops()) {
    if (node->description() == "FrameworkNode") {
      flag_success = false;
      VLOG(3) << "Can't convert op:" << node->get_friendly_name()
              << " to openvino op.";
    }
  }
  PADDLE_ENFORCE_EQ(flag_success,
                    true,
                    common::errors::Unavailable(
                        "Can't convert paddle model to openvino model. Please "
                        "check if the model is supported by openvino."));

  core_ = ov::Core();
  core_.set_property(
      ov::inference_num_threads(params_.cpu_math_library_num_threads));
  core_.set_property(ov::cache_dir(params_.model_opt_cache_dir));
  core_set_property(ov::inference_precision(params_.inference_precision));
  model_ =
      core.read_model(params_.model_program_path, params_.model_params_path);
  compiled_model_ = core.compile_model(model_, "CPU");
}

}  // namespace paddle::inference::openvino

template void paddle::inference::openvino::OpenVINOEngine::BindingInput<bool>(
    const std::string&, ov::element::Type, std::vector<size_t>, bool*, int64_t);
template void paddle::inference::openvino::OpenVINOEngine::BindingInput<float>(
    const std::string&,
    ov::element::Type,
    std::vector<size_t>,
    float*,
    int64_t);
template void paddle::inference::openvino::OpenVINOEngine::BindingInput<int>(
    const std::string&, ov::element::Type, std::vector<size_t>, int*, int64_t);
template void paddle::inference::openvino::OpenVINOEngine::BindingInput<double>(
    const std::string&,
    ov::element::Type,
    std::vector<size_t>,
    double*,
    int64_t);
