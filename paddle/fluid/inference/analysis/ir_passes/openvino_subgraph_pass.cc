// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/analysis/ir_passes/openvino_subgraph_pass.h"
#include <fcntl.h>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/utils/io_utils.h"

#include "paddle/fluid/inference/openvino/engine.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

#ifdef PADDLE_WITH_OPENVINO
#include "oneapi/tbb.h"
#include "openvino/frontend/manager.hpp"
#include "openvino/openvino.hpp"
#endif

namespace paddle {
namespace inference {
namespace analysis {
namespace {}  // namespace

using framework::ir::Node;

void analysis::OpenVINOSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("openvino_subgraph_pass", graph);

  VLOG(3) << "Running openvino_subgraph_pass.";
  if (graph->IsMainGraph()) {
    VLOG(3)
        << "The ID of block running openvino_subgraph_pass is: 0(main_graph)";
  } else {
    VLOG(3) << "The ID of block running openvino_subgraph_pass is: "
            << graph->GetBlockId();
  }

  std::string model_program_path = Get<std::string>("model_program_path");
  std::string model_params_path = Get<std::string>("model_params_path");
  std::string model_opt_cache_dir = Get<std::string>("model_opt_cache_dir");
  int cpu_math_library_num_threads = Get<int>("cpu_math_library_num_threads");
  bool use_static_engine = Get<bool>("use_static_engine");
  openvino::OpenVINOEngine::ConstructionParams params;
  params.model_program_path = model_program_path;
  params.model_params_path = model_params_path;
  params.model_opt_cache_dir = model_opt_cache_dir;
  params.cpu_math_library_num_threads = cpu_math_library_num_threads;
  auto inference_precision =
      static_cast<phi::DataType>(Get<int>("inference_precision"));
  auto precision = "fp32";
  if (inference_precision == phi::DataType::BFLOAT16) {
    precision = "bf16";
  }
  params.inference_precision = precision;

  std::string engine_key{"openvino"};
  openvino::OpenVINOEngine *ov_engine =
      inference::Singleton<inference::openvino::OVEngineManager>::Global()
          .Create(engine_key, params);
  ov_engine->BuildEngine();

  std::unordered_set<const Node *> nodes2remove;
  std::unordered_map<std::string, const Node *> white_nodes;
  std::vector<Node *> input_nodes;
  std::vector<Node *> output_nodes;
  for (auto *node : graph->Nodes()) {
    if (!(node->IsOp())) {
      continue;
    }
    if (node->Op()->Type() != "feed" && node->Op()->Type() != "fetch") {
      nodes2remove.insert(node);
    }
    if (node->Op()->Type() == "feed") {
      for (auto *var : node->outputs) {
        white_nodes.insert(std::make_pair(var->Var()->Name(), var));
        input_nodes.push_back(var);
      }
    }
    if (node->Op()->Type() == "fetch") {
      for (auto *var : node->inputs) {
        white_nodes.insert(std::make_pair(var->Var()->Name(), var));
        output_nodes.push_back(var);
      }
    }
  }
  for (auto *node : graph->Nodes()) {
    if (node->IsVar() && !white_nodes.count(node->Name())) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);

  framework::OpDesc openvino_desc;
  openvino_desc.SetType("openvino_engine");
  openvino_desc.SetAttr("model_opt_cache_dir", model_opt_cache_dir);
  openvino_desc.SetAttr("model_program_path", model_program_path);
  openvino_desc.SetAttr("model_params_path", model_params_path);
  openvino_desc.SetAttr("engine_key", engine_key);
  openvino_desc.SetAttr("inference_num_threads", cpu_math_library_num_threads);
  openvino_desc.SetAttr("inference_precision", precision);
  auto *openvino_node = graph->CreateOpNode(&openvino_desc);
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (auto *i : input_nodes) {
    i->outputs.push_back(openvino_node);
    input_names.push_back(i->Name());
  }
  for (auto *o : output_nodes) {
    o->inputs.push_back(openvino_node);
    output_names.push_back(o->Name());
  }
  openvino_node->inputs = std::move(input_nodes);
  openvino_node->outputs = std::move(output_nodes);

  auto *op_desc = openvino_node->Op();
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));
  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(openvino_subgraph_pass,
              paddle::inference::analysis::OpenVINOSubgraphPass);
