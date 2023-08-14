# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from functools import partial

import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


class TestLayernormShiftPartitionPass(PassAutoScanTest):
    #
    #     |           |                            |            |
    # other_op1     other_op2                  other_op1    other_op2
    #     |           |              fuse           \          /
    #     |------elementwise_add      ->    preln_layernorm_shift_partition
    #             |          |                        |      |
    #        other_op4  layernorm_shift_partition  other_op4  other_op3
    #                        |
    #                   other_op3

    def sample_predictor_configs(self, program_config):
        # trt dynamic_shape
        # config = self.create_trt_inference_config()
        # config.enable_tensorrt_engine(
        #     max_batch_size=1,
        #     workspace_size=102400,
        #     min_subgraph_size=0,
        #     precision_mode=paddle_infer.PrecisionType.Float32,
        #     use_static=False,
        #     use_calib_mode=False,
        # )
        # config.set_trt_dynamic_shape_info(
        #     {
        #         "input_data_x": [1, 9, 96],
        #         "input_data_y": [1, 9, 96],
        #     },
        #     {
        #         "input_data_x": [4, 3136, 768],
        #         "input_data_y": [4, 3136, 768],
        #     },
        #     {
        #         "input_data_x": [1, 784, 384],
        #         "input_data_y": [1, 784, 384],
        #     },
        # )
        # yield config, ['preln_layernorm_shift_partition'], (1e-5, 1e-5)

        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_use_gpu(100, 0, paddle_infer.PrecisionType.Half)
        config.enable_low_precision_io(True)
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=1 << 32,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input_data_x": [1, 3136, 96],
                "input_data_y": [1, 3136, 96],
            },
            {
                "input_data_x": [4, 3136, 96],
                "input_data_y": [4, 3136, 96],
            },
            {
                "input_data_x": [1, 3136, 96],
                "input_data_y": [1, 3136, 96],
            },
        )
        yield config, ['preln_layernorm_shift_partition'], (1e-2, 1e-2)

    def sample_program_config(self, draw):
        axis = [0, 1, 3, 2, 4, 5]
        epsilon = 0.000009999999747378752
        # begin_norm_axis has to be 2
        begin_norm_axis = 2
        batch_size = 4

        window_size = 7
        move_shape = 8
        dim = 96

        def generate_input(attrs):
            return np.random.random(
                [attrs[1]["batch_size"], *attrs[1]["input_dim"]]
            ).astype(np.float16)

        def generate_weight(attrs):
            return np.random.random(attrs[1]['input_dim'][-1]).astype(
                np.float32
            )

        attrs = [
            {
                'begin_norm_axis': begin_norm_axis,
                'epsilon': epsilon,
            },
            {
                'batch_size': batch_size,
                'input_dim': [(window_size * move_shape) ** 2, dim],
            },
            {
                'axis': axis,
                'input_resolution': window_size * move_shape,
                'move_shape': move_shape,
                'window_size': window_size,
            },
        ]

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["input_data_x"], "Y": ["input_data_y"]},
            outputs={"Out": ["ele_out"]},
            attrs={"axis": -1},
        )
        layer_norm_op = OpConfig(
            type="layer_norm",
            inputs={
                "X": ["ele_out"],
                "Bias": ["layer_norm_bias"],
                "Scale": ["layer_norm_scale"],
            },
            outputs={
                "Y": ["layer_norm_output1"],
                "Mean": ["layer_norm_output2"],
                "Variance": ["layer_norm_output3"],
            },
            attrs={
                "begin_norm_axis": attrs[0]["begin_norm_axis"],
                "epsilon": attrs[0]["epsilon"],
            },
        )
        reshape_op2 = OpConfig(
            type="reshape2",
            inputs={
                "X": ["layer_norm_output1"],
            },
            outputs={
                "Out": ["reshape_output2"],
                "XShape": ["reshape_output2_xshape"],
            },
            attrs={
                'shape': [
                    -1,
                    attrs[2]["input_resolution"],
                    attrs[2]["input_resolution"],
                    attrs[1]["input_dim"][-1],
                ]
            },
        )
        reshape_op3 = OpConfig(
            type="reshape2",
            inputs={
                "X": ["reshape_output2"],
            },
            outputs={
                "Out": ["reshape_output3"],
                "XShape": ["reshape_output3_xshape"],
            },
            attrs={
                'shape': [
                    -1,
                    attrs[2]["move_shape"],
                    attrs[2]["window_size"],
                    attrs[2]["move_shape"],
                    attrs[2]["window_size"],
                    attrs[1]["input_dim"][-1],
                ]
            },
        )
        transpose_op4 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["reshape_output3"],
            },
            outputs={"Out": ["transpose_output4"]},
            attrs={"axis": attrs[2]['axis']},
        )
        reshape_op5 = OpConfig(
            type="reshape2",
            inputs={
                "X": ["transpose_output4"],
            },
            outputs={
                "Out": ["reshape_output5"],
                "XShape": ["reshape_output5_xshape"],
            },
            attrs={
                'shape': [
                    -1,
                    attrs[2]["window_size"],
                    attrs[2]["window_size"],
                    attrs[1]["input_dim"][-1],
                ]
            },
        )
        reshape_op6 = OpConfig(
            type="reshape2",
            inputs={
                "X": ["reshape_output5"],
            },
            outputs={
                "Out": ["reshape_output6"],
                "XShape": ["reshape_output6_xshape"],
            },
            attrs={
                'shape': [
                    -1,
                    attrs[2]["window_size"] ** 2,
                    attrs[1]["input_dim"][-1],
                ]
            },
        )

        program_config = ProgramConfig(
            ops=[
                elementwise_add_op,
                layer_norm_op,
                reshape_op2,
                reshape_op3,
                transpose_op4,
                reshape_op5,
                reshape_op6,
            ],
            weights={
                "layer_norm_bias": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
                "layer_norm_scale": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
            },
            inputs={
                "input_data_x": TensorConfig(
                    data_gen=partial(generate_input, attrs)
                ),
                "input_data_y": TensorConfig(
                    data_gen=partial(generate_input, attrs)
                ),
            },
            outputs=["ele_out", "reshape_output6"],
        )

        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=50,
            passes=["preln_layernorm_x_fuse_pass"],
            max_duration=250,
            min_success_num=50,
        )


if __name__ == "__main__":
    unittest.main()
