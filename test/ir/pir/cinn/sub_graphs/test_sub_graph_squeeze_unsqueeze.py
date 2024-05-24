# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from base import *  # noqa: F403

from paddle.static import InputSpec


class SqueezeCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,
    ):
        return var_0.sin().squeeze([0, 2])


class UnsqueezeCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,
    ):
        return var_0.sin().unsqueeze([0, 2])


class TestSplitSqueeze(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[1, 12, 1, 64], dtype=paddle.float32),)
        self.net = SqueezeCase()
        self.atol = 1e-8


class TestSplitUnsqueeze(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[1, 12, 1, 64], dtype=paddle.float32),)
        self.net = UnsqueezeCase()
        self.atol = 1e-8


if __name__ == '__main__':
    unittest.main()
