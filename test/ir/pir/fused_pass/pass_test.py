#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
from paddle import pir


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = paddle.static.Program()
        self.feeds = None
        self.fetch_list = None
        self.fused_op_type = None
        self.pass_list = []
        self.pir_program = None
        self.fused_ops = []

        np.random.seed(123)
        random.seed(124)

    def run_pir_pass(self):
        if not isinstance(self.pass_list, list):
            self.pass_list = [self.pass_list]

        pm = pir.PassManager()
        for pass_name in self.pass_list:
            pm.add_pass(pass_name)

        pm.run(self.pir_program)

    def check_fused_ops(self):
        '''
        Check whether the fused ops are correct.
        '''
        if self.fused_op_type is None or len(self.fused_ops) < 0:
            return
        op_names = [op.name() for op in self.pir_program.global_block().ops]
        if self.fused_op_type:
            self.assertTrue(self.fused_op_type in op_names, "error!")
        for fused_op in self.fused_ops:
            self.assertTrue(fused_op not in op_names, "error!")

    def check_pass_correct(self, place, need_translate_to_pir=False, atol=1e-5):
        '''
        1.Check whether the pass is effective
        2.[todo]Check the accuracy before and after running the pass
        '''
        executor = paddle.static.Executor(place)
        # 转成新ir的 program

        self.assertTrue(
            need_translate_to_pir is False and self.pir_program is not None,
            "error!",
        )
        if need_translate_to_pir and self.pir_program is None:
            self.pir_program = pir.translate_to_pir(self.main_program.desc)

        self.run_pir_pass()
        self.check_fused_ops()
