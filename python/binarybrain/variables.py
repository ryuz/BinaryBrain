# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List


class Variables():
    """Variables class

       学習の為の Optimizer と実際の学習ターゲットの変数の橋渡しに利用されるクラス。
       内部的には各モデル内の重みや勾配を保有する Tensor をまとめて保持している。
    """
    
    def __init__(self):
        self.variables = core.Variables()

    @staticmethod
    def from_core(variables):
        new_variables = Variables()
        new_variables.variables = variables
        return new_variables
    
    def get_core(self):
        return self.variables
    
    def append(self, variables):
        """ 変数を追加

        Args:
            variables (Variables) : 追加する変数
        """
        self.variables.push_back(variables.get_core())
    
 