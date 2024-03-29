# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Optimizer(bb.Object):
    """Optimizer の基本クラス
    """
    
    def __init__(self, core_optimizer=None):
        super(Optimizer, self).__init__(core_object=core_optimizer)
    
    def set_variables(self, params, grads):
        """変数設定

        Args:
            params (Variables): 学習対象のパラメータ変数
            grads (Variables): paramsに対応する勾配変数
        """

        self.get_core().set_variables(params.get_core(), grads.get_core())
    
    def update(self):
        """パラメータ更新＆勾配ゼロクリア

            set_variablesで設定された勾配変数に基づいた学習をset_variablesで
            設定されたパラメータ変数に適用して、勾配をゼロクリアする
        """

        return self.get_core().update()

    def zero_grad(self):
        """勾配のゼロクリア

            set_variablesで設定された勾配変数をゼロクリアする
        """

        return self.get_core().zero_grad()
    
    def step(self):
        """パラメータ更新

            set_variablesで設定された勾配変数に基づいた学習をset_variablesで
            設定されたパラメータ変数に適用する
        """

        return self.get_core().step()

    def set_learning_rate(self, learning_rate):
        """学習率設定
        """
        self.get_core().set_learning_rate(learning_rate)


class OptimizerSgd(Optimizer):
    """SGD 最適化クラス

    Args:
        learning_rate (float): 学習率
    """
    def __init__(self, learning_rate=0.001, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerSgd', [dtype]).create(learning_rate=learning_rate)
        super(OptimizerSgd, self).__init__(core_optimizer=core_optimizer)


class OptimizerAdaGrad(Optimizer):
    """AdaGrad 最適化クラス

    Args:
        learning_rate (float): 学習率
    """

    def __init__(self, learning_rate=0.01, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerAdaGrad', [dtype]).create(learning_rate=learning_rate)
        super(OptimizerAdaGrad, self).__init__(core_optimizer=core_optimizer)


class OptimizerAdam(Optimizer):
    """Adam 最適化クラス

    Args:
        learning_rate (float): 学習率
        beta1 (float): beta1
        beta2 (float): beta2
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerAdam', [dtype]).create(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        super(OptimizerAdam, self).__init__(core_optimizer=core_optimizer)


