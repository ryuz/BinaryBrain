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
        """パラメータ更新

            set_variablesで設定された勾配変数に基づいた学習をset_variablesで
            設定されたパラメータ変数に適用する
        """

        return self.get_core().update()


class OptimizerSgd(Optimizer):
    """SGD 最適化クラス

    Args:
        learning_rate (float): 学習率
    """
    def __init__(self, learning_rate=0.01, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerSgd', [dtype]).create()
        super(OptimizerSgd, self).__init__(core_optimizer=core_optimizer)


class OptimizerAdaGrad(Optimizer):
    """AdaGrad 最適化クラス

    Args:
        learning_rate (float): 学習率
    """

    def __init__(self, learning_rate=0.01, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerAdaGrad', [dtype]).create()
        super(OptimizerAdaGrad, self).__init__(core_optimizer=core_optimizer)


class OptimizerAdam(Optimizer):
    """Adam 最適化クラス

    Args:
        learning_rate (float): 学習率
        beta1 (float): beta1
        beta2 (float): beta2
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, dtype=bb.DType.FP32):
        core_optimizer = bb.search_core_object('OptimizerAdam', [dtype]).create()
        super(OptimizerAdam, self).__init__(core_optimizer=core_optimizer)


