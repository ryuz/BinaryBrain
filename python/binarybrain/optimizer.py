# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Optimizer():
    """Optimizer class
    """
    
    def __init__(self):
        self.optimizer = None
    
    def set_variables(self, params, grads):
        self.optimizer.set_variables(params.get_core(), grads.get_core())
    
    def update(self):
        return self.optimizer.update()



class OptimizerAdam(Optimizer):
    """OptimizerAdam class
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        super(OptimizerAdam, self).__init__()
        self.optimizer = core.OptimizerAdam.create(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

