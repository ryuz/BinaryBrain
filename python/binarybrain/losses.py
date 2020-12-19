# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Loss():
    """Loss class
    """
    
    def __init__(self):
        self.loss = None
        
    def clear(self):
        self.loss.clear()
    
    def get(self):
        return self.loss.get_loss()

    def calculate(self, y_buf, t_buf, mini_batch_size=None):
        if mini_batch_size is None:
            mini_batch_size = t_buf.get_frame_size()
        return bb.FrameBuffer.from_core(self.loss.calculate_loss(y_buf.get_core(), t_buf.get_core(), mini_batch_size))



class LossSoftmaxCrossEntropy(Loss):
    """LossSoftmaxCrossEntropy class
    """
    
    def __init__(self):
        super(LossSoftmaxCrossEntropy, self).__init__()
        self.loss = core.LossSoftmaxCrossEntropy.create()


class LossMeanSquaredError(Loss):
    """LossSoftmaxCrossEntropy class
    """
    
    def __init__(self):
        super(LossMeanSquaredError, self).__init__()
        self.loss = core.LossMeanSquaredError.create()

