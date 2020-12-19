# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Model():
    """Model class
    """
    
    def __init__(self):
        self.model = None
    
    def set_input_shape(self, input_shape):
        return self.model.set_input_shape(input_shape)
    
    def get_parameters(self):
        return bb.Variables.from_core(self.model.get_parameters())

    def get_gradients(self):
        return bb.Variables.from_core(self.model.get_gradients())

    def forward(self, x_buf, train=True):
        return bb.FrameBuffer.from_core(self.model.forward(x_buf=x_buf.get_core(), train=train))

    def backward(self, dy_buf):
        return bb.FrameBuffer.from_core(self.model.backward(dy_buf.get_core()))


class Sequential(Model):
    """Sequential class
    """

    def __init__(self, model_list=[]):
        super(Sequential, self).__init__()
        self.model_list = model_list

    def append(self, model):
        self.model.append(model)

    def set_input_shape(self, shape):
        for model in self.model_list:
            shape = model.set_input_shape(shape)
        return shape

    def get_parameters(self):
        variables = bb.Variables()
        for model in self.model_list:
            variables.append(model.get_parameters())
        return variables

    def get_gradients(self):
        variables = bb.Variables()
        for model in self.model_list:
            variables.append(model.get_gradients())
        return variables
    
    def forward(self, x_buf, train=True):
        for model in self.model_list:
            x_buf = model.forward(x_buf, train)
        return x_buf

    def backward(self, dy_buf):
        for model in reversed(self.model_list):
            dy_buf = model.backward(dy_buf)
        return dy_buf


class DenseAffine(Model):
    """DenseAffine class
    """
    
    def __init__(self, output_shape, initialize_std=0.01, initializer="he", seed=1):
        super(DenseAffine, self).__init__()
        self.model = core.DenseAffine.create(output_shape=output_shape, initialize_std=initialize_std, initializer=initializer, seed=seed)


class ReLU(Model):
    """ReLU class
    """

    def __init__(self, bin_type=core.TYPE_FP32, real_type=core.TYPE_FP32):
        if bin_type==core.TYPE_FP32 and real_type==core.TYPE_FP32:
            self.model = core.ReLU.create()
        elif bin_type==core.TYPE_BIT and real_type==core.TYPE_FP32:
            self.model = core.ReLUBit.create()

