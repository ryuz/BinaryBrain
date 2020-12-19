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
    
    def send_command(self, command, send_to="all"):
        self.model.send_command(command=command, send_to=send_to)
    
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

    def send_command(self, command, send_to="all"):
        for model in self.model_list:
            model.send_command(command=command, send_to=send_to)

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


class RealToBinary(Model):
    """RealToBinary class
    """
    
    def __init__(self, bin_dtype=core.TYPE_FP32,
                        frame_modulation_size=1, depth_modulation_size=1, value_generator=None,
                        framewise=False, input_range_lo=0.0, input_range_hi=1.0):
        super(RealToBinary, self).__init__()
        
        try:
            creator = {
                core.TYPE_FP32: core.RealToBinary_fp32.create,
                core.TYPE_BIT:  core.RealToBinary_bit.create,
            }[bin_dtype]
        except:
            raise TypeError("unsupported")

        self.model = creator(frame_modulation_size, depth_modulation_size,
                            value_generator, framewise, input_range_lo, input_range_hi)

#        if bin_dtype==core.TYPE_FP32:
#            self.model = core.RealToBinary_fp32.create(
#                                frame_modulation_size=frame_modulation_size,
#                                depth_modulation_size=depth_modulation_size,
#                                value_generator=value_generator,
#                                framewise=framewise,
#                                input_range_lo=input_range_lo,
#                                input_range_hi=input_range_hi)
#        elif bin_dtype==core.TYPE_BIT:
#            self.model = core.RealToBinary_bit.create(
#                                frame_modulation_size=frame_modulation_size,
#                                depth_modulation_size=depth_modulation_size,
#                                value_generator=value_generator,
#                                framewise=framewise,
#                                input_range_lo=input_range_lo,
#                                input_range_hi=input_range_hi)
#        else:
#            raise ValueError("parameter error")
        

class BinaryToReal(Model):
    """BinaryToReal class
    """
    
    def __init__(self, *, bin_dtype=core.TYPE_FP32, frame_modulation_size=1, output_shape=[]):
        super(BinaryToReal, self).__init__()
        
        if bin_dtype==core.TYPE_FP32:
            self.model = core.BinaryToReal_fp32.create(frame_modulation_size=frame_modulation_size, output_shape=output_shape)
        elif bin_dtype==core.TYPE_BIT:
            self.model = core.BinaryToRealBit_fp32.create(frame_modulation_size=frame_modulation_size, output_shape=output_shape)
        else:
            raise ValueError("parameter error")



class DenseAffine(Model):
    """DenseAffine class
    """
    
    def __init__(self, output_shape, initialize_std=0.01, initializer="he", seed=1):
        super(DenseAffine, self).__init__()
        self.model = core.DenseAffine.create(output_shape=output_shape, initialize_std=initialize_std, initializer=initializer, seed=seed)


class DifferentiableLut(Model):
    """DifferentiableLut class
       微分可能LUTモデル
       StocasticLUT + BatchNormalization + Binarize(HardTanh)
    """

    def __init__(self, output_shape, N=6, bin_dtype=core.TYPE_FP32, real_dtype=core.TYPE_FP32,
                    connection='random', binarize=True, batch_norm=True, momentum=0.0, gamma= 0.3, beta=0.5, seed=1):
        super(DifferentiableLut, self).__init__()
        
        if not binarize and not batch_norm:
            # StochasticLut 演算のみ
            try:
                creator = {
                    core.TYPE_FP32: {
                        6: core.StochasticLut6_fp32.create,
                        5: core.StochasticLut5_fp32.create,
                        4: core.StochasticLut4_fp32.create,
                        2: core.StochasticLut2_fp32.create,
                    },
                    core.TYPE_BIT: {
                        6: core.StochasticLut6_bit.create,
                        5: core.StochasticLut5_bit.create,
                        4: core.StochasticLut4_bit.create,
                        2: core.StochasticLut2_bit.create,
                    },
                }[bin_dtype][N]
            except:
                raise TypeError("unsupported")

                self.model = creator(output_shape, connection, seed)
        
        elif binarize and batch_norm:
            # 条件が揃えば BatchNorm と 二値化を一括演算
            try:
                creator = {
                    core.TYPE_FP32: {
                        6: core.DifferentiableLut6_fp32.create,
                        5: core.DifferentiableLut5_fp32.create,
                        4: core.DifferentiableLut4_fp32.create,
                        2: core.DifferentiableLut2_fp32.create,
                    },
                    core.TYPE_BIT: {
                        6: core.DifferentiableLut6_bit.create,
                        5: core.DifferentiableLut5_bit.create,
                        4: core.DifferentiableLut4_bit.create,
                        2: core.DifferentiableLut2_bit.create,
                    },                    
                }[bin_dtype][N]
            except:
                raise TypeError("unsupported")
            
            self.model = creator(output_shape, batch_norm, connection, momentum, gamma, beta, seed)
        
        else:
            # 個別演算
            try:
                creator = {
                    core.TYPE_FP32: {
                        6: core.DifferentiableLutDiscrete6_fp32.create,
                        5: core.DifferentiableLutDiscrete5_fp32.create,
                        4: core.DifferentiableLutDiscrete4_fp32.create,
                        2: core.DifferentiableLutDiscrete2_fp32.create,
                    },
                    core.TYPE_BIT: {
                        6: core.DifferentiableLutDiscrete6_bit.create,
                        5: core.DifferentiableLutDiscrete5_bit.create,
                        4: core.DifferentiableLutDiscrete4_bit.create,
                        2: core.DifferentiableLutDiscrete2_bit.create,
                    },                    
                }[bin_dtype][N]
            except:
                raise TypeError("unsupported")
            
            self.model = creator(output_shape, batch_norm, connection, momentum, gamma, beta, seed)


class ConvolutionIm2Col(Model):
    """ConvolutionIm2Col class
    """
    def __init__(self, filter_size=(1, 1), stride=(1, 1),
                        padding='valid', border_mode=core.BB_BORDER_REFLECT_101, border_value=0.0,
                        fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):
        super(ConvolutionIm2Col, self).__init__()

        if fw_dtype==core.TYPE_FP32 and bw_dtype==core.TYPE_FP32:
            self.model = core.ConvolutionIm2Col.creat(filter_h_size=filter_size[0], filter_w_size=filter_size[1],
                                    y_stride=stride[0], x_stride=stride[1], padding=padding, border_mode=border_mode)
        elif fw_dtype==core.TYPE_BIT and bw_dtype==core.TYPE_FP32:
            self.model = core.ConvolutionIm2ColBit.creat(filter_h_size=filter_size[0], filter_w_size=filter_size[1],
                                    y_stride=stride[0], x_stride=stride[1], padding=padding, border_mode=border_mode)
        else:
            raise TypeError("unsupported")


class ConvolutionCol2Im(Model):
    """ConvolutionCol2Im class
    """
    def __init__(self, output_size=(1, 1), fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):
        super(ConvolutionCol2Im, self).__init__()

        if fw_dtype==core.TYPE_FP32 and bw_dtype==core.TYPE_FP32:
            self.model = core.ConvolutionCol2Im.creat(h_size=output_size[0], w_size=output_size[1])
        elif fw_dtype==core.TYPE_BIT and bw_dtype==core.TYPE_FP32:
            self.model = core.ConvolutionCol2ImBit.creat(h_size=output_size[0], w_size=output_size[1])
        else:
            raise TypeError("unsupported")

class Convolution(Sequential):
    """LoweringConvolution class
    """
    def __init__(self, sub_layer, *, filter_size=(1, 1), stride=(1, 1),
                        padding='valid', border_mode=core.BB_BORDER_REFLECT_101, border_value=0.0,
                        fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):
        super(Convolution, self).__init__()
        if fw_dtype==core.TYPE_FP32 and bw_dtype==core.TYPE_FP32:
            self.im2col = core.ConvolutionIm2Col(filter_size=filter_size, stride=stride,
                                padding=padding, border_mode=border_mode, border_value=border_value,
                                fw_dtype=fw_dtype, bw_dtype=bw_dtype)
        else:
            self.im2col = core.ConvolutionIm2ColBit(filter_size=filter_size, stride=stride,
                                padding=padding, border_mode=border_mode, border_value=border_value,
                                fw_dtype=fw_dtype, bw_dtype=bw_dtype)
        self.sub_layer = sub_layer
        self.col2im    = None  # 後で決定
        self.fw_dtype = fw_dtype











class ReLU(Model):
    """ReLU class
    """

    def __init__(self, bin_dtype=core.TYPE_FP32, real_dtype=core.TYPE_FP32):
        if bin_dtype==core.TYPE_FP32 and real_dtype==core.TYPE_FP32:
            self.model = core.ReLU.create()
        elif bin_dtype==core.TYPE_BIT and real_dtype==core.TYPE_FP32:
            self.model = core.ReLUBit.create()
        else:
            raise ValueError("parameter error")




