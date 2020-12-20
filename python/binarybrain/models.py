# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Model():
    """Model class
       ネットワーク各層の演算モデルの基底クラス
       C++で作成されたcoreモデルをラッピングするための基本機能を提供する
    """
    
    def __init__(self, core_model=None):
        self.core_model = core_model
    
    def get_core_model(self):
        return self.core_model
    
    def send_command(self, command, send_to="all"):
        self.get_core_model().send_command(command=command, send_to=send_to)
    
    def set_input_shape(self, input_shape):
        return self.get_core_model().set_input_shape(input_shape)
    
    def get_parameters(self):
        return bb.Variables.from_core(self.get_core_model().get_parameters())
    
    def get_gradients(self):
        return bb.Variables.from_core(self.get_core_model().get_gradients())
    
    def forward(self, x_buf, train=True):
        return bb.FrameBuffer.from_core(self.get_core_model().forward(x_buf=x_buf.get_core(), train=train))
    
    def backward(self, dy_buf):
        return bb.FrameBuffer.from_core(self.get_core_model().backward(dy_buf.get_core()))


class Sequential(Model):
    """Sequential class
       複数レイヤーを直列に接続してグルーピングするクラス
    """
    
    def __init__(self, model_list=[]):
        super(Sequential, self).__init__()
        self.model_list = model_list
    
    def get_core_model(self):
        # C++のコアの同機能に渡してしまうと Python からの扱いが不便になるので普段はListで管理して必要な時のみ変換する       
        core_model = core.Sequential.create()
        for model in self.model_list:
            core_model.add(model.get_core_model())
        return core_model
    
    def set_model_list(self, model_list):
        self.model_list = model_list
    
    def get_model_list(self, model_list):
        return self.model_list
    
    def append(self, model):
        self.model_list.append(model)
    
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
        実数値をバイナリ値に変換する。
        バイナリ変調機能も有しており、フレーム方向に変調した場合フレーム数(=ミニバッチサイズ)が
        増える。
        またここでビットパッキングが可能であり、32フレームのbitをint32に詰め込みメモリ節約可能である
    """
    
    def __init__(self, bin_dtype=core.TYPE_FP32,
                        frame_modulation_size=1, depth_modulation_size=1, value_generator=None,
                        framewise=False, input_range_lo=0.0, input_range_hi=1.0):
        try:
            core_creator = {
                core.TYPE_FP32: core.RealToBinary_fp32.create,
                core.TYPE_BIT:  core.RealToBinary_bit.create,
            }[bin_dtype]
        except:
            raise TypeError("unsupported")

        core_model = core_creator(frame_modulation_size, depth_modulation_size,
                            value_generator, framewise, input_range_lo, input_range_hi)
        super(RealToBinary, self).__init__(core_model=core_model)

        
class BinaryToReal(Model):
    """BinaryToReal class
        バイナリ値を実数値に戻す。その際にフレーム方向に変調されたデータを積算して
        元に戻すことが可能である
    """
    
    def __init__(self, *, bin_dtype=core.TYPE_FP32, frame_modulation_size=1, output_shape=[]):
        try:
            core_creator = {
                core.TYPE_FP32: core.BinaryToReal_fp32.create,
                core.TYPE_BIT:  core.BinaryToReal_bit.create,
            }[bin_dtype]
        except:
            raise TypeError("unsupported")

        core_model = core_creator(frame_modulation_size=frame_modulation_size, output_shape=output_shape)

        super(BinaryToReal, self).__init__(core_model=core_model)


class DenseAffine(Model):
    """DenseAffine class
       普通のDenseAffine
    """
    
    def __init__(self, output_shape, initialize_std=0.01, initializer="he", seed=1):
        core_creator = core.DenseAffine.create
        
        core_model = core_creator(output_shape=output_shape, initialize_std=initialize_std, initializer=initializer, seed=seed)

        super(DenseAffine, self).__init__(core_model=core_model)


class DifferentiableLut(Model):
    """DifferentiableLut class
       微分可能LUTモデル
       StocasticLUT + BatchNormalization + Binarize(HardTanh)
    """

    def __init__(self, output_shape, N=6, bin_dtype=core.TYPE_FP32, real_dtype=core.TYPE_FP32,
                    connection='random', binarize=True, batch_norm=True, momentum=0.0, gamma= 0.3, beta=0.5, seed=1):
        
        # 設定に応じて機能をパッキングしたモデルが使える場合は自動選択する
        if not binarize and not batch_norm:
            # StochasticLut 演算のみ
            try:
                core_creator = {
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

            core_model  = core_creator(output_shape, connection, seed)
        
        elif binarize and batch_norm:
            # 条件が揃えば BatchNorm と 二値化を一括演算
            try:
                core_creator = {
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
            
            core_model = core_creator(output_shape, batch_norm, connection, momentum, gamma, beta, seed)
        
        else:
            # 個別演算
            try:
                core_creator = {
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
            
            core_model = core_creator(output_shape, batch_norm, connection, momentum, gamma, beta, seed)

        super(DifferentiableLut, self).__init__(core_model=core_model)

    def W(self):
        return bb.Tensor.from_core(self.get_core_model().W())
    
    def dW(self):
        return bb.Tensor.from_core(self.get_core_model().dW())

    
class ConvolutionIm2Col(Model):
    """ConvolutionIm2Col class
       畳み込みの lowering における im2col 層
    """
    def __init__(self, filter_size=(1, 1), stride=(1, 1),
                        padding='valid', border_mode=core.BB_BORDER_REFLECT_101, border_value=0.0,
                        fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):

        try:
            core_creator = {
                core.TYPE_FP32: core.ConvolutionIm2Col_fp32.create,
                core.TYPE_BIT:  core.ConvolutionIm2Col_bit.create,
            }[fw_dtype]
        except:
            raise TypeError("unsupported")

        core_model = core_creator(filter_h_size=filter_size[0], filter_w_size=filter_size[1],
                                y_stride=stride[0], x_stride=stride[1], padding=padding, border_mode=border_mode)

        super(ConvolutionIm2Col, self).__init__(core_model=core_model)


class ConvolutionCol2Im(Model):
    """ConvolutionCol2Im class
       畳み込みの lowering における col2im 層
    """
    def __init__(self, output_size=(1, 1), fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):
        try:
            core_creator = {
                core.TYPE_FP32: core.ConvolutionCol2Im_fp32.create,
                core.TYPE_BIT:  core.ConvolutionCol2Im_bit.create,
            }[fw_dtype]
        except KeyError:
            raise TypeError("unsupported")

        core_model = core_creator(output_size[0], output_size[1])
        
        super(ConvolutionCol2Im, self).__init__(core_model=core_model)


class Convolution2d(Sequential):
    """Convolution class
       Lowering による畳み込み演算をパッキングするクラス
    """
    def __init__(self, sub_layer, filter_size=(1, 1), stride=(1, 1),
                        padding='valid', border_mode=core.BB_BORDER_REFLECT_101, border_value=0.0,
                        fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):
        
        try:
            self.core_creator = {
                core.TYPE_FP32: core.Convolution2d_fp32.create,
                core.TYPE_BIT:  core.Convolution2d_bit.create,
            }[fw_dtype]
        except KeyError:
            raise TypeError("unsupported")
        
        super(Convolution2d, self).__init__(model_list=[])
        
        self.filter_size  = filter_size
        self.stride       = stride
        self.padding      = padding
        self.border_mode  = border_mode
        self.border_value = border_value
        self.fw_dtype     = fw_dtype
        self.bw_dtype     = bw_dtype
        
        self.im2col = ConvolutionIm2Col(filter_size=filter_size, stride=stride,
                                padding=padding, border_mode=border_mode, border_value=border_value,
                                fw_dtype=fw_dtype, bw_dtype=bw_dtype)
        self.sub_layer = sub_layer
        self.col2im    = None  # 後で決定

    def get_core_model(self):
        return self.core_creator(self.sub_layer.get_core_model(), self.filter_size[0], self.filter_size[1], self.stride[0], self.stride[1], self.padding, self.border_mode, self.border_value)
    
    def set_input_shape(self, shape):
        
        # 出力サイズ計算
        input_c_size = shape[0]
        input_h_size = shape[1]
        input_w_size = shape[2]
        if self.padding == "valid":
            output_h_size = ((input_h_size - self.filter_size[0] + 1) + (self.stride[0] - 1)) // self.stride[0]
            output_w_size = ((input_w_size - self.filter_size[1] + 1) + (self.stride[1] - 1)) // self.stride[1]
        elif self.padding == "same":
            output_h_size = (input_h_size + (self.stride[0] - 1)) // self.stride[0]
            output_w_size = (input_w_size + (self.stride[1] - 1)) // self.stride[0]
        else:
            raise ValueError("illegal padding value")
        
        self.col2im = ConvolutionCol2Im(output_size=[output_h_size, output_w_size], fw_dtype=self.fw_dtype, bw_dtype=self.bw_dtype)
        
        super(Convolution2d, self).set_model_list([self.im2col, self.sub_layer, self.col2im])
        
        return super(Convolution2d, self).set_input_shape(shape)


class MaxPooling(Model):
    """MaxPooling class
    """

    def __init__(self, filter_size=(2, 2), fw_dtype=core.TYPE_FP32, bw_dtype=core.TYPE_FP32):

        try:
            core_creator = {
                core.TYPE_FP32: core.MaxPooling_fp32.create,
                core.TYPE_BIT:  core.MaxPooling_bit.create,
            }[fw_dtype]
        except:
            raise TypeError("unsupported")

        core_model = core_creator(filter_size[0], filter_size[1])

        super(MaxPooling, self).__init__(core_model=core_model)


class ReLU(Model):
    """ReLU class
    """

    def __init__(self, dtype=core.TYPE_FP32):

        try:
            core_creator = {
                core.TYPE_FP32: core.ReLU.create,
            }[dtype]
        except:
            raise TypeError("unsupported")

        core_model = core_creator()

        super(ReLU, self).__init__(core_model=core_model)

