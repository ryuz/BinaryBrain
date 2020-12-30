# -*- coding: utf-8 -*-

import numpy as np

import binarybrain      as bb
import binarybrain.core as core


# ----- LUT Layer -----

def make_verilog_lut_layers(module_name: str, net):
    layers = bb.get_model_list(net, flatten=True)
    core_layers = []
    for layer in layers:
        core_layers.append(layer.get_core())
    return core.make_verilog_lut_layers(module_name, core_layers)

def dump_verilog_lut_layers(f, module_name: str, net):
    ''' make verilog source of LUT layers
        変換できないモデルは影響ない層とみなして無視するので注意
    
        Args:
            f (StreamIO) : 出力先ストリーム
            module_name (str): モジュール名
            net (Model): 変換するネット
            
        Returns:
            Verilog source code (str)
    '''
    f.write(make_verilog_lut_layers(module_name=module_name, net=net))

def export_verilog_lut_layers(file_name: str, module_name: str, net):
    with open(file_name, 'w') as f:
        dump_verilog_lut_layers(f, module_name, net)



# ----- Convolutional LUT Layer -----

def make_verilog_lut_cnv_layers(module_name: str, net):
    layers = bb.get_model_list(net, flatten=True)
    core_layers = []
    for layer in layers:
        core_layers.append(layer.get_core())
    return core.make_verilog_lut_cnv_layers(module_name, core_layers)

def dump_verilog_lut_cnv_layers(f, module_name: str, net):
    ''' dump verilog source of Convolutional LUT layers
        
        畳み込み層を含むネットを AXI4 Stream Video 形式のVerilogソースコードして
        出力する。
        縮小を伴う MaxPooling 層は最後に1個だけ挿入を許される

        Args:
            f (StreamIO) : 出力先ストリーム
            module_name (str): モジュール名
            net (Model): 変換するネット
    '''
    f.write(make_verilog_lut_cnv_layers(module_name, net))

def export_verilog_lut_cnv_layers(file_name: str, module_name: str, net):
    with open(file_name, 'w') as f:
        dump_verilog_lut_cnv_layers(f, module_name, net)



# ----- For Simuration -----

def __dump_bin_digit(f, v):
    if v:
        f.write('1')
    else:
        f.write('0')

def __dump_bin_int(f, v, digits):
    for i in range(digits):
        __dump_bin_digit(f, ((v >> (digits-1-i)) & 1))

def __dump_bin_img(f, img):
    img = np.array(img).flatten()[::-1]
    for v in img:
        __dump_bin_digit(f, v > 0.5)

def dump_verilog_readmemb_image_classification(f, loader, *, class_digits=8):
    """verilog用データダンプ
    verilog の $readmemb() での読み込み用データ作成

    クラスID + 画像データの形式で出力する

    Args:
        f (StreamIO): 出力先
        loader (Loader): モジュール名
        class_digits (int)): クラス分類のbit数
    """

    for images, labels in loader:
        for x, t in zip(images, labels):
            __dump_bin_int(f, t, class_digits)
            f.write('_')
            __dump_bin_img(f, x)
            f.write('\n')
