# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core

def make_verilog_lut_layers(module_name: str, net):
    ''' make verilog source LUT layers
        変換できないモデルは影響ない層とみなして無視するので注意
    
        Args:
            module_name (str): モジュール名
            net (Model): 変換するネット
            
        Returns:
            Verilog source code (str)
    '''
    layers = bb.get_model_list(net, flatten=True)
    core_layers = []
    for layer in layers:
        core_layers.append(layer.get_core_model())
    return core.make_verilog_lut_layers(module_name, core_layers)


def make_verilog_lut_cnv_layers(module_name: str, net):
    ''' make verilog source LUT layers
        変換できないモデルは影響ない層とみなして無視するので注意
    
        Args:
            module_name (str): モジュール名
            net (Model): 変換するネット
            
        Returns:
            Verilog source code (str)
    '''
    layers = bb.get_model_list(net, flatten=True)
    core_layers = []
    for layer in layers:
        core_layers.append(layer.get_core_model())
    return core.make_verilog_lut_cnv_layers(module_name, core_layers)


def export_verilog_lut_layers(file_name: str, module_name: str, net):
    with open(file_name, 'w') as f:
        f.write(make_verilog_lut_layers(module_name, net))

        
def make_verilog_lut_cnv_layers(file_name: str, module_name: str, net):
    with open(file_name, 'w') as f:
        f.write(make_verilog_lut_cnv_layers(module_name, net))

