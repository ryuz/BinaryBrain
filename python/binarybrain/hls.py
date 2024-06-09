# -*- coding: utf-8 -*-

import io
import numpy as np

import binarybrain      as bb
import binarybrain.core as core


def make_lut_func_name(name, node):
    return "%s_lut_%d"%(name, node)


def dump_hls_lut_node(f, name, lut, node):
    f.write("\ninline ap_uint<1> %s(\n"%(make_lut_func_name(name, node)))
    n = lut.get_node_connection_size(node)
    s = lut.get_lut_table_size(node)
    for i in range(n):
        f.write("        ap_uint<1> in_data%d"%(i))
        if i < n-1:
            f.write(",\n")
        else:
            f.write(")\n")
    f.write("{\n")
    f.write("    #pragma HLS inline\n\n")
    f.write("    ap_uint<%d> index;\n"%(n))
    for i in range(n):
        f.write("    index[%d] = in_data%d;\n"%(i, i))
    f.write("    \n")
    f.write("    const ap_uint<1> table[%d] = {"%(s))
    for i in range(s):
        f.write("%d,"%(lut.get_lut_table(node ,i)))
    f.write("};\n")
    f.write("    #pragma HLS bind_storage variable=table type=ROM_1P impl=LUTRAM\n")
    f.write("    return table[index];\n")
    f.write("}\n\n")

def dump_hls_lut_layer(f, name, lut):
    ''' dump HLS source of LUT layer
    
        Args:
            f (StreamIO) : 出力先ストリーム
            name (str): 関数名
            lut (Model): 変換するネット
    '''

    ins  = lut.get_input_node_size()
    outs = lut.get_output_node_size()
    for node in range(outs):
        dump_hls_lut_node(f, name, lut, node)
    
    f.write("\n")
    f.write("inline ap_uint<%d> %s(ap_uint<%d> in_data)\n"%(outs, name, ins))
    f.write("{\n")
    f.write("    ap_uint<%d>  out_data;\n"%(outs))
    for node in range(outs):
        f.write("    out_data[%d] = %s("%(node, make_lut_func_name(name, node)))
        n = lut.get_node_connection_size(node)
        for i in range(n):
            f.write("in_data[%d]"%(lut.get_node_connection_index(node, i)))
            if i < n-1: 
                f.write(",")
            else:
                f.write(");\n")
    f.write("    return out_data;\n")   
    f.write("}\n\n")


def make_hls_lut_layer(name, lut):
    ''' make HLS source of LUT layer
    
        Args:
            name (str): 関数名
            lut (Model): 変換するネット
            
        Returns:
            HLS source code (str)
    '''
    
    with io.StringIO() as f:
        dump_hls_lut_layer(f, name, lut)
        return f.getvalue()
