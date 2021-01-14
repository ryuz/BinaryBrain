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
    layers = bb.get_model_list_for_rtl(net)
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
        class_digits (int): クラス分類のbit数
    """

    for images, labels in loader:
        for x, t in zip(images, labels):
            __dump_bin_int(f, t, class_digits)
            f.write('_')
            __dump_bin_img(f, x)
            f.write('\n')



def make_image_tile(rows, cols, img_gen):
    """画像をタイル状に並べて大きな画像にする

       学習用の c, h, w 順の画像データをタイル状に結合する

    Args:
        rows (int)): 縦の結合枚数
        cols (int)): 横の結合枚数
        gen (ndarray): 画像を返すジェネレータ

    Returns:
        img (ndarray) : 作成した画像
    """

    def make_image_tile_h(cols, img_gen):
        img = img_gen.__next__()
        for _ in range(1, cols):
            img = np.concatenate((img, img_gen.__next__()), axis=img.ndim-1)
        return img

    img = make_image_tile_h(cols, img_gen)
    for _ in range(1, rows):
        img = np.concatenate((img, make_image_tile_h(cols, img_gen)), axis=img.ndim-2)
    return img


def write_ppm(fname, img):
    """ppmファイルの出力

        学習用の c, h, w 順の画像データを ppm形式で保存する

    Args:
        fname (str): 出力ファイル名
        img (ndarray): モジュール名
    """

    # gray to color
    if img.ndim == 3 and img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    elif img.ndim == 2:
        img = np.stack((img, img, img))

    # shpe check    
    assert(img.ndim == 3 and img.shape[0] >= 3)
    
    # write
    with open(fname, 'w') as f:
        f.write('P3\n')
        f.write('%d %d\n' % (img.shape[2], img.shape[1]))
        f.write('255\n')
        img = img.transpose(1, 2, 0).reshape(-1, 3)
        for v in img:
            f.write('%d %d %d\n' % (v[0], v[1], v[2]))

