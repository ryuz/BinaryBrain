# -*- coding: utf-8 -*-

import pickle
import numpy as np
import inspect
from typing import List

import binarybrain      as bb
import binarybrain.core as core


# ---- シリアライズ時のデータフォーマット定義 ----

def int_to_bytes(value: int):
    return value.to_bytes(8, 'little')

def int_from_bytes(data: bytes) -> (bytes, int):
    value = int.from_bytes(data[0:8], 'little')
    return data[8:], value

def bool_to_bytes(value: bool):
    return value.to_bytes(1, 'little')

def bool_from_bytes(data: bytes) -> (bytes, bool):
    value = bool.from_bytes(data[0:1], 'little')
    return data[1:], value


def string_to_bytes(value: str):
    value = value.encode(encoding='utf-8')
    data = int_to_bytes(len(value))
    data += value
    return data

def string_from_bytes(data: bytes) -> (bytes, str):
    data, str_len = int_from_bytes(data)
    value = data[0:str_len].decode(encoding='utf-8')
    return data[str_len:], value


def dump_object_header(object_name):
    return core.Object.write_header(object_name)

def load_object_header(data):
    load_size, object_name = core.Object.read_header(data)
    return data[load_size:], object_name



# ---- core のクラス一覧取得管理 ----

_core_class_list = inspect.getmembers(bb.core, inspect.isclass)
_core_class_dict = {k: v for (k, v) in _core_class_list}

def get_core_class_list():
    return _core_class_list

def get_core_class_dict():
    return _core_class_dict

def search_core_class(class_name):
    return _core_class_dict[class_name]


def get_core_subclass_list(superclass):
    class_list = []
    for c in get_core_class_list():
        if issubclass(c[1], superclass):
            class_list.append(c)
    return class_list

def get_core_subclass_dict(superclass):
    return {k: v for (k, v) in get_core_subclass_list(superclass)}


# ---- core.Object ----

def make_core_object_name(object_name, dtypes):
    """core側のオブジェクト名を生成 (オジェクト名＋テンプレート型)"""
    for dtype in dtypes:
        object_name = object_name + '_' + bb.dtype_to_name(dtype)
    return object_name

def split_core_object_name(core_object_name):
    """core側のオブジェクト名を分解"""
    args = core_object_name.split('_')
    dtypes = []
    for dtype_name in args[1:]:
        dtypes.append(bb.dtype_from_name(dtype_name))
    return args[0], dtypes


# core の Object class を管理
_core_object_list = get_core_subclass_list(core.Object)
_core_object_dict = get_core_subclass_dict(core.Object)

def get_core_object_list():
    return _core_object_list

def get_core_object_dict():
    return _core_object_dict

def search_core_object(object_name, dtypes):
    core_object_name = make_core_object_name(object_name, dtypes)
    return _core_object_dict[core_object_name]



# ---- core.Object 再生成 ----

def core_object_reconstruct_loads(data):
    load_size, core_object = core.object_reconstruct(data)
    return data[load_size:], core_object


_object_creator_list = []

def object_creator_regist(creator):
    _object_creator_list.append(creator)


def object_reconstruct_loads(data: bytes):
    # ヘッダ読み込み
    _, object_name = core.Object.read_header(data)
    split_name = object_name.split('_')
    name = split_name[0]
    dtypes = []
    for dtype_name in split_name[1:]:
        dtypes.append(bb.dtype_from_name(dtype_name))

    # python 内で対応クラスを探索
    for creator in _object_creator_list:
        data, obj = creator(data, name, dtypes)
        if obj is not None:
            return data, obj
    
    # core を探索
    load_size, obj = core.object_reconstruct_loads(data)
    if obj is not None:
        return data[load_size:], obj
    
    print('object_reconstruct not object found : %s'%object_name)
    return data, None


def object_loads(data: bytes):
    """バイト列からオブジェクトを再構築する

    Args:
        data(bytes): バイト列
    
    Returns:
        obj (Object): 再構築したオブジェクト
    """

    data, obj = bb.object_reconstruct_loads(data)
    if data != b'':
        print('[object_loads] warrning: data is too long')
    return obj


def object_load(filename: str):
    """ファイルからオブジェクトを再構築する

    Args:
        filename(str): ファイル名
    
    Returns:
        obj (Object): 再構築したオブジェクト
    """
    with open(filename, 'rb') as f:
        return object_loads(f.read())


# ---- Objectクラス ----

class Object():
    def __init__(self, core_object=None):
        self.core_object = core_object

    def get_core(self):
        return self.core_object
    
    def get_object_name(self):
        core_object = self.get_core()
        if core_object is not None:
            return core_object.get_object_name()
        return ''

    def dumps(self):
        core_object = self.get_core()
        if core_object is not None:
            return core_object.dump_object()
        return b''
    
    def loads(self, data):
        core_object = self.get_core()
        if core_object is not None:
            load_size = core_object.load_object(data)
            return data[load_size:]
        return data

