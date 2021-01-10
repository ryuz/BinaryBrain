# -*- coding: utf-8 -*-

import pickle
import numpy as np
#import re
import inspect
from typing import List

import binarybrain      as bb
import binarybrain.core as core


# core のクラス一覧取得
core_class_list = inspect.getmembers(bb.core, inspect.isclass)
core_class_dict = {k: v for (k, v) in core_class_list}

def get_core_class_list():
    return core_class_list

def get_core_class_dict():
    return core_class_dict

def get_core_subclass_list(superclass):
    class_list = []
    for c in get_core_class_list():
        if issubclass(c[1], superclass):
            class_list.append(c)
    return class_list

def get_core_subclass_dict(superclass):
    return {k: v for (k, v) in get_core_subclass_list(superclass)}
    
def search_core_class(class_name, dtypes=(), class_dict=get_core_class_dict()):
    objct_name = class_name
    for dtype in dtypes:
        objct_name += '_' + bb.dtype_to_name(dtype)
    return class_dict[objct_name]


core_object_list = get_core_subclass_dict(core.Object)
core_object_dict = get_core_subclass_dict(core.Object)

def search_core_object(class_name, dtypes=()):
    return search_core_class(class_name, dtypes, class_dict=core_object_dict)




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

