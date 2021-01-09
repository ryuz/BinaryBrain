# -*- coding: utf-8 -*-

import pickle
import numpy as np
from typing import List

import binarybrain      as bb
import binarybrain.core as core


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

