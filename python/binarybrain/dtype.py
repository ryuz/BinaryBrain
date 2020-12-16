# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np

# type
fp32  = core.TYPE_FP32
fp64  = core.TYPE_FP64
int8  = core.TYPE_INT8
int16 = core.TYPE_INT16
int32 = core.TYPE_INT32
int64 = core.TYPE_INT64
int8  = core.TYPE_UINT8
int16 = core.TYPE_UINT16
int32 = core.TYPE_UINT32
int64 = core.TYPE_UINT64


dict_dtype_numpy_to_bb = {
    np.float32: core.TYPE_FP32,
    np.float64: core.TYPE_FP64,
    np.int8   : core.TYPE_INT8,
    np.int16  : core.TYPE_INT16,
    np.int32  : core.TYPE_INT32,
    np.int64  : core.TYPE_INT64,
    np.uint8  : core.TYPE_UINT8,
    np.uint16 : core.TYPE_UINT16,
    np.uint32 : core.TYPE_UINT32,
    np.uint64 : core.TYPE_UINT64,
    
}

dict_dtype_bb_to_numpy = {
    core.TYPE_FP32:   np.float32,
    core.TYPE_FP64:   np.float64,
    core.TYPE_INT8:   np.int8,
    core.TYPE_INT16:  np.int16,
    core.TYPE_INT32:  np.int32,
    core.TYPE_INT64:  np.int64,
    core.TYPE_UINT8:  np.uint8,
    core.TYPE_UINT16: np.uint16,
    core.TYPE_UINT32: np.uint32,
    core.TYPE_UINT64: np.uint64,
}
    
def dtype_numpy_to_bb(dtype):
    if   dtype == np.float32: return core.TYPE_FP32
    elif dtype == np.float64: return core.TYPE_FP64
    elif dtype == np.int8:    return core.TYPE_INT8
    elif dtype == np.int16:   return core.TYPE_INT16
    elif dtype == np.int32:   return core.TYPE_INT32
    elif dtype == np.int64:   return core.TYPE_INT64
    elif dtype == np.uint8:   return core.TYPE_UINT8
    elif dtype == np.uint16:  return core.TYPE_UINT16
    elif dtype == np.uint32:  return core.TYPE_UINT32
    elif dtype == np.uint64:  return core.TYPE_UINT64
    return None

def dtype_bb_to_numpy(dtype):
    return dict_dtype_bb_to_numpy[dtype]

