# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np

# type
dtype_bit   = core.TYPE_BIT
dtype_fp32  = core.TYPE_FP32
dtype_fp32  = core.TYPE_FP32
dtype_fp64  = core.TYPE_FP64
dtype_int8  = core.TYPE_INT8
dtype_int16 = core.TYPE_INT16
dtype_int32 = core.TYPE_INT32
dtype_int64 = core.TYPE_INT64
dtype_int8  = core.TYPE_UINT8
dtype_int16 = core.TYPE_UINT16
dtype_int32 = core.TYPE_UINT32
dtype_int64 = core.TYPE_UINT64


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
    if   dtype == core.TYPE_FP32:   return np.float32
    elif dtype == core.TYPE_FP64:   return np.float64
    elif dtype == core.TYPE_INT8:   return np.int8
    elif dtype == core.TYPE_INT16:  return np.int16
    elif dtype == core.TYPE_INT32:  return np.int32
    elif dtype == core.TYPE_INT64:  return np.int64
    elif dtype == core.TYPE_UINT8:  return np.uint8
    elif dtype == core.TYPE_UINT16: return np.uint16
    elif dtype == core.TYPE_UINT32: return np.uint32
    elif dtype == core.TYPE_UINT64: return np.uint64
    return None

