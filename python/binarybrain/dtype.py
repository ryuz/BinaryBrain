# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from enum import IntEnum


class DType(IntEnum):
    """データ型定義
    """

    BIT    = (0x0000 + 1)
    BINARY = (0x0000 + 2)
    FP16   = (0x0100 + 16)
    FP32   = (0x0100 + 32)
    FP64   = (0x0100 + 64)
    INT8   = (0x0200 + 8)
    INT16  = (0x0200 + 16)
    INT32  = (0x0200 + 32)
    INT64  = (0x0200 + 64)
    UINT8  = (0x0300 + 8)
    UINT16 = (0x0300 + 16)
    UINT32 = (0x0300 + 32)
    UINT64 = (0x0300 + 64)


class Border(IntEnum):
    CONSTANT    = 0
    REFLECT     = 1
    REFLECT_101 = 2
    REPLICATE   = 3
    WRAP        = 4


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

def dtype_to_name(dtype):
    if   dtype == core.TYPE_BIT:    return 'bit'
    elif dtype == core.TYPE_FP32:   return 'fp32'
    elif dtype == core.TYPE_FP64:   return 'fp64'
    elif dtype == core.TYPE_INT8:   return 'int8'
    elif dtype == core.TYPE_INT16:  return 'int16'
    elif dtype == core.TYPE_INT32:  return 'int32'
    elif dtype == core.TYPE_INT64:  return 'int64'
    elif dtype == core.TYPE_UINT8:  return 'uint8'
    elif dtype == core.TYPE_UINT16: return 'uint16'
    elif dtype == core.TYPE_UINT32: return 'uint32'
    elif dtype == core.TYPE_UINT64: return 'uint64'
    return None

def dtype_from_name(name):
    if   name == 'bit':    return core.TYPE_BIT
    elif name == 'fp32':   return core.TYPE_FP32
    elif name == 'fp64':   return core.TYPE_FP64
    elif name == 'int8':   return core.TYPE_INT8
    elif name == 'int16':  return core.TYPE_INT16
    elif name == 'int32':  return core.TYPE_INT32
    elif name == 'int64':  return core.TYPE_INT64
    elif name == 'uint8':  return core.TYPE_UINT8
    elif name == 'uint16': return core.TYPE_UINT16
    elif name == 'uint32': return core.TYPE_UINT32
    elif name == 'uint64': return core.TYPE_UINT64 
    return None


