# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class Tensor():
    def __init__(self, shape: List[int], dtype: int = core.TYPE_FP32, host_only=False):
        if shape is not None:
            self.tensor = core.Tensor(shape, dtype, host_only)
    
    def numpy(self) -> np.ndarray:
        dtype = self.tensor.get_type()
        if dtype == core.TYPE_FP32:
            return self.tensor.numpy_fp32()
        elif dtype == core.TYPE_FP64:
            return self.tensor.numpy_fp64()
        elif dtype == core.TYPE_INT8:
            return self.tensor.numpy_int8()
        elif dtype == core.TYPE_INT16:
            return self.tensor.numpy_int16()
        elif dtype == core.TYPE_INT32:
            return self.tensor.numpy_int32()
        elif dtype == core.TYPE_INT64:
            return self.tensor.numpy_int64()
        elif dtype == core.TYPE_UINT8:
            return self.tensor.numpy_uint8()
        elif dtype == core.TYPE_UINT16:
            return self.tensor.numpy_uint16()
        elif dtype == core.TYPE_UINT32:
            return self.tensor.numpy_uint32()
        elif dtype == core.TYPE_UINT64:
            return self.tensor.numpy_uint64()
    
    @staticmethod
    def from_numpy(ndarray: np.ndarray, host_only=False):
        if not ndarray.flags['C_CONTIGUOUS']:
            ndarray = ndarray.copy(order='C')
        tensor = Tensor(shape=None)
        if ndarray.dtype == np.float32:
            tensor.tensor = bb.core.Tensor.from_numpy_fp32(ndarray, host_only)
        elif ndarray.dtype == np.float64:
            tensor.tensor = bb.core.Tensor.from_numpy_fp64(ndarray, host_only)
        elif ndarray.dtype == np.int8:
            tensor.tensor = bb.core.Tensor.from_numpy_int8(ndarray, host_only)
        elif ndarray.dtype == np.int16:
            tensor.tensor = bb.core.Tensor.from_numpy_int16(ndarray, host_only)
        elif ndarray.dtype == np.int32:
            tensor.tensor = bb.core.Tensor.from_numpy_int32(ndarray, host_only)
        elif ndarray.dtype == np.int64:
            tensor.tensor = bb.core.Tensor.from_numpy_int64(ndarray, host_only)
        elif ndarray.dtype == np.uint8:
            tensor.tensor = bb.core.Tensor.from_numpy_uint8(ndarray, host_only)
        elif ndarray.dtype == np.uint16:
            tensor.tensor = bb.core.Tensor.from_numpy_uint16(ndarray, host_only)
        elif ndarray.dtype == np.uint32:
            tensor.tensor = bb.core.Tensor.from_numpy_uint32(ndarray, host_only)
        elif ndarray.dtype == np.uint64:
            tensor.tensor = bb.core.Tensor.from_numpy_uint64(ndarray, host_only)
        else:
            raise TypeError("unsupported")
        return tensor


