# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class Tensor():
    """Tensor class
    """
    
    def __init__(self, shape: List[int]=None, *, dtype = bb.DType.FP32, host_only=False):
        """Constructor
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        if shape is not None:
            self.core_tensor = core.Tensor(shape, dtype.value, host_only)

    @staticmethod
    def from_core(core_tensor):
        new_tensor = Tensor(shape=None)
        new_tensor.core_tensor = core_tensor
        return new_tensor
    
    def get_core(self):
        return self.core_tensor
        
    def get_type(self) -> int:
        """get data type.
        
        Returns:
            data type.
        """
        return bb.DType(self.core_tensor.get_type())
    
    def get_shape(self) -> List[int]:
        """get shape.
        
        Returns:
            shape
        """
        return self.core_tensor.get_shape()
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy
        
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        dtype = self.core_tensor.get_type()
        if dtype == bb.DType.FP32:
            return self.core_tensor.numpy_fp32()
        elif dtype == bb.DType.FP64:
            return self.core_tensor.numpy_fp64()
        elif dtype == bb.DType.INT8:
            return self.core_tensor.numpy_int8()
        elif dtype == bb.DType.INT16:
            return self.core_tensor.numpy_int16()
        elif dtype == bb.DType.INT32:
            return self.core_tensor.numpy_int32()
        elif dtype == bb.DType.INT64:
            return self.core_tensor.numpy_int64()
        elif dtype == bb.DType.UINT8:
            return self.core_tensor.numpy_uint8()
        elif dtype == bb.DType.UINT16:
            return self.core_tensor.numpy_uint16()
        elif dtype == bb.DType.UINT32:
            return self.core_tensor.numpy_uint32()
        elif dtype == bb.DType.UINT64:
            return self.core_tensor.numpy_uint64()
    
    @staticmethod
    def from_numpy(ndarray: np.ndarray, host_only=False):
        """Create from NumPy
        
        Args:
            ndarray (np.ndarray): array of NumPy
            host_only (bool): flag of host only
        """
        
        if not ndarray.flags['C_CONTIGUOUS']:
            ndarray = ndarray.copy(order='C')
        new_tensor = Tensor(shape=None)
        if ndarray.dtype == np.float32:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_fp32(ndarray, host_only)
        elif ndarray.dtype == np.float64:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_fp64(ndarray, host_only)
        elif ndarray.dtype == np.int8:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_int8(ndarray, host_only)
        elif ndarray.dtype == np.int16:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_int16(ndarray, host_only)
        elif ndarray.dtype == np.int32:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_int32(ndarray, host_only)
        elif ndarray.dtype == np.int64:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_int64(ndarray, host_only)
        elif ndarray.dtype == np.uint8:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_uint8(ndarray, host_only)
        elif ndarray.dtype == np.uint16:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_uint16(ndarray, host_only)
        elif ndarray.dtype == np.uint32:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_uint32(ndarray, host_only)
        elif ndarray.dtype == np.uint64:
            new_tensor.core_tensor = bb.core.Tensor.from_numpy_uint64(ndarray, host_only)
        else:
            raise TypeError("unsupported")
        return new_tensor


    def __add__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = self.core_tensor + x.core_tensor
        else:
            new_tensor.core_tensor = self.core_tensor + float(x)
        return new_tensor

    def __sub__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = self.core_tensor - x.core_tensor
        else:
            new_tensor.core_tensor = self.core_tensor - float(x)
        return new_tensor

    def __mul__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = self.core_tensor * x.core_tensor
        else:
            new_tensor.core_tensor = self.core_tensor * float(x)
        return new_tensor

    def __truediv__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = self.core_tensor / x.core_tensor
        else:
            new_tensor.core_tensor = self.core_tensor / float(x)
        return new_tensor

    def __radd__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = x.core_tensor + self.core_tensor 
        else:
            new_tensor.core_tensor = float(x) + self.core_tensor
        return new_tensor

    def __rsub__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = x.core_tensor - self.core_tensor 
        else:
            new_tensor.core_tensor = float(x) - self.core_tensor
        return new_tensor

    def __rmul__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = x.core_tensor * self.core_tensor 
        else:
            new_tensor.core_tensor = float(x) * self.core_tensor
        return new_tensor

    def __rtruediv__(self, x):
        new_tensor = Tensor(shape=None)
        if type(x) == Tensor:
            new_tensor.core_tensor = x.core_tensor / self.core_tensor 
        else:
            new_tensor.core_tensor = float(x) / self.core_tensor
        return new_tensor
    
    def __iadd__(self, x):
        if type(x) == Tensor:
            self.core_tensor += x.core_tensor 
        else:
            self.core_tensor += float(x)
        return self
    
    def __isub__(self, x):
        if type(x) == Tensor:
            self.core_tensor -= x.core_tensor 
        else:
            self.core_tensor -= float(x)
        return self
    
    def __imul__(self, x):
        if type(x) == Tensor:
            self.core_tensor *= x.core_tensor 
        else:
            self.core_tensor *= float(x)
        return self
    
    def __itruediv__(self, x):
        if type(x) == Tensor:
            self.core_tensor /= x.core_tensor 
        else:
            self.core_tensor /= float(x)
        return self
    