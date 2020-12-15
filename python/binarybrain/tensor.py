# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class Tensor():
    """Tensor class
    """
    
    def __init__(self, shape: List[int], dtype: int = core.TYPE_FP32, host_only=False):
        """Constructor
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        if shape is not None:
            self.tensor = core.Tensor(shape, dtype, host_only)
    
    def get_type(self) -> int:
        """get data type.
        
        Returns:
            data type.
        """
        return self.tensor.get_type()
    
    def get_shape(self) -> List[int]:
        """get shape.
        
        Returns:
            shape
        """
        return self.tensor.get_shape()
        
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy
        
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
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
        """Create from NumPy
        
        Args:
            ndarray (np.ndarray): array of NumPy
            host_only (bool): flag of host only
        """
        
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


    def __add__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = self.tensor + x.tensor
        else:
            tensor.tensor = self.tensor + float(x)
        return tensor

    def __sub__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = self.tensor - x.tensor
        else:
            tensor.tensor = self.tensor - float(x)
        return tensor

    def __mul__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = self.tensor * x.tensor
        else:
            tensor.tensor = self.tensor * float(x)
        return tensor

    def __truediv__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = self.tensor / x.tensor
        else:
            tensor.tensor = self.tensor / float(x)
        return tensor

    def __radd__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = x.tensor + self.tensor 
        else:
            tensor.tensor = float(x) + self.tensor
        return tensor

    def __rsub__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = x.tensor - self.tensor 
        else:
            tensor.tensor = float(x) - self.tensor
        return tensor

    def __rmul__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = x.tensor * self.tensor 
        else:
            tensor.tensor = float(x) * self.tensor
        return tensor

    def __rtruediv__(self, x):
        tensor = Tensor(shape=None)
        if type(x) == Tensor:
            tensor.tensor = x.tensor / self.tensor 
        else:
            tensor.tensor = float(x) / self.tensor
        return tensor
    
    def __iadd__(self, x):
        if type(x) == Tensor:
            self.tensor += x.tensor 
        else:
            self.tensor += float(x)
        return self
    
    def __isub__(self, x):
        if type(x) == Tensor:
            self.tensor -= x.tensor 
        else:
            self.tensor -= float(x)
        return self
    
    def __imul__(self, x):
        if type(x) == Tensor:
            self.tensor *= x.tensor 
        else:
            self.tensor *= float(x)
        return self
    
    def __itruediv__(self, x):
        if type(x) == Tensor:
            self.tensor /= x.tensor 
        else:
            self.tensor /= float(x)
        return self
    