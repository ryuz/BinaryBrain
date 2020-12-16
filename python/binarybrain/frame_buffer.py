# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class FrameBuffer():
    """FrameBuffer class
    """
    
    
    def __init__(self, frame_size: int = 0, shape: List[int] = [], dtype: int = 0, host_only: bool = False):
        """Constructor
        Args:
            frame_size (int): size of frame
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        self.buf = core.FrameBuffer(frame_size, shape, dtype, host_only)
    
    def get_type(self) -> int:
        """get data type.
        
        Returns:
            data type.
        """
        return self.buf.get_type()
    
    def get_frame_size(self) -> int:
        """get size of frame.
        
        Returns:
            frame size.
        """
        return self.buf.get_frame_size()
        
    def get_node_size(self) -> int:
        """get size of node.
        
        Returns:
            node size.
        """
        return self.buf.get_node_size()
    
    def get_node_shape(self) -> List[int]:
        """get shape of node.
        
        Returns:
            shape
        """
        return self.buf.get_node_shape()
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy
        
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        dtype = self.buf.get_type()
        if dtype == core.TYPE_BIT:
            ndarray = self.buf.numpy_uint8()
        if dtype == core.TYPE_BINARY:
            ndarray = self.buf.numpy_uint8()
        elif dtype == core.TYPE_FP32:
            ndarray = self.buf.numpy_fp32()
        elif dtype == core.TYPE_FP64:
            ndarray = self.buf.numpy_fp64()
        elif dtype == core.TYPE_INT8:
            ndarray =  self.buf.numpy_int8()
        elif dtype == core.TYPE_INT16:
            ndarray =  self.buf.numpy_int16()
        elif dtype == core.TYPE_INT32:
            ndarray =  self.buf.numpy_int32()
        elif dtype == core.TYPE_INT64:
            ndarray =  self.buf.numpy_int64()
        elif dtype == core.TYPE_UINT8:
            ndarray =  self.buf.numpy_uint8()
        elif dtype == core.TYPE_UINT16:
            ndarray =  self.buf.numpy_uint16()
        elif dtype == core.TYPE_UINT32:
            ndarray =  self.buf.numpy_uint32()
        elif dtype == core.TYPE_UINT64:
            ndarray =  self.buf.numpy_uint64()
        else:
            raise TypeError("unexpected dtype")
        
        tran = [ndarray.ndim-1] + list(range(0, ndarray.ndim-1))
        ndarray = ndarray.transpose(tran)
        if dtype != core.TYPE_BIT:
            shape = ndarray.shape
            shape[0] = self.buf.get_frame_size()
            ndarray = ndarray.resize(shape)
        
        return ndarray
    
    @staticmethod
    def from_numpy(ndarray: np.ndarray, host_only=False):
        """Create from NumPy
        
        Args:
            ndarray (np.ndarray): array of NumPy
            host_only (bool): flag of host only
        """
        
        shape = list(ndarray.shape)
        bb_dtype = bb.dtype_numpy_to_bb(ndarray.dtype)
        frame_size   = shape[0]
        frame_stride = core.FrameBuffer.calc_frame_stride(bb_dtype, frame_size) 
        shape[0] = frame_stride // core.dtype_get_byte_size(bb_dtype)
        ndarray.resize(shape, refcheck=False)
        
        tran = list(range(1, ndarray.ndim)) + [0]
        ndarray = ndarray.transpose(tran)
        ndarray = ndarray.copy(order='C')
        
        buf = FrameBuffer()
        if ndarray.dtype == np.float32:
            buf.buf = bb.core.FrameBuffer.from_numpy_fp32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:], host_only)
        elif ndarray.dtype == np.float64:
            buf.buf = bb.core.FrameBuffer.from_numpy_fp64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int8:
            buf.buf = bb.core.FrameBuffer.from_numpy_int8(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int16:
            buf.buf = bb.core.FrameBuffer.from_numpy_int16(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int32:
            buf.buf = bb.core.FrameBuffer.from_numpy_int32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int64:
            buf.buf = bb.core.FrameBuffer.from_numpy_int64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint8:
            buf.buf = bb.core.FrameBuffer.from_numpy_uint8(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint16:
            buf.buf = bb.core.FrameBuffer.from_numpy_uint16(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint32:
            buf.buf = bb.core.FrameBuffer.from_numpy_uint32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint64:
            buf.buf = bb.core.FrameBuffer.from_numpy_uint64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        else:
            raise TypeError("unsupported")
        return buf
    
    
    def __add__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = self.buf + x.buf
        else:
            buf.buf = self.buf + float(x)
        return buf

    def __sub__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = self.buf - x.buf
        else:
            buf.buf = self.buf - float(x)
        return buf

    def __mul__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = self.buf * x.buf
        else:
            buf.buf = self.buf * float(x)
        return buf

    def __truediv__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = self.buf / x.buf
        else:
            buf.buf = self.buf / float(x)
        return buf

    def __radd__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = x.buf + self.buf 
        else:
            buf.buf = float(x) + self.buf
        return buf

    def __rsub__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = x.buf - self.buf 
        else:
            buf.buf = float(x) - self.buf
        return buf

    def __rmul__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = x.buf * self.buf 
        else:
            buf.buf = float(x) * self.buf
        return buf

    def __rtruediv__(self, x):
        buf = FrameBuffer()
        if type(x) == FrameBuffer:
            buf.buf = x.buf / self.buf 
        else:
            buf.buf = float(x) / self.buf
        return buf
    
    def __iadd__(self, x):
        if type(x) == FrameBuffer:
            self.buf += x.buf 
        else:
            self.buf += float(x)
        return self
    
    def __isub__(self, x):
        if type(x) == FrameBuffer:
            self.buf -= x.buf 
        else:
            self.buf -= float(x)
        return self
    
    def __imul__(self, x):
        if type(x) == FrameBuffer:
            self.buf *= x.buf 
        else:
            self.buf *= float(x)
        return self
    
    def __itruediv__(self, x):
        if type(x) == FrameBuffer:
            self.buf /= x.buf 
        else:
            self.buf /= float(x)
        return self

