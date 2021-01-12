# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class FrameBuffer(bb.Object):
    """FrameBuffer class
       
        BinaryBrainでの学習データを格納する特別な型である
        バッチと対応する 1次元のframe項と、各レイヤーの入出力ノードに対応する
        多次元のnode項を有している
        numpy の ndarray に変換する際は axis=0 が frame 項となり、以降が
        node項となる

        Tensor と異なり、frame 項に対して reshape を行うことはできず、
        node 項に対しても transpose することはできない。

        node に関しては 2次元以上の shape も持ちうるが、実際のレイヤー間接続に
        際しては、畳み込みなどの次元に意味を持つノード以外では、ノード数さえ
        あっていれば接続できるものが殆どである(多くの処理系で必要とする
        flatten が省略できる)。

        host_only フラグを指定すると device(GPU側) が利用可能であっても
        host(CPU側) のみにメモリを確保する

    Args:
        frame_size (int): frame サイズ
        shape (list[int]):  node シェイプ
        dtype (int): Data type
        host_only (bool): flag of host only
    """
    
    def __init__(self, frame_size: int = 0, shape: List[int] = [], dtype = bb.DType.FP32, host_only: bool = False, core_buf=None):
        if core_buf is None:
            core_buf = core.FrameBuffer(frame_size, shape, dtype.value, host_only)
        super(FrameBuffer, self).__init__(core_object=core_buf)

    @staticmethod
    def from_core(core_buf):
        return FrameBuffer(core_buf=core_buf)

    def is_host_only(self) -> bool:
        return self.get_core().is_host_only()

    def get_type(self) -> int:
        """データ型取得
        
        Returns:
            dtype (DType)
        """
        return bb.DType(self.get_core().get_type())
    
    def get_frame_size(self) -> int:
        """get size of frame.
        
        Returns:
            frame size.
        """
        return self.get_core().get_frame_size()
    
    def get_frame_stride(self) -> int:
        """get stride of frame.
        
        Returns:
            frame stride.
        """
        return self.get_core().get_frame_stride()
        
    def get_node_size(self) -> int:
        """get size of node.
        
        Returns:
            node size.
        """
        return self.get_core().get_node_size()
    
    def get_node_shape(self) -> List[int]:
        """get shape of node.
        
        Returns:
            shape
        """
        return self.get_core().get_node_shape()
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy
        
        Args:
            shape (list[int]):  Shape of created array
            dtype (int): Data type
            host_only (bool): flag of host only
        """
        dtype = self.get_type()
        if dtype == bb.DType.BIT:
            ndarray = self.get_core().numpy_uint8()
        if dtype == bb.DType.BINARY:
            ndarray = self.get_core().numpy_uint8()
        elif dtype == bb.DType.FP32:
            ndarray = self.get_core().numpy_fp32()
        elif dtype == bb.DType.FP64:
            ndarray = self.get_core().numpy_fp64()
        elif dtype == bb.DType.INT8:
            ndarray =  self.get_core().numpy_int8()
        elif dtype == bb.DType.INT16:
            ndarray =  self.get_core().numpy_int16()
        elif dtype == bb.DType.INT32:
            ndarray =  self.get_core().numpy_int32()
        elif dtype == bb.DType.INT64:
            ndarray =  self.get_core().numpy_int64()
        elif dtype == bb.DType.UINT8:
            ndarray =  self.get_core().numpy_uint8()
        elif dtype == bb.DType.UINT16:
            ndarray =  self.get_core().numpy_uint16()
        elif dtype == bb.DType.UINT32:
            ndarray =  self.get_core().numpy_uint32()
        elif dtype == bb.DType.UINT64:
            ndarray =  self.get_core().numpy_uint64()
        else:
            raise TypeError("unexpected dtype")

        tran = [ndarray.ndim-1] + list(range(0, ndarray.ndim-1))
        ndarray = ndarray.transpose(tran)
        if dtype != bb.DType.BIT:
            shape = list(ndarray.shape)
            shape[0] = self.get_core().get_frame_size()
            ndarray = np.resize(ndarray, shape)
        
        return ndarray
        
    @staticmethod
    def from_numpy(ndarray: np.ndarray, host_only=False):
        """Create from NumPy
        
        Args:
            ndarray (np.ndarray): array of NumPy
            host_only (bool): flag of host only
        """
        
        shape = list(ndarray.shape)
        assert(len(shape) >= 2)
        bb_dtype = bb.dtype_numpy_to_bb(ndarray.dtype)
        frame_size   = shape[0]
        frame_stride = core.FrameBuffer.calc_frame_stride(bb_dtype, frame_size) 
        shape[0] = frame_stride // core.dtype_get_byte_size(bb_dtype)
        ndarray = np.resize(ndarray, shape)
        
        tran = list(range(1, ndarray.ndim)) + [0]
        ndarray = ndarray.transpose(tran)
        ndarray = ndarray.copy(order='C')
        
        if ndarray.dtype == np.float32:
            core_buf = bb.core.FrameBuffer.from_numpy_fp32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:], host_only)
        elif ndarray.dtype == np.float64:
            core_buf = bb.core.FrameBuffer.from_numpy_fp64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int8:
            core_buf = bb.core.FrameBuffer.from_numpy_int8(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int16:
            core_buf = bb.core.FrameBuffer.from_numpy_int16(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int32:
            core_buf = bb.core.FrameBuffer.from_numpy_int32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.int64:
            core_buf = bb.core.FrameBuffer.from_numpy_int64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint8:
            core_buf = bb.core.FrameBuffer.from_numpy_uint8(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint16:
            core_buf = bb.core.FrameBuffer.from_numpy_uint16(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint32:
            core_buf = bb.core.FrameBuffer.from_numpy_uint32(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        elif ndarray.dtype == np.uint64:
            core_buf = bb.core.FrameBuffer.from_numpy_uint64(ndarray, bb_dtype, frame_size, frame_stride, shape[1:],host_only)
        else:
            core_buf = None
            raise TypeError("unsupported")
        return FrameBuffer(core_buf=core_buf)
    
    
    def __add__(self, x):
        if type(x) == FrameBuffer:
            core_buf = self.get_core() + x.get_core()
        else:
            core_buf = self.get_core() + float(x)
        return FrameBuffer(core_buf=core_buf)

    def __sub__(self, x):
        if type(x) == FrameBuffer:
            core_buf = self.get_core() - x.get_core()
        else:
            core_buf = self.get_core() - float(x)
        return FrameBuffer(core_buf=core_buf)

    def __mul__(self, x):
        if type(x) == FrameBuffer:
            core_buf = self.get_core() * x.get_core()
        else:
            core_buf = self.get_core() * float(x)
        return FrameBuffer(core_buf=core_buf)

    def __truediv__(self, x):
        if type(x) == FrameBuffer:
            core_buf = self.get_core() / x.get_core()
        else:
            core_buf = self.get_core() / float(x)
        return FrameBuffer(core_buf=core_buf)

    def __radd__(self, x):
        if type(x) == FrameBuffer:
            core_buf = x.get_core() + self.get_core() 
        else:
            core_buf = float(x) + self.get_core()
        return FrameBuffer(core_buf=core_buf)

    def __rsub__(self, x):
        if type(x) == FrameBuffer:
            core_buf = x.get_core() - self.get_core() 
        else:
            core_buf = float(x) - self.get_core()
        return FrameBuffer(core_buf=core_buf)

    def __rmul__(self, x):
        if type(x) == FrameBuffer:
            core_buf = x.get_core() * self.get_core() 
        else:
            core_buf = float(x) * self.get_core()
        return FrameBuffer(core_buf=core_buf)

    def __rtruediv__(self, x):
        if type(x) == FrameBuffer:
            core_buf = x.get_core() / self.get_core() 
        else:
            core_buf = float(x) / self.get_core()
        return FrameBuffer(core_buf=core_buf)
    
    def __iadd__(self, x):
        core_buf = self.get_core()
        if type(x) == FrameBuffer:
            core_buf += x.get_core() 
        else:
            core_buf += float(x)
        return self
    
    def __isub__(self, x):
        core_buf = self.get_core()
        if type(x) == FrameBuffer:
            core_buf -= x.get_core() 
        else:
            core_buf -= float(x)
        return self
    
    def __imul__(self, x):
        core_buf = self.get_core()
        if type(x) == FrameBuffer:
            core_buf *= x.get_core() 
        else:
            core_buf *= float(x)
        return self
    
    def __itruediv__(self, x):
        core_buf = self.get_core()
        if type(x) == FrameBuffer:
            core_buf /= x.get_core() 
        else:
            core_buf /= float(x)
        return self

