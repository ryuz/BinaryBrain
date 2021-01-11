# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class LossFunction():
    """LossFunction class
       損失関数の基底クラス
    """
    
    def __init__(self, core_loss=None):
        self.core_loss = core_loss
        
    def clear(self):
        """値のクリア

           集計をクリアする。通常 epoch の単位でクリアして再集計を行う
        """
        self.core_loss.clear()
    
    def get(self):
        """値の取得

        Returns:
            loss(float) : 現在までの損失値を返す
        """
        return self.core_loss.get_loss()

    def calculate(self, y_buf, t_buf, mini_batch_size=None):
        """損失の計算

            mini_batch_size はメモリなどの都合でミニバッチをさらに分割する場合などに用いる。通常はNoneでよい。

        Args:
            y_buf (FrameBuffer): forward演算結果
            t_buf (FrameBuffer): 教師データ
            mini_batch_size (int): ミニバッチサイズ

        Returns:
            dy_buf (FrameBuffer) : 逆伝搬させる誤差を返す
        """
        if mini_batch_size is None:
            mini_batch_size = t_buf.get_frame_size()
        return bb.FrameBuffer.from_core(self.core_loss.calculate_loss(y_buf.get_core(), t_buf.get_core(), mini_batch_size))



class LossSoftmaxCrossEntropy(LossFunction):
    """LossSoftmaxCrossEntropy class

       Softmax(活性化) と CrossEntropy(損失関数) の複合
       両者を統合すると計算が簡略化される。
       
       利用に際しては最終段にSoftmaxが挿入されるので注意すること。
    """
    
    def __init__(self, dtype=bb.DType.FP32):
        core_loss = bb.search_core_object('LossSoftmaxCrossEntropy', [dtype]).create()
        super(LossSoftmaxCrossEntropy, self).__init__(core_loss=core_loss)


class LossMeanSquaredError(LossFunction):
    """LossMeanSquaredError class

        平均二乗誤差(MSE)を計算して誤差として戻す
    """
    
    def __init__(self, dtype=bb.DType.FP32):
        core_loss = bb.search_core_object('LossSoftmaxCrossEntropy', [dtype]).create()
        super(LossMeanSquaredError, self).__init__(core_loss=core_loss)

