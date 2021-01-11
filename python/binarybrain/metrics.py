# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Metrics():
    """Metrics class
        評価関数の基底クラス
    """
    
    def __init__(self, core_metrics=None):
        self.core_metrics = core_metrics
        
    def clear(self):
        """値のクリア

           集計をクリアする。通常 epoch の単位でクリアして再集計を行う
        """
        self.core_metrics.clear()
    
    def get(self):
        """値の取得

        Returns:
            metrics(float) : 現在までの損失値を返す
        """
        return self.core_metrics.get_metrics()

    def calculate(self, y_buf, t_buf):
        """評価の計算

        Args:
            y_buf (FrameBuffer): forward演算結果
            t_buf (FrameBuffer): 教師データ
        """
        return self.core_metrics.calculate_metrics(y_buf.get_core(), t_buf.get_core())

    def get_metrics_string(self):
        """評価対象の文字列取得

        評価関数ごとに評価値の単位が異なるため計算しているものの文字列を返す
        平均二乗誤差(MSE)であったり、認識率(accuracy)であったり
        getで得られる値を、表示やログで表す際に利用できる

        Args:
            metrics_string (str): 評価対象の文字列取得
        """
        return self.core_metrics.get_metrics_string()


class MetricsCategoricalAccuracy(Metrics):
    """MetricsCategoricalAccuracy class

       クラス分類用の評価関数
       一致率を accuracy として計算する
    """
    
    def __init__(self, dtype=bb.DType.FP32):
        core_metrics = bb.search_core_object('MetricsCategoricalAccuracy', [dtype]).create()
        super(MetricsCategoricalAccuracy, self).__init__(core_metrics=core_metrics)


class MetricsMeanSquaredError(Metrics):
    """MetricsMeanSquaredError class

       平均二乗誤差の評価関数
       教師信号との平均二乗誤差を計算する
    """
    
    def __init__(self, dtype=bb.DType.FP32):
        core_metrics = bb.search_core_object('MetricsMeanSquaredError', [dtype]).create()
        super(MetricsMeanSquaredError, self).__init__(core_metrics=core_metrics)

