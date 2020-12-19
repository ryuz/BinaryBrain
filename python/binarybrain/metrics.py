# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Metrics():
    """Metrics class
    """
    
    def __init__(self):
        self.metrics = None
        
    def clear(self, input_shape):
        self.metrics.clear()
    
    def get(self):
        return self.metrics.get_metrics()

    def calculate(self, y_buf, t_buf):
        return self.metrics.calculate_metrics(y_buf.get_core(), t_buf.get_core())

    def get_metrics_string(self):
        return self.metrics.get_metrics_string()


class MetricsCategoricalAccuracy(Metrics):
    """MetricsCategoricalAccuracy class
    """
    
    def __init__(self):
        super(MetricsCategoricalAccuracy, self).__init__()
        self.metrics = core.MetricsCategoricalAccuracy.create()


class MetricsMeanSquaredError(Metrics):
    """MetricsMeanSquaredError class
    """
    
    def __init__(self):
        super(MetricsMeanSquaredError, self).__init__()
        self.metrics = core.MetricsMeanSquaredError.create()

