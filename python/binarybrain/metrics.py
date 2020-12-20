# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List



class Metrics():
    """Metrics class
    """
    
    def __init__(self, core_metrics=None):
        self.core_metrics = core_metrics
        
    def clear(self):
        self.core_metrics.clear()
    
    def get(self):
        return self.core_metrics.get_metrics()

    def calculate(self, y_buf, t_buf):
        return self.core_metrics.calculate_metrics(y_buf.get_core(), t_buf.get_core())

    def get_metrics_string(self):
        return self.core_metrics.get_metrics_string()


class MetricsCategoricalAccuracy(Metrics):
    """MetricsCategoricalAccuracy class
    """
    
    def __init__(self):
        core_metrics = core.MetricsCategoricalAccuracy.create()
        super(MetricsCategoricalAccuracy, self).__init__(core_metrics=core_metrics)


class MetricsMeanSquaredError(Metrics):
    """MetricsMeanSquaredError class
    """
    
    def __init__(self):
        core_metrics = core.MetricsMeanSquaredError.create()
        super(MetricsMeanSquaredError, self).__init__(core_metrics=core_metrics)

