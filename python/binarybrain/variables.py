# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core
import numpy as np
from typing import List

class Variables():
    """Variables class
    """
    
    def __init__(self):
        self.variables = core.Variables()

    @staticmethod
    def from_core(variables):
        new_variables = Variables()
        new_variables.variables = variables
        return new_variables
    
    def get_core(self):
        return self.variables
    
    def append(self, variables):
        self.variables.push_back(variables.get_core())
    
 
