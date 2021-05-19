# arma.py

import sklearn as skl
import numpy as np
from src.pmodel import PredictionModel

class ARMA(PredictionModel):
    """
    Class for autoregressive moving average with LASSO regularization.
    """
    
    def __init__(self, input_length=14, lam=1.):
        """
        Args:
            input_length (int): days for input time-series
            lam (float): regularizer coefficient
        """
        PredictionModel.__init__(self)
        self._input_length = input_length
        self._lam = lam
        
    def fit(self, x):
        self._fitted = True
        raise NotImplementedError
        
    def predict_hospitalizations(self, days):
        assert self._fitted, "Model not yet fit"
        raise NotImplementedError
        
    def predict_deaths(self, days):
        assert self._fitted, "Model not yet fit"
        raise NotImplementedError