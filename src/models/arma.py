# arma.py

import sklearn as skl
import numpy as np
from src.pmodel import PredictionModel
from sklearn.linear_model import Lasso

class ARMA(PredictionModel):
    """
    Class for autoregressive moving average with LASSO regularization.
    """

    def __init__(self, roll_window_size=14, lam=1., alpha = 5.0, normalize = False, max_iter = 1e7):
        """
        Args:
            input_length (int): days for input time-series
            lam (float): regularizer coefficient
        """
        PredictionModel.__init__(self)
        self._lam = lam
        self._normalize = normalize
        self._alpha = alpha
        self._max_iter = max_iter
        self._roll_window_size = roll_window_size
        self._model = Lasso(alpha=self._alpha, max_iter=self._max_iter)

    def fit(self, x):
        """
        Fit model parameters using data.
        Args:
            x (np.array): (ntimes, 3) data
        """

        c_train, h_train, d_train = x[:,0], x[:,1], x[:,2]

        if self._normalize:
            self._max_h = max(h_train)
            self._max_d = max(d_train)
        else:
            self._max_h, self._max_d = 1, 1

        h_train = h_train/self._max_h
        d_train = d_train/self._max_d

        train_size = h_train.shape[0]

        x_list, y_list = [], []

        for i in range(train_size - self._roll_window_size):
            x_list.append(np.concatenate((h_train[i:(i+self._roll_window_size)],d_train[i:(i+self._roll_window_size)])))
            y_list.append(np.concatenate(([h_train[i+self._roll_window_size]],[d_train[i+self._roll_window_size]])))

        self._X = np.stack(x_list)
        self._Y = np.stack(y_list)

        self._model.fit(self._X, self._Y)

        self._fitted = True

    def predict(self, days):
        pred_horizon = days
        pred = np.zeros((pred_horizon,2))
        Z = self._X[[-1],:]

        for i in range(pred_horizon):
            pred[i,:] = self._model.predict(Z)
            for j in range(2 * self._roll_window_size):
                if (j == self._roll_window_size - 1):  #28 = 2*14 where 14 is the moving-window length. Should make it L
                    Z[0,j] = pred[i,0]
                elif (j == 2 * self._roll_window_size - 1):
                    Z[0,j] = pred[i,1]
                else:
                    Z[0,j] = Z[0,j+1]

        pred[:,0] = self._max_h * pred[:,0]
        pred[:,1] = self._max_d * pred[:,1]

        return pred

    def predict_hospitalizations(self, days):
        assert self._fitted, "Model not yet fit"

        pred = self.predict(days)
        return pred[:,0]

    def predict_deaths(self, days):
        assert self._fitted, "Model not yet fit"

        pred = self.predict(days)
        return pred[:,1]
