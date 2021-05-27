# pmodel.py

import numpy as np
from src.pmodel import PredictionModel
from src.models.tuned_estimators import estimate_params2
from src.models.gamma_predictions_fixed import gamma2_populate_V_U
from src.models.gamma import GAMMA

class GAMMA_2(GAMMA):

    def fit(self, x):
        """
        Fit model parameters using data.
        Args:
            x (np.array): (ntimes, 3) data
        """

        self._t = x.shape[0] #Length of the dataset
        self._V = np.zeros(self._t+self.populate_days) #Vulnerable individuals.
        self._U = np.zeros(self._t+self.populate_days) #Non-vulnerable individuals.

        #Dataset processing.
        self._V[:self._t], self._U[:self._t] = self.data_preprocessing(x)

        #Evaluate the gammas used in the predictions.
        gammas = self.evaluate_gammas(self._V, self._U)

        #Uses the gammas to populate future V, U values given the dynamics models
        self._V, self._U = gamma2_populate_V_U(self._V, self._U, gammas, self._t - self._delta2 - 1)

        self._fitted = True

    def evaluate_gammas(self, V, U):
        """
        Get gammas for the period of interest. I.e., we take the last period of num_past_days
        for which we know both U and V. We use them to get the gammas. The gammas will
        remain constant for all next predictions (from self._t -self._delta2 onwards).

        Args:
            V (np.array): (ntimes,)  data of vulnerable individuals
            U (np.array): (ntimes,)  data of non-vulnerable individuals
        Returns:
            gm (np.array): (4,)  gammas
        """

        u = U[self._t - self.num_past_days - self._delta2:self._t - self._delta2]
        v = V[self._t - self.num_past_days - self._delta2:self._t - self._delta2]

        gm = estimate_params2(v, u)

        return gm.flatten()
