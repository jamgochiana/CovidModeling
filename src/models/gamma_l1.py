# pmodel.py

import numpy as np
from src.pmodel import PredictionModel
from src.models.tuned_estimators import l1_estimate_params4
from src.models.gamma_predictions_fixed import gamma4_populate_V_U
from src.models.gamma import GAMMA


class GAMMA_L1(GAMMA):


    def __init__(self, model_type, delta1, delta2, delta3, p, num_past_days, lbd):
        """
        Initialize prediction model parameters

        """
        super().__init__(model_type, delta1, delta2, delta3, p, num_past_days)
        self._lbd = lbd

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


        Comments:
            -2 ?????
        """

        u = U[self._t - self.num_past_days - self._delta2:self._t - self._delta2]
        v = V[self._t - self.num_past_days - self._delta2:self._t - self._delta2]

        gm = l1_estimate_params4(v, u, self._lbd)

        return gm[self.num_past_days - 2].flatten()
