# pmodel.py

import numpy as np
from src.pmodel import PredictionModel
from src.models.tuned_estimators import estimate_params4
from src.models.gamma_predictions_fixed import gamma4_populate_V_U


class GAMMA(PredictionModel):

    def __init__(self, model_type, delta1, delta2, delta3, p, num_past_days):
        """
        Initialize prediction model parameters

        """
        self._fitted = False # change to True once model is fit
        self._model_type = model_type #we will add the option to switch between different ways to evaluate gammas

        self._delta1 = delta1 #days it takes to go to hospital
        self._delta2 = delta2 #days it takes to die
        self._delta3 = delta3 #days you are infected
        self._p = p #prob of daying
        self.num_past_days = num_past_days # number of days for which gammas should be const ish
        self.populate_days = 100 #number of days into the future we valuate U, V

    def fit(self, x):
        """
        Fit model parameters using data.
        Args:
            x (np.array): (ntimes, 3) data
        """

        self._t = x.shape[0] #Length of the dataset
        self._V = np.zeros(self._t+self.populate_days) #Vulnerable ind.
        self._U = np.zeros(self._t+self.populate_days) #Non-vulnerable ind.

        #Dataset processing.
        self._V[:self._t], self._U[:self._t] = self.data_preprocessing(x)

        #Evaluate the gammas used in the predictions.
        gammas = self.evaluate_gammas(self._V, self._U)

        #Uses the gammas to populate future V, U values given the dynamics models
        self._V, self._U = gamma4_populate_V_U(self._V, self._U, gammas, self._t - self._delta2 - 1)

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

        gm = estimate_params4(v, u)

        return gm.flatten()


    def predict_cases(self, days):
        """
        Use model to .
        Args:
            days (int): number of days forward to predict
        Returns:
            pred (np.array): (days,)-shaped array of case predictions
        """

        print('Cases prediction not implemented')
        return None

    def predict_hospitalizations(self, days):
        """
        Fit model to data.
        Args:
            days (int): number of days forward to predict
        Returns:
            pred (np.array): (days,)-shaped array of hospitalization predictions
        """

        if self._fitted:
            hospitalizations = self._V[self._t - self._delta1:self._t - self._delta1 + days]
            return hospitalizations
        else:
            print('Model is not fitted!')
            return None

    def predict_deaths(self, days):
        """
        Fit model to data.
        Args:
            days (int): number of days forward to predict
        Returns:
            pred (np.array): (days,)-shaped array of death predictions
        """

        if self._fitted:
            nb = self._V[self._t - self._delta2:self._t - self._delta2 + days] + self._U[self._t - self._delta2:self._t - self._delta2 + days]
            deaths = nb * self._p
            return deaths
        else:
            print('Model is not fitted!')
            return None

    def data_preprocessing(self, train):
        """
        Processes the dataset.
        Args:
            train (np.array): (ntimes, 3) data
        Returns:
            V (np.array): (ntimes,)  data of vulnerable individuals
            U (np.array): (ntimes,)  data of non-vulnerable individuals
        """

        V = np.zeros(len(train))
        U = np.zeros(len(train))
        cases_so_far = np.zeros(len(train))

        # Estimate type V active cases for each day and total cases (U + V).
        for t in range(self._delta2, len(train)):
            if t == self._delta2:
                cases_so_far[t - self._delta2] = train[t, 2] / self._p
            else:
                cases_so_far[t - self._delta2] = cases_so_far[t - self._delta2 - 1] + train[t, 2] / self._p

            V[t - self._delta1] = train[t, 1]

        # Estimate type U active cases for each day.
        for t in range(self._delta3 + self._delta2, len(train)):
            U[t - self._delta2] = cases_so_far[t - self._delta2] - cases_so_far[t - self._delta2 - self._delta3] - V[t - self._delta2]

        return V, U
