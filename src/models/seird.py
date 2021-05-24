# sird.py
import sys
sys.path.append('./src/models/seird_src/CEEM')

from src.pmodel import PredictionModel

import numpy as np
import torch
from torch import nn
from ceem.dynamics import C2DSystem, AnalyticObsJacMixin, DynJacMixin
from ceem.opt_criteria import *
from ceem.ceem import CEEM

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()

class SEIRD(PredictionModel):

    def __init__(self):
        """
        Initialize prediction model parameters

        """
        self._fitted = False # change to True once model is fit
        self._system = None

    def fit(self, x):
        """
        Fit model parameters using data.
        Data order: cases, hospitalizations, deaths.
        Not using hospitalizations.

        TO DO: Check dtype is correct.

        Args:
            x (np.array): (ntimes, 3) data
        """

        #Initialize parameters.
        T = x.shape[0]
        C0 = torch.from_numpy(x[:,0]).reshape(1,T,1) #confirmed cases
        S0 = torch.ones_like(C0, dtype=dtype) * C0.max() * 100 # susceptible
        I0 = C0 #infected
        D0 = torch.tensor(x[:,2], dtype=dtype).view(1,T,1) #deaths
        R0 = 0.5 * (C0 + D0) #recovered
        xsm = torch.cat([S0,I0,R0,D0,C0],dim=-1).log().detach()

        B = 1 #batches I think
        t = torch.arange(T, dtype=dtype).view(B,T).detach()
        y = torch.stack([C0, D0], dim=-1).log().view(B,T,2).detach()

        self._system = CompartmentalSystem(beta=0.1, gamma=0.04, mu=0.004)

        # specify criteria for smoothing and learning
        wstd = 0.33
        ystd = 1.0

        Sig_v_inv = torch.ones(2) / (ystd ** 2)
        Sig_w_inv = torch.ones(5) / (wstd ** 2)

        smoothing_criteria = []
        for b in range(B):
            # note: in this example B=1, but this is what we would do if it weren't
            obscrit = GaussianObservationCriterion(Sig_v_inv, t[b:b+1], y[b:b+1]) # pass in the inverse covariance
            dyncrit = GaussianDynamicsCriterion(Sig_w_inv, t[b:b+1])
            smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

        # specify solver kwargs
        smooth_solver_kwargs = {'verbose': 0, # to supress NLS printouts
                                'tr_rho': 0.1, # trust region for smoothing
                                'ftol':1e-5, # solve smoothing to coarse tolerance
                                'gtol':1e-5,
                                'xtol':1e-5
                               }

        # specify learning criteria
        # note: learning criteria do not need to be specified separately for each b in B
        dyncrit = GaussianDynamicsCriterion(Sig_w_inv, t)
        learning_criteria = [dyncrit]# since the observation objective doesnt depend on parameters
        learning_params = [list(self._system.parameters())] # the parameters we want optimized
        learning_opts = ['scipy_minimize'] # the optimzer
        learner_opt_kwargs = {'method': 'Nelder-Mead', # optimizer for learning
                              'tr_rho': 0.1 # trust region for learning
                             }

        # instantiate CEEM and train
        ceem = CEEM(smoothing_criteria, learning_criteria, learning_params, learning_opts,
                        [], lambda x: False)

        ceem.train(xs=xsm, sys=self._system, nepochs=100, smooth_solver_kwargs=smooth_solver_kwargs,
                   learner_opt_kwargs=learner_opt_kwargs)

        self._xsm = xsm
        self._fitted = True

    def predict_cases_and_deaths(self, days):
        """
        Use model to predict cases and deaths for days steps.
        Args:
            days (int): number of days forward to predict
        Returns:
            C (np.array): (days,)-shaped array of case predictions
            D (np.array): (days,)-shaped array of death predictions
        """

        assert self._fitted == True, "Model is not fitted"

        S, I, R, D, C = [], [], [], [], []

        #Intializes with the last day from the training set.
        x = self._xsm[:,-1,:].reshape(1,1,5) #Change it

        #Receives and feeds recurently predictions for days steps.
        # TO DO: It doesn't perform as well as expected. Unless, I made a typo.
        for day in range(days):
            x = self._system.step(torch.Tensor([day]).reshape(1),x)
            data = x.exp().detach().numpy()
            S.append(data[0,:,0])
            I.append(data[0,:,1])
            R.append(data[0,:,2])
            D.append(data[0,:,3])
            C.append(data[0,:,4])

        C = np.array(C)[:,0]
        D = np.array(D)[:,0]

        return C, D


    def predict_cases(self, days):
        """
        Use model to predict cases for days steps.
        Args:
            days (int): number of days forward to predict
        Returns:
            C (np.array): (days,)-shaped array of case predictions
        """

        C, _ = self.predict_cases_and_deaths(days)

        return C

    def predict_hospitalizations(self, days):
        """
        Fit model to data.
        Args:
            days (int): number of days forward to predict
        Returns:
            pred (np.array): (days,)-shaped array of hospitalization predictions
        """
        print('Hospitalization prediction not implemented')
        return None

    def predict_deaths(self, days):
        """
        Use model to predict deaths for days steps.
        Args:
            days (int): number of days forward to predict
        Returns:
            D (np.array): (days,)-shaped array of death predictions
        """

        _, D = self.predict_cases_and_deaths(days)

        return D

class CompartmentalSystem(C2DSystem, nn.Module, AnalyticObsJacMixin, DynJacMixin):
    """
    System predicting log of confirmed cases and deaths.
    """

    def __init__(self, beta, gamma, mu):

        C2DSystem.__init__(self, dt=1.0, # integration time-step
                                 method='midpoint' # integration scheme
                          )
        nn.Module.__init__(self)

        self.logbeta = nn.Parameter(torch.tensor(beta).log())
        self.loggamma = nn.Parameter(torch.tensor(gamma).log())
        self.logmu = nn.Parameter(torch.tensor(mu).log())

        self._xdim = 5 # log [S, I, R, D, C]
        self._ydim = 2

    def step_derivs(self, t, x, u=None):

        beta, gamma, mu = (self.logbeta.exp(), self.loggamma.exp(), self.logmu.exp())
        x = x.exp()

        S = x[...,0:1]
        I = x[...,1:2]
        R = x[...,2:3]
        D = x[...,3:4]

        N = S+I+R+D

        dSdt = -beta*S*I/N
        dIdt = -dSdt - (gamma+mu)*I
        dRdt = gamma * I
        dDdt = mu * I
        dCdt = -dSdt

        dxdt_ = torch.cat([dSdt, dIdt, dRdt, dDdt, dCdt], dim=-1)
        dxdt = dxdt_ / x

        return dxdt

    def observe(self, t, x, u=None):

        return x[...,[4,3]]

    def jac_obs_x(self, t, x, u=None):

        B, T, n = x.shape

        return torch.eye(5)[[4,3],:].expand(B, T, 2, 5)
