# pmodel.py

class PredictionModel:

    def __init__(self):
        """
        Initialize prediction model parameters

        """
        self._fitted = False # change to True once model is fit
        
    def fit(self, x):
        """
        Fit model parameters using data.
        Args:
            x (torch.tensor): (ntimes, 3) data
        """
        raise NotImplementedError
        
    def predict_cases(self, days):
        """
        Use model to .
        Args:
            days (int): number of days forward to predict 
        Returns:
            pred (np.array): (days,)-shaped array of case predictions
        """
        print('Case prediction not implemented')
        return None
        
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
        Fit model to data.
        Args:
            days (int): number of days forward to predict 
        Returns:
            pred (np.array): (days,)-shaped array of death predictions
        """
        print('Death prediction not implemented')
        return None