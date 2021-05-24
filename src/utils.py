# utils.py
import pandas as pd
import numpy as np

def read_csv(path):
    """
    Read csv of processed data and return data, columns, and dates.

    Args:
        path (string): path to data csv
    Returns:
        data (np.array): (n_times, n_vars)-sized array of COVID data
        dates (pd.Series): (n_times)-sized series of pd.datetimes
        columns (list of strings): list of column names

    """
    df = pd.read_csv(path)
    dates = pd.to_datetime(df.iloc[:,0].values)
    columns = list(df.columns.values[1:])
    data = df.iloc[:,1:].to_numpy()
    return data, dates, columns


def moving_average(data, dates, days=7):
    """
    Perform moving averaging over data.

    Args:
        data (np.array): (n_times, n_vars)-sized array of COVID data
        dates (pd.Series): (n_times)-sized series of pd.datetimes
        days (int): number of days to average over
    Returns:
        mva_data (np.array): (n_times-days+1, n_vars)-sized array of COVID data
        mva_dates (pd.Series): (n_times-days+1)-sized series of pd.datetimes
    """
    T, V = data.shape
    mva_dates = dates[(days-1):]
    stacked = np.stack([data[i:T-days+i+1] for i in range(days)])
    mva_data = stacked.mean(axis=0)
    return mva_data, mva_dates

def train_test_split(data, dates, test_days=7):
    """
    Split data into training and testing portions.

    Args:
        data (np.array): (n_times, n_vars)-sized array of COVID data
        test_days (int): number of days to reserve for testing
    Returns:
        train (np.array): (n_times-test_days, n_vars)-sized array of training COVID data
        test (np.array): (test_days, n_vars)-sized array of testing COVID data
        train_dates (pd.Series): (n_times-test_days)-sized series of pd.datetimes
        test_dates (pd.Series): (test_days)-sized series of pd.datetimes
    """
    T, V = data.shape
    train, train_dates = data[:(T-test_days)], dates[:(T-test_days)]
    test, test_dates = data[(T-test_days):], dates[(T-test_days):]
    return train, test, train_dates, test_dates

def train_test_split_multi(data, dates, train_days=42, test_days=7,
                           batch_size=25, seed=0):
    """
    Split data into training and testing portions.

    Args:
        data (np.array): (n_times, n_vars)-sized array of COVID data
        train_days (int): number of days to reserve for training
        test_days (int): number of days to reserve for testing
        batch_size (int): number of test sequences
        seed (int): random seed
    Returns:
        train (np.array): (batch_size, n_times-test_days, n_vars)-sized array of training COVID data
        test (np.array): (batch_size, test_days, n_vars)-sized array of testing COVID data
        train_dates (list of pd.Series): list of batch_size (n_times-test_days)-sized series of pd.datetimes
        test_dates (list of pd.Series): list of batch_size (test_days)-sized series of pd.datetimes
    """
    T, V = data.shape
    np.random.seed(0)
    train, test, train_dates, test_dates = [], [], [], []
    for _ in range(batch_size):
        start = np.random.randint(T-train_days-test_days)
        train.append(data[start:(start+train_days)])
        train_dates.append(dates[start:(start+train_days)])
        test.append(data[(start+train_days):(start+train_days+test_days)])
        test_dates.append(dates[(start+train_days):(start+train_days+test_days)])
    train = np.stack(train)
    test = np.stack(test)
    return train, test, train_dates, test_dates

# metrics

def rmse(pred, true):
    """
    Return RMSE between predicted and true trajectories.

    Args:
        pred (np.array): (B,n_times)-sized predicted data trajectory
        true (np.array): (B,n_times)-sized true data trajectory
    Returns:
        rmse (np.array): (B,) root mean squared error between the trajectories
    """
    assert pred.shape==true.shape, "Shape mismatch"
    rmse = ((pred-true)**2).mean(-1)**0.5
    return rmse

def mape(pred, true):
    """
    Return MAPE between predicted and true trajectories.

    Args:
        pred (np.array): (B,n_times)-sized predicted data trajectory
        true (np.array): (B,n_times)-sized true data trajectory
    Returns:
        mape (np.array): (B,) mean absolute percentage error between the trajectories (expressed as a decimal)
    """
    assert pred.shape==true.shape, "Shape mismatch"
    mape = abs((true-pred)/true).mean(-1)
    return mape

def mae(pred, true):
    """
    Return MAE between predicted and true trajectories.

    Args:
        pred (np.array): (B,n_times)-sized predicted data trajectory
        true (np.array): (B,n_times)-sized true data trajectory
    Returns:
        mape (np.array): (B,) mean absolute  error between the trajectories (expressed as a decimal)
    """
    assert pred.shape==true.shape, "Shape mismatch"
    mae = abs(true-pred).mean(-1)
    return mae
