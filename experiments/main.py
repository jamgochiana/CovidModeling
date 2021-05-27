### main.py

import sys
sys.path.append('./')

import numpy as np
from src.models.arma import ARMA
from src.models.seird import SEIRD
from src.models.gamma import GAMMA
from src.models.gamma_l1 import GAMMA_L1
from src.models.gamma_2 import GAMMA_2
from src import utils

import matplotlib.pyplot as plt

def main(model_name, data_file, i_days, mva, batch_size, p_days):
    """
    Args:
        model_name (string): model name
        data_file (string): data_file
        i_days (int): number of input days per test sequence
        mva (bool): whether to apply 7-day moving average
        batch_size (int): number of test sequences
        p_days (int): number of prediction days per test sequence
    """

    # initialize model
    if model_name == 'arma':
        model = ARMA()
    elif model_name == 'seird':
        model = SEIRD()
    elif model_name == 'gamma':
        model_type = 'default' #There is no other model for now.
        delta1 = 11
        delta2 = 18
        delta3 = 14
        p = 0.02
        num_past_days = 7
        model = GAMMA(model_type, delta1, delta2, delta3, p, num_past_days) #Config file needs to be added.
    elif model_name == 'gamma_l1':
        model_type = 'default' #There is no other model for now.
        delta1 = 11
        delta2 = 18
        delta3 = 14
        p = 0.02
        num_past_days = 7
        lbd = 1000
        model = GAMMA_L1(model_type, delta1, delta2, delta3, p, num_past_days, lbd) #Config file needs to be added.
    elif model_name == 'gamma_2':
        model_type = 'default' #There is no other model for now.
        delta1 = 11
        delta2 = 18
        delta3 = 14
        p = 0.02
        num_past_days = 7
        lbd = 1000
        model = GAMMA_2(model_type, delta1, delta2, delta3, p, num_past_days) #Config file needs to be added.


    else:
        raise('Invalid model type:', model)

    # load datafile
    data_dir = './datasets/processed/'
    data, dates, columns = utils.read_csv(data_dir+data_file)

    # apply moving average
    if mva:
        data, dates = utils.moving_average(data, dates, days=7)

    # split up data into batch_size train+test sequences
    datasplit = utils.train_test_split_multi(data, dates, train_days=i_days,
                                             test_days=p_days, batch_size=batch_size,
                                             seed=0)
    train, test, train_dates, test_dates = datasplit

    B = train.shape[0] #Arec: Is B batch size?

    # fit and predict on each sequence
    c_preds, h_preds, d_preds = [], [], []
    for i in range(B):

        # refit model with new sequence
        model.fit(train[i])

        # predict days

        c_preds.append(model.predict_cases(p_days))
        h_preds.append(model.predict_hospitalizations(p_days))
        d_preds.append(model.predict_deaths(p_days))

    # evaluate metrics
    print("Model:", model_name)

    c_true, h_true, d_true = test[...,0], test[...,1], test[...,2]

    for (pred, true, name) in [(c_preds, c_true, "Cases"),
                               (h_preds, h_true, "Hospitalizations"),
                               (d_preds, d_true, "Deaths")]:

        print(f'{name}...')
        # skip if no prediction
        if pred[0] is None:
            print("%s: no predictions" %(name))
            continue

        # batch predictions
        pred_batch = np.stack(pred)

        # run metrics on batches
        rmses = utils.rmse(true, pred_batch)
        maes = utils.mae(true, pred_batch)
        mapes = utils.mape(true, pred_batch)

        # report mean and std around mean (std / sqrt(B))
        for (metric, name) in [(rmses,"RMSE"),
                               (maes,"MAE"),
                               (mapes,"MAPE")]:
            print('%s: %f \pm %f'%(name, metric.mean(), metric.std()/(len(metric)**0.5)))

    #plotting
    utils.plotting(train, test, c_preds, h_preds, d_preds, model_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Covid Modeling Experiment.')
    parser.add_argument('--model', choices=['arma', 'seird', 'gamma', 'gamma_l1', 'gamma_2'],
                        default='arma', help='model')
    parser.add_argument('--data', type=str, required=True,
                        help='datafile (with .csv, no path)')
    parser.add_argument('--i', default=42, type=int,
                       help='number of input days (default 42)')
    parser.add_argument('--p', default=7, type=int,
                       help='number of prediction days (default 7)')
    parser.add_argument('--B', default=25, type=int,
                       help='number of testing sequences (default 25)')
    parser.add_argument('--mva', type=bool, default=True,
                       help='Whether to apply moving average')
    args = parser.parse_args()

    main(args.model, args.data, args.i, args.mva, args.B, args.p)
