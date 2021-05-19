### main.py

import sys
sys.path.append('./')

import numpy as np
from src.models.arma import ARMA
from src.models.seird import SEIRD
from src.models.andrei import ANDREI
from src import utils

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
    elif model_name == 'andrei':
        model = ANDREI()
    else:
        raise('Invalid model type:', model)
    
    # load datafile
    data_dir = './datasets/processed/'
    data, dates, columns = utils.read_csv(data_dir+data_file)
    
    # apply moving average
    if mva:
        data, dates = utils.moving_average(data, dates, days=7)
    
    # split up data into batch_size train+test sequences
    
    
    c_preds = []
    h_preds = []
    d_preds = []
    # for each sequence
    
        # initialize new model
        
        # fit model
        model.fit(s)
        
        # predict days
        
        c_preds.append(model.predict_cases(p_days))
        h_preds.append(model.predict_hospitalizations(p_days))
        d_preds.append(model.predict_deaths(p_days))
    
    # evaluate metrics
    print("Model:", model_name)
    for (true, pred, name) in [(c_preds, c_true, "Cases"), 
                               (h_preds, h_true, "Hospitalizations"), 
                               (d_preds, d_true, "Deaths")]:
        # skip if no prediction
        if pred[0] is None:
            print("%s: no predictions" %(name))
            continue
    
        # batch predictions
        pred_batch = np.stack(pred)
        
        # run metrics on batches
        rmses = utils.rmse(true, pred)
        maes = utils.mae(true, pred)
        mapes = utils.mape(true, pred)   
        
        # report mean and std around mean (std / sqrt(B))
        for (metric, name) in [(rmses,"RMSE"),
                               (maes,"MAE"),
                               (mapes,"MAPE")]:
            print('%s: %f \pm %f'%(name, metric.mean(), metric.std()/(len(metric)**0.5)))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Covid Modeling Experiment.')
    parser.add_argument('--model', choices=['arma', 'seird', 'andrei'], 
                        default='arma', help='model')
    parser.add_argument('--data', type=str, required=True
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
    
    
    
    

