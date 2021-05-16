import sys, os
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from src import utils

# load data from processed csv
path_loc = './datasets/processed/sf.csv'
data, dates, columns = utils.read_csv(path_loc)
print("Data columns: ", columns)
print("Max Daily Deaths: ", data[:,2].max())

# without averaging

train, test, train_dates, test_dates = utils.train_test_split(data, dates)

plt.figure()
plt.title('SF Data w/o moving average')
plt.plot(train_dates, train)
plt.plot(test_dates, test)
plt.legend(columns)
plt.show()

# with 7-day moving average
mva_data, mva_dates = utils.moving_average(data, dates, days=7)
train, test, train_dates, test_dates = utils.train_test_split(mva_data, mva_dates)

plt.figure()
plt.title('SF Data w/ 7-day moving average')
plt.plot(train_dates, train)
plt.plot(test_dates, test)
plt.legend(columns)
plt.show()
