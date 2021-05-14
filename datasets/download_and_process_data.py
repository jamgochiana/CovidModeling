# download_and_process_data.py
import sys, os
import requests
sys.path.append('./')
import pandas as pd
import numpy as np
import datasets.process_utils.utils

# urls
hospitalizations = "https://data.sfgov.org/api/views/nxjg-bhem/rows.csv" #?accessType=DOWNLOAD
cases_deaths = "https://data.sfgov.org/api/views/tvq9-ec9w/rows.csv" #?accessType=DOWNLOAD

# download path
pathdir = './datasets/raw/'
if not os.path.exists(pathdir):
    os.mkdir(pathdir)

# hospitalizations
if not os.path.exists(pathdir+'hospitalizations.csv'):
    req = requests.get(hospitalizations)
    url_content = req.content
    csv_file = open(pathdir+'hospitalizations.csv', 'wb')
    csv_file.write(url_content)
    csv_file.close()

# cases and deaths
if not os.path.exists(pathdir+'cases_deaths.csv'):
    req = requests.get(cases_deaths)
    url_content = req.content
    csv_file = open(pathdir+'cases_deaths.csv', 'wb')
    csv_file.write(url_content)
    csv_file.close()

    
# process data
hosp = pd.read_csv(pathdir+'hospitalizations.csv')
c_d = pd.read_csv(pathdir+'cases_deaths.csv')

h = utils.process_hospitalizations(hosp)
c, d = utils.process_cases_deaths(c_d)
x = np.stack((c, h, d))

# save out
processdir = './datasets/processed/'
if not os.path.exists(processdir):
    os.mkdir(processdir)
np.savetxt(processdir+'sf.csv', x, delimiter=',')
    