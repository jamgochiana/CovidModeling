# utils.py
import pandas as pd
import numpy as np

def process_hospitalizations(df):
    """
    Process the hospitalization dataframe
    
    Args:
        df (pd.DataFrame): raw hospitalization dataframe
    Returns:
        h (pd.Series): processed hospitalizations series
    """
    
    hosp = df[df["Hospital"]=="All SF Hospitals"]
    hosp = hosp[["reportDate", "PatientCount"]].groupby("reportDate").sum()
    dates = pd.to_datetime(hosp.index.values)
    data = hosp.iloc[:,0].to_numpy()
    h = pd.Series(data, index = dates, name = "hospitalizations")
    return h

def process_cases(df):
    """
    Process the cases from the cases and deaths dataframe
    
    Args:
        df (pd.DataFrame): cases and deaths by cause dataframe
    Returns:
        c (pd.Series): processed cases series
    """    
    cases = df[df["Case Disposition"]=="Confirmed"]
    cases = cases[["Specimen Collection Date", 
                   "Case Count"]].groupby("Specimen Collection Date").sum()
    dates = pd.to_datetime(cases.index.values)
    data = cases.iloc[:,0].to_numpy()
    c = pd.Series(data, index = dates, name = "cases")
    return c
    
def process_deaths(df):
    """
    Process the deaths from the cases and deaths dataframe
    
    Args:
        df (pd.DataFrame): cases and deaths by cause dataframe
    Returns:
        d (pd.Series): processed cases series
    """    
    deaths = df[df["Case Disposition"]=="Death"]
    deaths = deaths[["Specimen Collection Date", 
                     "Case Count"]].groupby("Specimen Collection Date").sum()
    dates = pd.to_datetime(deaths.index.values)
    data = deaths.iloc[:,0].to_numpy()
    d = pd.Series(data, index = dates, name = "deaths")
    return d

def merge_data(series):
    """
    Merge dataframes into single df with proper dates and zeros
    
    Args:
        series (list of pd.Series): list of series to combine
    Returns:
        df (pd.DataFrame): processed single dataframe with mutual dates
    """
    start = max([min(s.index.values) for s in series])
    end = min([max(s.index.values) for s in series])
    combined = pd.concat(series, axis=1).fillna(0)
    dates = pd.date_range(start=start, end=end)
    df = combined.reindex(dates, fill_value=0)
    return df
