#from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.preprocessing import RobustScaler

import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

def data_original():
    data = pd.read_csv('./data/AIDS_Classification.csv')
    return data

def data_50000():
    data = pd.read_csv('./data/AIDS_Classification_50000.csv')
    return data

def _sample_balance_df(dataset, with_reindex = False, seed = 42):
    # copy the dataset and divide into infected and not infected
    df = dataset.copy()
    df_infected = df[df['infected']==1] 
    df_not_infected = df[df['infected']==0]

    # sample the same number of rows from the not infected dataframe as there are infected ones.
    num_infected = df_infected.shape[0]
    df_sample_not_infected = df_not_infected.sample(num_infected, random_state=seed)

    df_balanced = pd.concat([df_infected,df_sample_not_infected])
    # we shuffle it once more at the end.
    # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    if with_reindex: # shuffle and reindex
        df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)  
    else: # just shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=seed)  
    return df_balanced


def data_balanced():
    data = data_original()
    return _sample_balance_df(data)

def data_time730(df = data_original(), drop_censored = False):
    """ 
    replaces feature "time" with 
    - time730: counts the days over the 2 years time target 
    - time_censored: counts the censored days up to the 2 years time target
    """

    if "time" not in df.columns:
        print("Warning: the data does not have a feature 'time' to be replaced with time730.")
        return df

    # this shifted ReLU does the job:
    df["time730"] = np.maximum(0, df['time'] - 730)

    # censored patients: they quit the study before 2 years
    if not drop_censored:
        df["time_censored"] = np.maximum(0,730 - df['time'])

    df= df.drop(columns = ['time'])
    return df

def xy_train(dataset=None, scale = True):
    if dataset is None:
        dataset = data_original()
    X, y = dataset.drop('infected', axis=1), dataset['infected']
    X_train, _, y_train , _ = train_test_split(X,y,test_size = 0.2, random_state=42, stratify = y)

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)

    return X_train, y_train

def xy_train_test(dataset = None, scale = True):
    if dataset is None:
        dataset = data_original()
    X, y = dataset.drop('infected', axis=1), dataset['infected']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  
        X_test_scaled = scaler.transform(X_test) 
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)


    return X_train, X_test, y_train, y_test


