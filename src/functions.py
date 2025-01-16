#from sklearn.metrics import *
#from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np


class ScalerWrapper:
    def __init__(self, scaler, columns):
        """
        Initialize the scaler wrapper with a specified scaler.
        """
        self.scaler = scaler 
        self.columns = columns

    def fit_transform(self, X, y = None):
        """
        Fit the scaler to the data and transform it, retaining column names.
        :param X: pd.DataFrame, input data
        :return: pd.DataFrame, scaled data with original column names
        """
        scaled_data = self.scaler.fit_transform(X)
        return pd.DataFrame(scaled_data, columns=self.columns, index=X.index)

    def transform(self, X):
        """
        Transform the data using a previously fitted scaler, retaining column names.
        :param X: pd.DataFrame, input data
        :return: pd.DataFrame, scaled data with original column names
        """
        scaled_data = self.scaler.transform(X)
        return pd.DataFrame(scaled_data, columns=self.columns, index=X.index)

    def fit(self, X):
        """
        Fit the scaler to the data.
        :param X: pd.DataFrame, input data
        """
        self.scaler.fit(X)
        return self

    def inverse_transform(self, X):
        """
        Inverse transform the scaled data to the original scale.
        :param X: pd.DataFrame, scaled data
        :return: pd.DataFrame, original data scale with column names
        """
        original_data = self.scaler.inverse_transform(X)
        return pd.DataFrame(original_data, columns=self.columns, index=X.index)


def plot_correlations(df):
    correlations = df.corr().loc['infected'].sort_values()
    correlations.drop('infected',inplace = True)
    sns.barplot(data = correlations)
    plt.xticks(rotation=90)
    plt.ylabel ("correlation")
    plt.title("Correlation with target 'infected'")

    plt.show()

def plot_mutual_info(dataframe):
    plt.figure()
    df = dataframe.copy()
    X = df.drop('infected', axis=1)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(X)
    X = pd.DataFrame(scaled_data, columns=X.columns)
    y = df['infected']
    mutual_info = mutual_info_regression(X, y)

    mutual_info = pd.Series(mutual_info)     
    mutual_info.index = X.columns
    mutual_info.sort_values(ascending=False)
    mutual_info.sort_values(ascending=False).plot.bar()
    plt.tight_layout()
    plt.show()

def hello():
    print("functions!")
