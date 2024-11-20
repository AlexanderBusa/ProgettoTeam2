#from sklearn.metrics import *
#from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np


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
