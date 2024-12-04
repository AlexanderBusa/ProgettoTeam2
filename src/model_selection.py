import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from matplotlib_venn import venn2
from tqdm import tqdm



def validation_accuracy_score(Xtrain,ytrain,model, n_splits = 5):
    # Create the K-Fold cross-validator
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation measuring the Accuracy
    scores = cross_val_score(
        model, Xtrain, ytrain, cv=cv, scoring='accuracy', n_jobs=-1
    )

    # Calculate the mean of the MSE scores
    mean_accuracy = scores.mean() 
    return mean_accuracy

def rfe_kfold_cv(X, y, max_k=None, cv=5, verbose=False):
    if max_k is None:
        max_k = X.shape[1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    results_rfe = []
    for k in range(1, max_k + 1):
        
        rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=k)
        rfe.fit(X_scaled, y)
        selected_features = X_scaled.columns[rfe.support_]
        
        model = LogisticRegression(random_state=42)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled[selected_features], y, cv=skf, scoring='accuracy')
        
        results_rfe.append({
            "k": k,
            "cv_mean_score": round(np.mean(cv_scores), 4),
            "features": list(selected_features)
        })
    
    return pd.DataFrame(results_rfe).set_index('k')


def rfe_evaluation_cv(dataset, max_k=None, cv=5):
    """ 
    returns : DataFrame with columns 
        k:: integer (number of features)
        score:: float (mean accuracy score in cross-validation)
        features:: list (names of selected features)
    """
    X,y = datasets.xy_train(dataset)

    r = rfe_kfold_cv(X,y,max_k,cv)
    r = r.rename(columns={'cv_mean_score': 'score'})
    return r

def skb_evaluation_cv(dataset = None, max_k= None, n_splits = 5):
    """ 
    returns : DataFrame with columns 
        k:: integer (number of features)
        score:: float (mean accuracy score in cross-validation)
        features:: list (names of selected features)
    """
    X,y = datasets.xy_train(dataset) 

    if max_k is None:
        max_k = len(X.columns)

    results_skb = []
    for k in range(1,max_k+1):
        skb_features = skb_select_features(X,y,k)
        model = LogisticRegression(random_state = 42)
        score = validation_accuracy_score(X[skb_features],y,model,n_splits = n_splits)
        results_skb.append({"k": k, "score": round(score,4), "features":list(skb_features)})
    return pd.DataFrame(results_skb).set_index('k')

