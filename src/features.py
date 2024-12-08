import numpy as np

def onehot_trt(dataset, drop = True):
    """
    OneHotEncoder for treatment options
    # trt0: zdv only
    # trt1: zdv + ddl
    # trt2: # zdd + zal
    # trt3 = 1-trt0-trt1-trt2: ddl only
    """
    if "treat" not in dataset.columns:
        print("Warning: the data does not have a feature 'treat' to be OneHotEncoded.")
        return dataset

    df = dataset.copy()
    df['trt0'] = (df['treat']==0).astype(int)
    df['trt1'] = (df['treat']==1).astype(int)
    df['trt2'] = (df['treat']==2).astype(int)
    if drop:
        df = df.drop(columns = ['trt', 'treat'])
    return df

def drop_str(dataset, drop = True):
    df = dataset.copy()
    if drop:
        df = df.drop(columns = 'str2 z30 preanti'.split())
    return df

def time730(dataset, drop_censored = False):
    """ 
    replaces feature "time" with 
    - time730: counts the days over the 2 years time target 
    - time_censored: counts the censored days up to the 2 years time target
    """
    if "time" not in dataset.columns:
        print("Warning: the data does not have a feature 'time' to be replaced with time730.")
        return dataset
    
    df = dataset.copy()

    # this shifted ReLU does the job:
    df["time730"] = np.maximum(0, df['time'] - 730)

    # censored patients: they quit the study before 2 years
    if not drop_censored:
        df["time_censored"] = np.maximum(0,730 - df['time'])

    df= df.drop(columns = ['time'])
    return df

def engineer(
        dataset, 
        strat = None,
        time = "730",
        trt = "onehot"):
    """
    Applies the necessary feature engineering steps
    """
    df = dataset.copy()

    if strat == "drop":
        df = drop_str(df)

    if time == "730":
        df = time730(df)

    if trt == "onehot":
        df = onehot_trt(df)

    return df 