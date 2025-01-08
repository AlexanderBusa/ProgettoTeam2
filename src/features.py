import numpy as np

def onehot_trt(dataset, drop = True):
    """
    OneHotEncoder for treatment options
    # trt0: zdv only
    # trt1: zdv + ddl
    # trt2: # zdd + zal
    # trt3 = 1-trt0-trt1-trt2: ddl only
    """
    if "trt" not in dataset.columns:
        print("Warning: the data does not have a feature 'treat' to be OneHotEncoded.")
        return dataset

    df = dataset.copy()
    df['trt0'] = (df['trt']==0).astype(int)
    df['trt1'] = (df['trt']==1).astype(int)
    df['trt2'] = (df['trt']==2).astype(int)
    if drop:
        df = df.drop(columns = ['trt', 'treat'])
    return df

def drop_str(dataset, drop = True):
    df = dataset.copy()
    if drop:
        df = df.drop(columns = 'str2 z30 preanti'.split())
    return df

def labratio(dataset, drop = False):
    df = dataset.copy()
    # interpretation:
    ## ratio = 1 ----> no change
    ## ratio = 2 ----> lab values were double at the beginning (worsening situation)
    ## ratio = 0.5 ----> lab values were half at the beginning (improving situation)
    ## ratio = 0 ----> lab values were critical at the beginning
    # we are not doing cd420 / cd40 because sometimes cd40 is zero
    df['cd4ratio'] = df['cd40'] / df['cd420']
    df['cd8ratio'] = df['cd80'] / df['cd820']
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

def offtrtinteraction(dataset):

    df = dataset.copy()

    for feature in "hemo karnof race cd80 age":
        if feature in dataset.columns:
            df[feature+"_off"] = df[feature]*df['offtrt']

    for feature in "trt1 oprior strat gender homo":
        if feature in dataset.columns:
            df[feature+"_on"] = df[feature]*(1-df['offtrt'])

    return df

def timeofftrt(dataset, drop_offtrt = True, drop_time = True):
    """ 
    replaces feature "time" and "offtrt" with four variables 
    - time2_off, time2_off, time2_off, time2_off
    which depend on how the patient follows the treatment in the first 101 weeks
    """

    if "time" not in dataset.columns:
        print("Warning: the data does not have a feature 'time' for timeofftrt feature engineering.")
        return dataset
    
    if "offtrt" not in dataset.columns:
        print("Warning: the data does not have a feature 'offtrt' for timeofftrt feature engineering.")
        return dataset
    
    df = dataset.copy()

    # indicator function of patients that have been treated for at least 101 weeks
    mask_time2 = (df['time'] > 750).astype(int)
    mask_time1 = (df['time'] <= 707).astype(int)


    df['time2_off'] =  df['time'] * mask_time2 * df['offtrt']
    df['time1_off'] =  df['time'] * mask_time1 * df['offtrt']
    df['time2_on'] =  df['time'] * mask_time2 * (1-df['offtrt'])
    df['time1_on'] = df['time'] * mask_time1 * (1-df['offtrt'])  
    
    df['bool_time2_off'] =  mask_time2 * df['offtrt']
    #df['bool_time1_off'] = (1-mask_time2) * df['offtrt']
    df['bool_time2_on'] =   mask_time2 * (1-df['offtrt'])
    df['bool_time1_on'] = mask_time1 * (1-df['offtrt']) 

    df['bool_timeexact_on'] = (1-mask_time2) *(1-mask_time1) * (1-df['offtrt']) 
    df['bool_timeexact_off'] = (1-mask_time2) *(1-mask_time1)* df['offtrt']


      
      # (*)

    # (*) for patients that have been treated for less than 101 weeks and have not gone off-treatment, 
    #     we know they are all infected, regardless of time, so there's no need to include the time dependence

    if drop_offtrt:
        df= df.drop(columns = ['offtrt'])

    if drop_time:
        df= df.drop(columns = ['time'])
    return df

def engineer(
        dataset, 
        strat = None,
        lab = None,
        time = "730",
        trt = "onehot",
        offtrt = None):
    """
    Applies the necessary feature engineering steps
    """
    df = dataset.copy()

    if strat == "drop":
        df = drop_str(df)

    if time == "730":
        df = time730(df)

    if time == "offtrt":
        df = timeofftrt(df)

    if trt == "onehot":
        df = onehot_trt(df)

    if lab == "ratio":
        df = labratio(df)

    if offtrt == "interact":
        df = offtrtinteraction(df)

    return df 