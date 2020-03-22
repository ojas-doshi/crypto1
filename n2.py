from pylab import *
import pandas as pd
import numpy as np



def Datapull(Stock):
    try:
        df = (pd.io.data.DataReader(Stock,'yahoo',start='01/01/2010'))
        return df
        print ('Retrieved', Stock)
        time.sleep(5)
    except e:
        print ('Main Loop', str(e))


def RSIfun(price, n=14):
    delta = price['Close'].diff()
    #-----------
    dUp= delta[delta > 0]
    dDown= delta[delta < 0]

    RolUp=pd.rolling_mean(dUp, n)
    RolDown=pd.rolling_mean(dDown, n).abs()

    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    return rsi

Stock='AAPL'
df=Datapull(Stock)
RSIfun(df)