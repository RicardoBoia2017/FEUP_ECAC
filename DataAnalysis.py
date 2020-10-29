import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from Preprocess import *

def loan_per_year(df, isMerged = True):
    if isMerged:
        df['year'] = df['loan_date'].dt.year
    else:
        df = df_transformDate(df, 'date')
        df['year'] = df['date'].dt.year

    df['year'] = df['loan_date'].dt.year if isMerged else df['date'].dt.year
    df_years = {}
    for i in df['year'].unique():
        df_years[i] = df[df['year'] == i]
        df_years[i].drop(['year'], axis = 1, inplace = True)
  
    return df_years


def getCorr(df, size=(11,9)):
    corr = df.corr()
    ax = plt.subplots(figsize=size)
    ax = sns.heatmap(corr,  annot = True, linewidths=.1, mask = np.triu(corr), cmap='coolwarm')
    plt.show()

def histDf(df_years):
    for year in df_years:
        print("\n\nLoans of the year {}".format(year))
        df = df_years[year]
        df['month'] = df['date'].dt.month_name().str.slice(stop=3)
        loans_succ = df[df['status']==1]
        loans_unsucc = df[df['status']==-1]
        
        features = ['month','amount', 'duration', 'payments']
        bins = {'month': 13, 'amount': 20, 'duration': 10, 'payments': 20}
        
        for f in features:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            loans_succ[f].hist(bins=bins[f], ax=ax1, label='sucessful', color='green', alpha=0.6)
            loans_unsucc[f].hist(bins=bins[f], ax=ax2, label='unsucessful', color='red', alpha=0.6)
            plt.show()
        print()