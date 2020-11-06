import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import math
from Preprocess import *

def loan_per_year(df, haveCategorical = True, isMerged = True):
    if isMerged and haveCategorical:
        df['loan_year'] = df['loan_date'].dt.year
    elif isMerged and not haveCategorical:
        df_date = df.copy()
        df_transformDate(df_date, 'loan_date')
        df['year'] = df_date['loan_date'].dt.year
    else:
        df_transformDate(df, 'date')
        df['year'] = df['date'].dt.year

    df_years = {}
    for i in df['year'].unique():
        df_years[i] = df[df['year'] == i]
        df_years[i].drop(['year'], axis = 1, inplace = True)
  
    return df_years

def getDescribe(df_var):
    for i in df_var:
        print(i)
        print("\n")
        if (i == 'client'):
            age = getAge(df_var[i])
            gender = getGender(df_var[i])
            df = pd.merge(age, gender, on="client_id")
            print(df.describe())
        else:
            print (df_var[i].describe())
        print("\n\n")

def getCorr(df, size=(11,9)):
    corr = df.corr()
    ax = plt.subplots(figsize=size)
    ax = sns.heatmap(corr,  annot = True, linewidths=.1, mask = np.triu(corr), cmap='coolwarm', size=2.5)
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

def main_finding1(df):

    g = sns.catplot(x="status", kind="count", palette=sns.color_palette(['red','blue']), data=df)
    
    (g.set_axis_labels("", "Count").set_xticklabels(["Unsuccessful", "Successful"]).set_titles("{col_name} {col_var}").despine(left=True))

def main_finding2(df):
    df = getGender(df)
    g = sns.catplot(x="gender", kind="count", palette=sns.color_palette(['pink','blue']), data=df)
    
    (g.set_axis_labels("", "Count").set_xticklabels(["Female", "Male"]).set_titles("{col_name} {col_var}").despine(left=True))
   

def main_finding3(df):
    df = getAge(df)
    def truncage(x):
        return math.trunc(x/10)*10
    df['age'] = df['age'].apply(truncage)
    g = sns.catplot(x="age", kind="count", data=df)
    
    (g.set_axis_labels("", "Count").set_xticklabels(["10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s"]).set_titles("{col_name} {col_var}").despine(left=True))

def main_finding4(df):
    df_4 = df[['region', 'client_id']]
    df_4.groupby(['region']).sum().plot(kind='bar', subplots=True, shadow = True, startangle=90, figsize=(15,10), autopct='%1.1f%%')

def main_finding5(df):
    client = df_var['client']
    a = getAge(client)
    ax = a['age'].plot.hist(bins=12, alpha=0.5)

def execFinding():
    main_finding3(df_var['client'])

    main_finding4(pd.merge(df_var['client'], df_var['district'], left_on = 'district_id', right_on='code'))