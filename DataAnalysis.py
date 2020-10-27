import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

def loan_per_year(df):
    df['year'] = df['loan_date'].dt.year
    df_years = {}
    for i in df['year'].unique():
        df_years[i] = df[df['year'] == i]
        df_years[i].drop(['year'], axis = 1, inplace = True)
  
    return df_years