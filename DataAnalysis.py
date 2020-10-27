import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

def loan_per_year(df):
    years = np.array(range(93,99))
    df_years = []
    # for i in years:
    #     df_years[i] = 