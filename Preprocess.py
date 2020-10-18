import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

# Read csv

account = pd.read_csv('data/account.csv', delimiter=";")
card_test = pd.read_csv('data/card_test.csv', delimiter=";")
card_train = pd.read_csv('data/card_train.csv', delimiter=";")
client = pd.read_csv('data/client.csv', delimiter=";")
disp = pd.read_csv('data/disp.csv', delimiter=";")
district = pd.read_csv('data/district.csv', delimiter=";")
loan_test = pd.read_csv('data/loan_test.csv', delimiter=";")
loan_train = pd.read_csv('data/loan_train.csv', delimiter=";")
trans_test = pd.read_csv('data/trans_test.csv', delimiter=";")
trans_train = pd.read_csv('data/trans_train.csv', delimiter=";")

def renameColumns():
    account.rename({'date': 'account_date', 'district_id': 'account_district_id'}, axis=1, inplace = True)
    client.rename({'district_id': 'client_district_id'}, axis=1, inplace = True)
    loan_train.rename({'date': 'loan_date'}, axis=1, inplace = True)
    loan_test.rename({'date': 'loan_date'}, axis=1, inplace = True)

def main():
    renameColumns()
    print(account.select_dtypes(include=['object']).nunique())

if __name__ == '__main__':
    main()