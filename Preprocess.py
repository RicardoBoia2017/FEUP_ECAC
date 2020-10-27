import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from os import walk, path
from datetime import date

# Read csv

#dic_csv = {}
#df_var = {}

def read_data(isProcessed = False):
    dic_csv = {}
    for root,d_names,f_names in walk("./data/"):
        for f in f_names:
            filename = f.rsplit('.', 1)[0]
            dic_csv[path.join(root, f)] = ('processed_' + filename) if isProcessed else filename
    return dic_csv

def make_dataframe(dic):
    df_var = {}
    for key in dic:
        df_var[dic[key]] = pd.read_csv(key, delimiter=";")
    return df_var

def separate(df, column):
    list = df[column].unique().tolist()
    dic = {}
    for l in list:
        dic[l] = df[df[column]==l]
    return dic

def concat_train_test(dic, df):
    lis = [d for d in list(dic.values()) if ('test' in d or 'train' in d)]
    train = []
    test = []
    names = []
    for l in lis:
        names.append(l.split("_")[0])
        if (l.split("_")[1] == 'train'):
            train.append(l)
        else:
            test.append(l)
    names = list(set(names))
    
    for n in names:
        t1 = [d for d in train if n in d][0]
        t2 = [d for d in test if n in d][0]
        
        df[n] = pd.concat([df[t1],df[t2]])
    

def getDate(d):
    year = 1900+int(str(d)[0:2])
    month = int(str(d)[2:4])
    gender = 'M' if (month < 50) else 'F'
    month = month if (month < 50) else month - 50
    day = int(str(d)[4:6])
    return {'year': year, 'month': month, 'day': day, 'gender': gender}

def transformDate(d):
    d = getDate(d)
    return str(d['year'])+'-'+str(d['month'])+'-'+str(d['day'])
    
def getGender(df):
    list = []
    for row in df.itertuples(index = True):
        d = getDate(row.birth_number)
        birth_number = row.birth_number 
        if (d['gender'] == 'F'):
            birth_number -= 5000
        list.append([row.client_id, birth_number, d['gender'], row.district_id])
    
    return pd.DataFrame(list, columns=['client_id', 'birth_number', 'gender', 'district_id'])

def getColumns(df):
    sum = 0
    for key in df:
        if not 'train' in key and not 'test' in key:
            print('{} -> {}'.format(key, len(df_var[key].columns)))
            print('{}\n'.format(df_var[key].columns))

            sum = sum + len(df_var[key].columns)
    print(sum)


    #credit in cash', 'collection from another bank', nan,'withdrawal in cash', 'remittance to another bank',
#       'credit card withdrawal'

def n_credit_cash(col):
    return sum(col == 'credit in cash')

def n_collection_from_another_bank(col):
    return sum(col == 'collection from another bank')

def n_withdrawal_in_cash(col):
    return sum(col == 'withdrawal in cash')

def n_remittance_to_another_bank(col):
    return sum(col == 'remittance to another bank')

def n_credit_card_withdrawal(col):
    return sum(col == 'credit card withdrawal')

def n_interest_credited(col):
    return sum(col == 'none')

def n_credit(col):
    return sum(col == 'credit')

def n_withdrawal(col):
    return sum(col == 'withdrawal') + sum(col == 'withdrawal in cash')


# Merging Information

def Merge(df_var, haveCategorical = False):
    # Copying information from dictonary into dataframes

    client = df_var['client'].copy()
    disp = df_var['disp'].copy()
    card = df_var['card'].copy()
    loan = df_var['loan'].copy()
    account = df_var['account'].copy()
    district = df_var['district'].copy()
    card = card.rename({'type': 'card_type'}, axis=1)
    disp = disp.rename({'type': 'disp_type'}, axis=1)

    # Merging informations

    merge1 = pd.merge(client, disp[disp['disp_type'] =='OWNER'], on ='client_id')
    merge2 = pd.merge(merge1, account, on = 'account_id')
    merge3 = pd.merge(merge2, card[['card_id', 'disp_id', 'card_type']], on = 'disp_id', how = 'left')
    merge4 = pd.merge(merge3, district, left_on = 'district_id_x', right_on='code')
    merge5 = pd.merge(loan, merge4, on = 'account_id')

    # Dropping some columns
    merge5.drop(['client_id', 'disp_id', 'disp_type', 'card_id', 'code'], axis = 1, inplace = True)

    # Renaming columns
    merge5.rename({
        'date_x': 'loan_date', 
        'district_id_x': 'client_district', 
        'district_id_y': 'account_district',
        'date_y': 'account_date'
    }, axis = 1, inplace = True)

    pd.set_option('display.max_columns', None)
    
    # Processing transactions information 

    trans = df_var['trans'].copy()
    trans = trans.rename({
        'type': 'trans_type',
        'date': 'trans_date', 
        'amount': 'trans_amount',
        'balance': 'trans_balance'}, axis=1)
    trans.sort_values(by = ['account_id', 'trans_date'])
    trans.drop(['trans_id', 'k_symbol', 'account', 'bank'], axis=1, inplace = True)


    # Merging trans with previous information

    loan_info = pd.merge(merge5, trans, on = 'account_id')
    loan_info.fillna('none', inplace = True)
    
    #Columns that are not going to be changed
    cols = loan_info.columns.difference(['trans_date', 'trans_amount', 'trans_balance', 'trans_type', 'operation'], sort = False).tolist()

    # Group by operation

    loans = loan_info.groupby(by = cols, as_index = False).agg({
                        'trans_date': ['min', 'max'],
                        'trans_amount': ['min', 'max', 'mean', 'std', 'last'],
                        'trans_balance': ['min', 'max', 'mean', 'std', 'last'],
                        'trans_type':[n_credit, n_withdrawal],
                        'operation':[n_credit_cash, n_collection_from_another_bank, 
                                    n_withdrawal_in_cash, n_remittance_to_another_bank,
                                    n_credit_card_withdrawal, n_interest_credited]
                    })

    #Change columns name 
    loans.columns = ['%s%s' % (level1, '_%s' % level2 if level2 else '') for level1, level2 in loans.columns]
    categoricalDates = ['loan_date', 'trans_date_min', 'trans_date_max', 'birth_number', 'account_date',]
    if (haveCategorical):
        for catDate in categoricalDates:
            loans[catDate] = loans[catDate].apply(transformDate)
        loans.loc[loans['status']  == 1, 'status'] = 'Successful' 
        loans.loc[loans['status']  == -1, 'status'] = 'Unsuccessful' 
     
    return loans

def save_csv(df):
    df.sort_values(by = ['loan_id'])
    df.to_csv('loan_info')

def getCorr(df, size=(11,9)):
    corr = df.corr()
    ax = plt.subplots(figsize=size)
    ax = sns.heatmap(corr,  annot = True, linewidths=.1, mask = np.triu(corr), cmap='coolwarm')

def checkUniqueCategoricalVars():
    for f in CSV_files:
        print("{}:\n {}\n\n".format(f, CSV_files[f].select_dtypes(include=['object']).nunique()))