import pandas as pd
import sklearn as sk
import seaborn as sns
import numpy as np
from Preprocess import *
from DataAnalysis import *

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

SEED = 42

def metric_comparison(title, metric, ax):
    ax.set_title(title)
    sns.barplot(x = models_mean_scores.index, y = models_mean_scores[metric], ax=ax)

def find_best_params_kfold(model, X, y, param_grid, n_iter=10, n_splits=3, debug = True):  
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    search = GridSearchCV(estimator = model, param_grid = param_grid, scoring = metrics.make_scorer(auc_score, greater_is_better=True), n_jobs=-1, cv=kfold, verbose=2)
    
    result = search.fit(X, y.astype(int))
    if debug:
        print('Best Score: {}'.format(result.best_score_))
        print('Best Parameters: {}'.format(result.best_params_))    
    return (result.best_score_, result.best_params_)

def auc_score(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def apply_sampling(alg):
    return Pipeline([
                ('sampling', SMOTE(k_neighbors = 3, random_state = SEED)),
                ('classification', alg)
            ])
        

def ClassifierDecisionTree(X, y):  

    max_depth = [2, 4, 6, 8, 10]
    min_samples_split = [2, 4, 6]
    criterion = ['gini', 'entropy']
    splitter = ['best']
    max_features = ['auto', 'sqrt']
    class_weight = ['balanced', None]
    min_impurity_split = [0.05, 0.1, 0.23, 0.3]
    min_samples_leaf =  [1, 2, 4, 6]

    

    param_grid = dict(classification__max_depth=max_depth, classification__min_samples_split=min_samples_split, classification__criterion=criterion, classification__splitter=splitter, classification__max_features=max_features, classification__class_weight=class_weight, classification__min_impurity_split=min_impurity_split, classification__min_samples_leaf=min_samples_leaf)

    dt = apply_sampling(DecisionTreeClassifier(random_state=SEED))
    best_score, best_params = find_best_params_kfold(dt, X, y, param_grid, n_iter=50, n_splits=10)
    return best_score, best_params


def ClassifierGradientBoosting(X, y):
    loss = ['deviance', 'exponencial']
    learning_rate = [0.1, 0.2, 0.4]
    n_estimators = [100]
    subsample = [1, 0.5]
    criterion = ['friedman_mse', 'mse', 'mae']
    min_samples_split = [2, 4, 6]
    min_samples_leaf =  [1, 2, 4, 6]
    min_weight_fraction_leaf = [0, 1, 2]
    max_depth = [2, 3, 4]
    max_features = ['auto', 'sqrt']
    min_impurity_split = [0.05, 0.1, 0.23]

    param_grid = dict(classification__max_depth=max_depth, classification__min_samples_split=min_samples_split, classification__criterion=criterion, classification__loss=loss, classification__max_features=max_features, classification__learning_rate=learning_rate, classification__min_impurity_split=min_impurity_split, classification__min_samples_leaf=min_samples_leaf, classification__n_estimators=n_estimators, classification__subsample=subsample, classification__min_weight_fraction_leaf = min_weight_fraction_leaf)

    gb = apply_sampling(GradientBoostingClassifier(random_state=SEED))
    best_score, best_params = find_best_params_kfold(gb, X, y, param_grid, n_iter=50, n_splits=10)
    return best_score, best_params


def accuracy(tp,tn,fp,fn):
    return (tp + tn)/(tp+tn+fp+fn)

def precision(tp,fp):
    return tp/(tp+fp)

def recall(tp,fn):
    return tp/(tp+fn)

def f1score(tp,fp,fn):
    return 2*recall(tp,fn)/(recall(tp,fn)+precision(tp,fp))

def model_performance(model, train, test, normalize = False):
    
    X_train = train.drop(columns=['status', 'loan_id', 'account_id', "unemploymant rate '95 ", 'card_type', 'name', 'operation_n_credit_card_withdrawal'], axis=1)
    y_train = train['status'].copy()
    
    X_test = test.drop(columns=['status', 'loan_id', 'account_id', "unemploymant rate '95 ", 'card_type', 'name', 'operation_n_credit_card_withdrawal'], axis=1)
    y_test = test['status'].copy()
    if normalize:
        normalize_columns(X_train, X_train.columns)
        normalize_columns(X_test, X_test.columns)

    model = apply_sampling(model)
    model.fit(X_train, y_train.astype(int))
    y_pred = model.predict(X_test)
    fpr, tpr, _ = metrics.roc_curve(y_test.astype(int), y_pred)
    auc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test.astype(int), y_pred).ravel()
    print('AUC Score: {}\n\n'.format(auc))
    print('Accuracy: {}'.format(accuracy(tp,tn,fp,fn)))
    print('Precision: {}'.format(precision(tp,fp)))
    print('Recall: {}'.format(recall(tp,fn)))
    print('F1score: {}'.format(f1score(tp,fp,fn)))
    
    plot_roc_auc(fpr, tpr, auc)