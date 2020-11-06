import pandas as pd
import sklearn as sk
import seaborn as sns
import numpy as np

global models_mean_scores
models_mean_scores = pd.DataFrame()

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate, StratifiedKFold, GridSearchCV
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
    learning_rate = [0.1, 0.2, 0.4, 0.6]
    n_estimators = [100, 150, 200]
    subsample = [1, 0.5, 1.5, 2]
    criterion = ['friedman_mse', 'mse', 'mae']
    min_samples_split = [2, 4, 6]
    min_samples_leaf =  [1, 2, 4, 6]
    min_weight_fraction_leaf = [0, 1, 2, 3]
    max_depth = [2, 3, 4, 6]
    max_features = ['auto', 'sqrt']
    min_impurity_split = [0.05, 0.1, 0.23, 0.3]

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

