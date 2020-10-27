import pandas as pd
import sklearn as sk
import seaborn as sns
import numpuy as np

global models_mean_scores
models_mean_scores = pd.DataFrame()

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate

def cross_validationScores(modelName, model, X, y, n_folds=10):
    metrics = {'MAE': 'neg_mean_absolute_error', 
               'RMSE': 'neg_root_mean_squared_error'}
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    scores = cross_validate(model, X, y, cv=kfold, scoring=metrics, n_jobs=-1)

    # multiply by -1 because sklearn scoring metrics are negative
    mean_mae_score = np.multiply(scores['test_MAE'], -1).mean()
    std_mae_score = np.multiply(scores['test_MAE'], -1).std()
    mean_rmse_score = np.multiply(scores['test_RMSE'], -1).mean()
    std_rmse_score = np.multiply(scores['test_RMSE'], -1).std()

    model_score = pd.DataFrame([[mean_mae_score, mean_rmse_score]], 
                             columns=['MAE', 'RMSE'], index=[modelName])
    global models_mean_scores
    models_mean_scores = models_mean_scores.append(model_score)

    print(str(n_folds) + " fold Cross validation scores for " + modelName)
    score_df = pd.DataFrame([[mean_mae_score, std_mae_score], 
                                  [mean_rmse_score, std_rmse_score]], 
                                 columns=['Mean score', 'std'], 
                                 index=['MAE', 'RMSE'])
    return score_df

def metric_comparison(title, metric, ax):
    ax.set_title(title)
    sns.barplot(x = models_mean_scores.index, y = models_mean_scores[metric], ax=ax)

def find_best_params_kfold(model, X, y, param_grid, n_iter=10, n_splits=3):  
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    
    search = RandomizedSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, n_iter=n_iter, cv=kfold, verbose=0, random_state=0)
    
    result = search.fit(X, y)
    
    return result.cv_results_

def report(results, n_top=3):
    res = []
    res_i = 0
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            res.append([np.multiply(results['mean_test_score'][candidate], -1), 
                        results['std_test_score'][candidate]])
            for key in results['params'][candidate].keys():
                res[res_i].append(results['params'][candidate][key])
            res_i+=1
    
    columns = ['Mean absolute error', 'std'] + list(results['params'][candidate].keys())
    return pd.DataFrame(res, columns=columns)

from sklearn.tree import DecisionTreeClassifier
def ClassifierDecisionTree(X, y, X_test):  

    max_depth = range(2, 30, 1)
    min_samples_split = range(1, 10, 1)

    param_grid = dict(max_depth=max_depth, min_samples_split=min_samples_split)

    find_best_params_kfold(DecisionTreeClassifier(random_state=0), X, y, param_grid, n_iter=50, n_splits=10)

    classifier_cart = DecisionTreeClassifier(max_depth=2, min_samples_split=9, random_state=0)
    classifier_cart = classifier_cart.fit(X, y)

    y_test_cart = classifier_cart.predict(X_test)

    #df_pred = pd.DataFrame(data={'Id': X_test["loan_id"], 'Predicted': y_test_cart})
    #df_pred.to_csv('loan_pred_1.csv', index=False)

    y_test_cart = classifier_cart.predict(X_test)
    #df_pred = pd.DataFrame(data={'Id': X_test_owner["loan_id"], 'Predicted': y_test_cart})
    #df_pred.to_csv('loan_pred_3.csv', index=False)
    y_test_cart