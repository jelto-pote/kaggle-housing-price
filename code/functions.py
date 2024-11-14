from pandas import DataFrame, set_option

import json

from time import localtime, strftime

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import numpy as np

#-------------------------------------------------------------------------------------
# Objective function for RandomForest
def obj_rf(trial, data_map, runtime_map):
    param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 13),  
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),  
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state': 42}
    
    model = RandomForestRegressor(**param)
    model.fit(convert_sparse_to_df(data_map, 'X_train_encoded'), data_map['y_train'])
    y_pred = model.predict(data_map['X_test_encoded'])
    
    rmse = np.sqrt(mean_squared_error(data_map['y_test'], y_pred))
    
    return rmse

# Objective function for XBGBoost
def obj_xgb(trial, data_map, runtime_map):
    param = {'objective': "reg:squaredlogerror",
        'max_depth': trial.suggest_int('max_depth', 7, 16),
        'max_bin': trial.suggest_int('max_bin', 256, 2000), 
        'tree_method': 'auto', 
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5),  
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),  
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 0, 5),
        'alpha': trial.suggest_float('alpha', 0, 5),
        'enable_categorical': trial.suggest_categorical('enable_categorical', [True])} 
    
    model = XGBRegressor(**param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()

# Objective function for CatBoost
def obj_cat(trial, data_map, runtime_map):
    param = {
        'early_stopping_rounds': 50,
        'iterations': trial.suggest_int('iterations', 400, 600),  
        'depth': trial.suggest_int('depth', 3, 7),  
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.2),  
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 9.0),
        'random_seed': 42,
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 1),
        'verbose': trial.suggest_categorical('verbose', [False]),
        'loss_function': "RMSE"
    }
    model = CatBoostRegressor(cat_features=data_map['cat_cols_engineered'], **param)
    cv_scores = cross_val_score(model, data_map['X_train'], data_map['y_train'], cv=5, scoring=runtime_map['scoring'])
    return cv_scores.mean()


#-------------------------------------------------------------------------------------
def convert_sparse_to_df(data_map, sparse_matrix_key):
    
    if 'test' in sparse_matrix_key:
        index_vals =  data_map['X_test_index_values'] 
    elif 'pred' in sparse_matrix_key:
        index_vals = data_map['X_pred_index_values'] 
    else:
        index_vals = data_map['X_train_index_values']    
    
    
    return DataFrame.sparse.from_spmatrix(
        data_map[sparse_matrix_key], 
        columns=data_map['encoded_columns'], 
        index=index_vals  # This should be an array of values
    )

def get_current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

# Initialization
def initialize():
    set_option('display.max_columns', None)
    
    # Clean up previous logs
    with open('log.txt', 'w') as f1:
        pass

# Logging function
def log(msg, fname='log.txt'):
    if isinstance(msg, str):
        with open(fname, 'a') as file:
            file.write(msg + '\n')
            file.write('' + '\n')
    elif isinstance(msg, DataFrame):
        with open(fname, 'a') as file:
            msg.to_csv(fname, index=False)

# Function to make final predictions
def make_predictions(model, X_data, proba_func):
    if proba_func == 'decision_function':
        return model.decision_function(X_data)
    elif proba_func == 'pred_proba':
        return model.predict_proba(X_data)[:, 1]

def update_maps_from_config(data_map_file, model_map_file, runtime_map_file):
    with open(data_map_file, 'r') as f:
        data_map_config = json.load(f)
    
    with open(model_map_file, 'r') as f:
        model_map_config = json.load(f)
    
    with open(runtime_map_file, 'r') as f:
        runtime_map_config = json.load(f)    

    runtime_map_config['runtime'] = get_current_time()

    # Reconstruct the model_map objects
    model_map = {}
    for name, config in model_map_config.items():
        model_class = globals()[config['model']]  # Retrieve the model class from json using globals()
        obj_func = globals()[config['obj_func']]  # Retrieve the function from json using globals() 
        model_map[name] = {
            'model': model_class,
            'handles_cat': config['handles_cat'],
            'handles_sparse': config['handles_sparse'],
            'params': config['params'],
            'obj_func': obj_func,
            'retune': config['retune'],
            'refit': config['refit'],
            'pred_proba': config['pred_proba'],
            'proba_func': config['proba_func'],
            'perf': config['perf'],
            'kfold_perf': config['kfold_perf']
        }

    return data_map_config, model_map, runtime_map_config
