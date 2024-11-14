# General imports
from ast import literal_eval
from pandas import concat, DataFrame, read_csv, set_option
import numpy as np
from os.path import isfile
from joblib import dump, load
import json
from shutil import copy2
from time import gmtime, localtime, time, strftime

# Model-related imports (training, tuning, ..)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import optuna



# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

def update_maps_from_config(data_map_file, model_map_file, runtime_map_file):
    with open(data_map_file, 'r') as f:
        data_map_config = json.load(f)
    
    with open(model_map_file, 'r') as f:
        model_map_config = json.load(f)
    
    with open(runtime_map_file, 'r') as f:
        runtime_map_config = json.load(f)    

    runtime_map_config['runtime'] = strftime("%Y-%m-%d %H:%M:%S", localtime())

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

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() == 0:
            pass
        elif df[col].dtype in ['object', 'bool', 'category']:  # Categorical
            df[col].fillna('na', inplace=True)
        else:   
            # Numerical
            skewness = round(df[col].skew(), 1)  # Decide whether to use mean or mode
            if abs(skewness) < 0.5:  # Approximately normal distribution
                mean_value = df[col].mean()
                df.fillna({col: mean_value}, inplace=True)
                print(f'{col}: Filled missing values with mean {mean_value}')
            else:  
                # Skewed distribution
                median_value = df[col].median()
                df.fillna({col: median_value}, inplace=True)
                print(f'{col}: Filled missing values for {col} with mode {median_value}')
    return df

def split(X, y):
    print('Splitting train and test..')
    num_samples = len(X)
    print(f"Number of samples in the dataset: {num_samples}")

    if num_samples < 100:
        train_size = 0.5  # 50/50 split for small datasets
        print("Using 50/50 split for small dataset.")
    elif num_samples < 1000:
        train_size = 0.7  # 70/30 split for medium datasets
        print("Using 70/30 split for medium dataset.")
    else:
        train_size = 0.8  # 80/20 split for large datasets
        print("Using 80/20 split for large dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test

def onehotencode(data_map):
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform the training data on categorical columns only
    data_map['X_train_encoded'] = encoder.fit_transform(data_map['X_train'][data_map['cat_cols_raw']])
    data_map['X_test_encoded'] = encoder.transform(data_map['X_test'][data_map['cat_cols_raw']])

    data_map['encoder'] = encoder
    data_map['encoded_columns'] = encoder.get_feature_names_out(data_map['cat_cols_raw']).tolist()

    return data_map

def tune_train(data_map, model_map, runtime_map):
    for name in model_map:
        model_map = get_params(name, data_map, model_map, runtime_map)
        model_map = fit_models(name, data_map, model_map)

    for name in model_map:
        if model_map[name]['refit'] == 1 and model_map[name]['handles_cat'] and not "Cat" in name:
            print(f'Training: checking kfold performance for {name}..')

            # Calculate k-fold performance with RMSE as the metric
            cross_val_scores = cross_val_score(
                model_map[name]['model'], 
                data_map['X_train'], 
                data_map['y_train'], 
                cv=runtime_map['kfold'], 
                scoring='neg_root_mean_squared_error'
            )
            # Convert the scores to positive RMSE values
            rmse_scores = -cross_val_scores
            model_map[name]['kfold_perf'] = "%.2f (%.2f)" % (rmse_scores.mean(), rmse_scores.std())
            model_map[name]['perf'] = rmse_scores.mean()  # Use mean RMSE as performance metric

    write_current(model_map, runtime_map)
    return data_map, model_map

def get_params(name, data_map, model_map, runtime_map):
    try:
        param_df = read_csv('performance/best.csv', index_col='name')
    except:
        param_df = DataFrame(columns=['name','perf','kfold_perf','params','timestamp'])

    # Retune if 1. A retune is requested, 2. It has never been tuned before
    if model_map[name]['retune'] == 1 or (name not in param_df.index):
        print(f'Training: Hyperparameter tuning {name}..')
        start_time = time()
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: model_map[name]['obj_func'](trial, data_map, runtime_map), n_trials=runtime_map['n_trials'])
        model_params = study.best_params
        print(f"{name} done. Took {strftime('%H:%M:%S', gmtime(time() - start_time))} for {runtime_map['n_trials']}.")
    else:
        model_params = literal_eval(param_df.loc[name, 'params'])
    
    model_map[name]['params'] = model_params
    
    return model_map

def fit_models(name, data_map, model_map):
    print('Training: retrieving fit models or refitting models..')

    # If there is no best version of the model, or if a refit is requested, update model_map[name]['model'] with newly fitted model.
    if model_map[name]['refit'] == 1 or not isfile(f'models/{name}_best.joblib'):
        print(f'Training: fitting for {name}..')
        if model_map[name]['handles_cat']:
            if 'CatBoost' in name:
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train'], data_map['y_train'], data_map['cat_cols_engineered'])
            else:
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train'], data_map['y_train'])

        else:
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train_encoded'], data_map['y_train'])
    else:
        # In the other case, load the previously found best model as current model (no refit)
        model_map[name]['model'] = load(f'models/{name}_best.joblib')
    
    return model_map

def add_interaction_feature_number(dfs, col1, col2, operation, drop1=False, drop2=False):
    if len(dfs) != 2:
        raise ValueError("Expected exactly 2 DataFrames, but add_interaction_feature function got {}".format(len(dfs)))
    
    for i, df in enumerate(dfs):
        if operation == '+':
            feature = col1 + '_' + col2 + '_sum'
            df[feature] = df[col1] + df[col2]
        elif operation == '-':
            feature = col1 + '_' + col2 + '_diff'
            df[feature] = df[col1] - df[col2]
        elif operation == '*':
            feature = col1 + '_' + col2 + '_product'
            df[feature] = df[col1] * df[col2]
        elif operation == '/':
            feature = col1 + '_' + col2 + '_division'
            # Prevent division by zero
            df[feature] = df[col1] / df[col2].replace(0, np.nan)
        # Drop original columns if requested
        if drop1:
            df.drop(col1, axis=1, inplace=True)
        if drop2:
            df.drop(col2, axis=1, inplace=True)

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs[0], dfs[1]

def write_current(model_map, runtime_map):
    curr_perf = DataFrame(columns=['name','perf','kfold_perf','params', 'timestamp'])

    for name in model_map:
        #for model, name, perf, kfold_perf, param in zip(models, names, perfs, kfold_perfs, params):
        new_row = DataFrame({
            'name': name,
            'perf': model_map[name]['perf'],
            'kfold_perf': model_map[name]['kfold_perf'],
            'params': [model_map[name]['params']],
            'timestamp': runtime_map['runtime']})
        if curr_perf.empty:
            curr_perf = new_row
        else:
            curr_perf = concat([curr_perf, new_row], ignore_index=True)
        dump(model_map[name]['model'], f'models/{name}_current.joblib')
    print(curr_perf)
    curr_perf.to_csv('performance/current.csv', index=False)

def write_better(model_map):
    # If there is a best model/perf file, compare it with current run
    curr_perf = read_csv('../performance/current.csv', index_col='name')
    
    if isfile('../performance/best.csv'):
        best_perf = read_csv('../performance/best.csv', index_col='name')
        for name in model_map:
            if isfile(f'../models/{name}_best.joblib'):
                # Fetch current and best performance from the dataframe
                best_perf_value, curr_perf_value = best_perf.loc[name, 'perf'], curr_perf.loc[name, 'perf']
                # Compare based on the direction
                if curr_perf_value < best_perf_value:
                    best_perf.loc[name] = curr_perf.loc[name]
                    copy2(f'../models/{name}_current.joblib', f'../models/{name}_best.joblib')
                    print(f'Yeeehaaa! We\'ve got ourselves a new best candidate for model {name}!')
            else:
                copy2(f'../models/{name}_current.joblib', f'../models/{name}_best.joblib')

        best_perf.to_csv('../performance/best.csv')
    else:
        copy2('../performance/current.csv', '../performance/best.csv')

def plot_permutation_importances(data_map, model_map, runtime_map):
    # Investigate permutation importances of trained models
    print('EDA: Investigating permutation importances..')
    
    for name, model_info in model_map.items():
        if model_info['refit'] == 1 and model_info['handles_cat']:
            # Use the appropriate training data and feature names based on whether the model handles categorical data
            X_train = data_map['X_train']
            feature_names = data_map['X_train'].columns
 
            # Perform permutation importance using the appropriate training data
            perm_importance = permutation_importance(model_info['model'], X_train, data_map['y_train'], scoring=runtime_map['scoring'])

            print(f'Calculating permutation importance for {name}...')            
            # Sort and select the top features (default: top 30 or fewer)
            num_features_to_plot = len(feature_names) if len(feature_names) < 30 else 30
            perm_sorted_idx = perm_importance.importances_mean.argsort()[-num_features_to_plot:]
            perm_top_features = [feature_names[i] for i in perm_sorted_idx]
            perm_top_importances = perm_importance.importances_mean[perm_sorted_idx]

            # Plot permutation importances
            plt.figure(figsize=(30, 10))
            plt.barh(perm_top_features, perm_top_importances)
            plt.title(f'Permutation Feature Importance - {name}')
            plt.xlabel('Importance')
            plt.savefig(f'eda/model/{name}_permutation_importance.png')
            plt.close()

def plot_feature_importances(data_map, model_map):
    # Investigate feature importances of trained models
    print('\nEDA: Investigating feature importances...')
    
    for name, model_info in model_map.items():
        if model_info['refit'] == 1 and model_info['handles_cat']:
            # Choose the appropriate feature names based on categorical handling
            feature_names = data_map['X_train'].columns
            
            # Extract model feature importances
            importances = model_info['model'].feature_importances_
            
            # Determine the number of features to plot (max 30)
            num_features_to_plot = min(len(feature_names), 30)
            top_indices = np.argsort(importances)[-num_features_to_plot:]
            
            # Get the top features and their importances
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            
            # Plotting feature importance
            plt.figure(figsize=(30, 10))
            plt.barh(top_features, top_importances)
            plt.title(f'Feature Importance - {name}')
            plt.xlabel('Importance')
            plt.savefig(f'eda/model/{name}_feature_importance.png')
            plt.close()

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
    model.fit(data_map['X_train_encoded'], data_map['y_train'])
    y_pred = model.predict(data_map['X_test_encoded'])
    rmse = np.sqrt(mean_squared_error(data_map['y_test'], y_pred))
    
    return rmse

# Objective function for XBGBoost
def obj_xgb(trial, data_map, runtime_map):
    param = {'objective': "reg:squarederror",
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
    model.fit(data_map['X_train'], data_map['y_train'])
    predictions = model.predict(data_map['X_test'])
    return sqrt(mean_squared_error(data_map['y_test'], predictions))

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
    
    model.fit(data_map['X_train'], data_map['y_train'])
    predictions = model.predict(data_map['X_test'])
    return sqrt(mean_squared_error(data_map['y_test'], predictions))

def main():
    data_map, model_map, runtime_map = update_maps_from_config('config/data_map.json', 'config/model_map.json', 'config/runtime_map.json')  # Load project master data

    target_col = data_map['target_col']                                
    train_data = read_csv(data_map['train_file'])
    train_data.drop_duplicates(inplace=True)                                      # Drop fully duplicated records ..
    train_data.dropna(subset=[target_col], inplace=True)                          # Drop training records which don't have a target variable ..
    train_data.drop(data_map['drop_cols'], axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..
 
    # Note different columns, determine kfold type
    data_map['cat_cols_raw'] =  [col for col in train_data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_col]
    data_map['num_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
    runtime_map['kfold'] = KFold(n_splits=5, shuffle=True, random_state=7)

    # Create X, y for training data
    y_train_data = train_data.pop(target_col)
    X_train_data = train_data

    # Cast all object columns to category data type
    for col in data_map['cat_cols_raw']:
        train_data[col] = train_data[col].astype('category')


    # Train/test split
    data_map['X_train'], data_map['X_test'], data_map['y_train'], data_map['y_test'] = split(X_train_data, y_train_data)                # Train test split
    
    data_map['X_train'], data_map['X_test'] = fill_missing_values(data_map['X_train']), fill_missing_values(data_map['X_test'])         # This will be used for training (no data leakage)
    
    print(f"X_train shape: \n {data_map['X_train']}")
    print(f"X_test shape: \n {data_map['X_test']}")



    # Note different columns according to type, determine approx best kfold type
    data_map['cat_cols_raw'] =  [col for col in train_data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_col]
    data_map['num_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
    runtime_map['kfold'] = KFold(n_splits=5, shuffle=True, random_state=1)

    # very basic/quick feature engineering
    data_map['X_train'], data_map['X_test'] = add_interaction_feature_number([data_map['X_train'], data_map['X_test']], 'median_income', 'households', '/')
    data_map['X_train'], data_map['X_test'] = add_interaction_feature_number([data_map['X_train'], data_map['X_test']], 'population', 'households', '/')


    # Update maps to include engineered columns and data
    data_map['cat_cols_engineered'] = data_map['X_train'].select_dtypes(include=['category']).columns.tolist()
    data_map['num_cols_engineered'] = data_map['X_train'].select_dtypes(include=[np.number]).columns.tolist()



    data_map = onehotencode(data_map)                                                                                   # Add entries in data map for encoded data

    data_map, model_map = tune_train(data_map, model_map, runtime_map)                                                  # Train and evaluate models
    
    if runtime_map['plots'][-1]:
        plot_feature_importances(data_map, model_map)
        plot_permutation_importances(data_map, model_map, runtime_map)

if __name__ == "__main__":
    main()
