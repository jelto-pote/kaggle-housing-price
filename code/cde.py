from ast import literal_eval
from os.path import isfile
from time import gmtime, localtime, time, strftime
from shutil import copy2

from pandas import DataFrame, NA, concat, read_csv


from sklearn.model_selection import cross_val_score

import optuna
from joblib import dump, load

from load_data import load_data
from eda import eda, plot_feature_importances, plot_permutation_importances
from feature_engineering import compare_models_with_without_engineered_features, feature_engineering, onehotencode
from functions import convert_sparse_to_df, update_maps_from_config, log, make_predictions

def get_params(name, data_map, model_map, runtime_map):
    param_df = read_csv('../performance/best.csv', index_col='name')
    
    # Retune if 1. A retune is requested, 2. It has never been tuned before
    if model_map[name]['retune'] == 1 or (name not in param_df.index):
        print(f'Training: Hyperparameter tuning {name}..')
        start_time = time()
        study = optuna.create_study(direction=runtime_map['perf_metric_direction'])
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
    if model_map[name]['refit'] == 1 or not isfile(f'../models/{name}_best.joblib'):
        print(f'Training: fitting for {name}..')
        if model_map[name]['handles_cat']:
            if 'CatBoost' in name:
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train'], data_map['y_train'], data_map['cat_cols_engineered'])
            else:
                model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(data_map['X_train'], data_map['y_train'])

        else:
            model_map[name]['model'] = model_map[name]['model'](**model_map[name]['params']).fit(convert_sparse_to_df(data_map, 'X_train_encoded'), data_map['y_train'])
    else:
        # In the other case, load the previously found best model as current model (no refit)
        model_map[name]['model'] = load(f'../models/{name}_best.joblib')
    
    return model_map

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
        dump(model_map[name]['model'], f'../models/{name}_current.joblib')
    curr_perf.to_csv('../performance/current.csv', index=False)

def write_better(model_map, perf_metric_direction):
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

def tune_train(data_map, model_map, runtime_map):
    for name in model_map:
        model_map = get_params(name, data_map, model_map, runtime_map)
        model_map = fit_models(name, data_map, model_map)

    for name in model_map:
        if model_map[name]['refit'] == 1 and model_map[name]['handles_cat']:
            print(f'Training: checking kfold performance for {name}..')
            model_map[name]['cross_val_score'] = cross_val_score(model_map[name]['model'], data_map['X_train'], data_map['y_train'], cv=runtime_map['kfold'], scoring=runtime_map['scoring'])
            model_map[name]['kfold_perf'] = "%.2f%% (%.2f%%)" % (model_map[name]['cross_val_score'].mean()*100, model_map[name]['cross_val_score'].std()*100)
            model_map[name]['perf'] = model_map[name]['cross_val_score'].mean()*100

            
            
    write_current(model_map, runtime_map)
    write_better(model_map, runtime_map['perf_metric_direction'])

    return data_map, model_map
  
def main():
    cudf_available=0
 
    data_map, model_map, runtime_map = update_maps_from_config('config/data_map.json', 'config/model_map.json', 'config/runtime_map.json')

    # Load raw data and perform EDA on it
    data_map, runtime_map = load_data(data_map, runtime_map, cudf_available)                                            # Load data, split the datasets and fillna's without data leakage
    eda(data_map, runtime_map)                                                                                          # EDA on raw data
    data_map = feature_engineering(data_map)                                                                            # Feature engineering. Write to data_map
    eda(data_map, runtime_map)                                                                                          # EDA on engineered data

    data_map = compare_models_with_without_engineered_features(data_map, model_map, runtime_map)
    data_map = onehotencode(data_map)                                                                                   # Add entries in data map for encoded data
    data_map, model_map = tune_train(data_map, model_map, runtime_map)                                                  # Train and evaluate models

    if runtime_map['plots'][-1]:
        plot_feature_importances(data_map, model_map)
        plot_permutation_importances(data_map, model_map, runtime_map)

    del data_map['X_train'], data_map['X_train_encoded']                                                                # Free up memory. At this point, the training data is no longer needed


if __name__ == "__main__":
    main()