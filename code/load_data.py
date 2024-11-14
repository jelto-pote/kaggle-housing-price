from pandas import concat, DataFrame, read_csv

import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from functions import log

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
                log(f'{col}: Filled missing values with mean {mean_value}')
            else:  
                # Skewed distribution
                median_value = df[col].median()
                df.fillna({col: median_value}, inplace=True)
                log(f'{col}: Filled missing values for {col} with mode {median_value}')
    return df

def split(X, y):
    print('Splitting train and test..')
    num_samples = len(X)
    log(f"Number of samples in the dataset: {num_samples}")

    if num_samples < 100:
        train_size = 0.5  # 50/50 split for small datasets
        log("Using 50/50 split for small dataset.")
    elif num_samples < 1000:
        train_size = 0.7  # 70/30 split for medium datasets
        log("Using 70/30 split for medium dataset.")
    else:
        train_size = 0.8  # 80/20 split for large datasets
        log("Using 80/20 split for large dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test
    
def custom_filter(df, filter_col, filter_type, filter_amt):
    if filter_type not in ['gt', 'st']:
        raise ValueError("custom_filters function expected gt or st as condition_type, got {}".format(filter_type))
    
    if filter_type == 'gt':
        output = df[(df[filter_col] > filter_amt) | (df[filter_col].isna())]
    
    if filter_type == 'st':
        output = df[(df[filter_col] < filter_amt) | (df[filter_col].isna())]

    log(f'train data shape change after custom_filter: {df.shape}, {output.shape}')
    return output

# This function helps to load data correctly based on cudf availability. It's changed now, because we need portability
def load_df(data_map):
    df_train = DataFrame()
    for path, index in zip(data_map['train_files'], data_map['index_cols']):       
        df_tmp = read_csv(path, index_col=index) if index else read_csv(path)
        df_train = concat([df_train, df_tmp])

    del df_tmp

    return df_train

def load_data(data_map, runtime_map, cudf_available):
    print('\nLoading data..')
    target_col = data_map['target_cols'][0]                                      # For readability
    
    # Load csv files into dataframes
    train_data = load_df(data_map)
    
    # Note different columns, determine kfold type
    data_map['cat_cols_raw'] =  [col for col in train_data.select_dtypes(include=['object', 'category']).columns.tolist() if col != target_col]
    data_map['num_cols_raw'] = [col for col in train_data.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
    runtime_map['kfold'] = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) if runtime_map['task_type'] == 'classification' else KFold(n_splits=5, shuffle=True, random_state=7)

    # Cast all object columns to category data type
    for col in data_map['cat_cols_raw']:
        train_data[col] = train_data[col].astype('category')

    # Apply filters to training data
    filter_conditions = {
        #'person_age': (110, 'st'),
    }
    for col, (value, method) in filter_conditions.items():
        train_data = custom_filter(train_data, col, method, value)

    train_data.drop_duplicates(inplace=True)                                      # Drop fully duplicated records ..
    train_data.dropna(subset=[target_col], inplace=True)                          # Drop training records which don't have a target variable ..
    train_data.drop(data_map['drop_cols'], axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..

    # Create X, y for training data
    y_train_data = train_data.pop(target_col)
    X_train_data = train_data

    # Train/test split
    data_map['X_train'], data_map['X_test'], data_map['y_train'], data_map['y_test'] = split(X_train_data, y_train_data)   # Train test split
    data_map['X_train_no_engineered'] = data_map['X_train']                                                                # Save this for comparing model with/without feature engineering 
    
    # Note index values. This is needed for converting sparse back to dense later on
    data_map['X_train_index_values'] = data_map['X_train'].index.tolist()
    data_map['X_test_index_values'] = data_map['X_test'].index.tolist()

    data_map['X_train'], data_map['X_test'] = fill_missing_values(data_map['X_train']), fill_missing_values(data_map['X_test'])         # This will be used for training (no data leakage)
    
    del train_data, X_train_data, y_train_data

    log(f"X_train shape: {data_map['X_train']}")
    log(f"X_test shape: {data_map['X_test']}")


    return data_map, runtime_map
