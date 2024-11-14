from pandas import concat, cut

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from functions import log

def remove_outliers_isolation_forest(X_train, y_train, data_map, contamination=0.000001):
    # Create a copy of the original DataFrames to avoid modifying it
    cleaned_X_train = X_train.copy()
    
    # Initialize a list to store removed records
    train_removed_records = []

    # Loop through all numeric columns
    for col in data_map['num_cols_raw']:
        # Fit Isolation Forest on the X_train column and apply fitted model to X_test too
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        train_outliers = iso_forest.fit_predict(cleaned_X_train[[col]])  # Fit and predict on the specified column

        # Identify outlier rows
        train_outlier_mask = train_outliers == -1

        # Store removed records in the list
        train_removed_records.append(cleaned_X_train[train_outlier_mask])

        # Filter out the outliers
        cleaned_X_train = cleaned_X_train[~train_outlier_mask]  # Keep only non-outliers

    # Concatenate all removed records into a single DataFrame
    train_removed_df = concat(train_removed_records, ignore_index=True)

    # Print the removed records if any
    if not train_removed_df.empty:
        log("Removed records:")
        log(train_removed_df)

    cleaned_y_train = y_train[cleaned_X_train.index]

    return cleaned_X_train, cleaned_y_train

def cat_to_ordered_numeric(dfs, mapping, existing_col, replace_existing=False):  
    for i, df in enumerate(dfs):
        if replace_existing:
            # Apply the mapping directly to the existing column
            df[existing_col] = df[existing_col].map(mapping)
        else:
            # Apply the mapping to new column
            df[existing_col + '_mapped'] = df[existing_col].map(mapping)
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def add_interaction_feature_number(dfs, col1, col2, operation, drop1=False, drop2=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but add_interaction_feature function got {}".format(len(dfs)))
    
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

    return dfs[0], dfs[1], dfs[2]

def add_interaction_feature_raw(dfs, num_col, cat_col, operation, drop1=False, drop2=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but add_interaction_feature function got {}".format(len(dfs)))
    
    for i, df in enumerate(dfs):
        df[cat_col + '_' + num_col + '_grouped'] = df.groupby(cat_col, observed=True)[num_col].transform('median')

        # Drop original columns if requested
        if drop1:
            df.drop(cat_col, axis=1, inplace=True)
        if drop2:
            df.drop(num_col, axis=1, inplace=True)

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def create_binned_feature(dfs, col, bins, drop=False):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but create_binned_feature got {}".format(len(dfs)))
    
    bin_name = col + '_binned'
    
    for i in range(len(dfs)):
        # Create binned feature
        dfs[i][bin_name] = cut(dfs[i][col], bins=bins, labels=False)
        
        # Drop the original column if requested
        if drop:
            dfs[i] = dfs[i].drop(col, axis=1)

    return dfs[0], dfs[1], dfs[2]

def boxcox_transform_skewed_features(dfs, xform_cols, threshold=1):
    shift_value = 0
    # First, determine the minimum shift value across all dataframes
    for df in dfs:        
        # Find the minimum value for all numeric features
        for col in xform_cols:
            shift_value = min(df[col].min(), shift_value)
    
    # If the shift value is negative, calculate the shift amount
    shift_amount = 0
    if shift_value < 0:
        shift_amount = abs(shift_value) + 1  # Shift to avoid negative values
    
    # Now apply the shift and Box-Cox transformation to each dataframe
    for i, df in enumerate(dfs):        
        # Iterate over each to be transformed feature
        for col in xform_cols:
            # Shift the values to ensure they are all positive for Box-Cox
            if shift_amount > 0:
                df[col] = df[col] + shift_amount
                print(f"Shifted '{col}' by {shift_amount} to avoid negative values for Box-Cox transformation.")
            
            # Calculate skewness for the current feature
            skewness = df[col].skew()

            # Apply Box-Cox transformation if skewness exceeds the threshold
            if abs(skewness) > threshold:
                print(f"Applying Box-Cox transformation to '{col}' as skewness {skewness} exceeds threshold of {threshold}.")
                df[col], _ = stats.boxcox(df[col])
        
        # Update the dataframe in the list
        dfs[i] = df

    return dfs

def scale_selected_features(dfs, cols_to_scale):
    if len(dfs) != 3:
        raise ValueError(f"Expected exactly 3 DataFrames, but got {len(dfs)}")
    
    # Instantiate the scaler
    scaler = StandardScaler()

    for i, df in enumerate(dfs):
        # Check if all the specified columns exist in the DataFrame
        missing_cols = [col for col in cols_to_scale if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing in DataFrame {i+1}: {missing_cols}")
        
        # Scale the selected columns
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Update the DataFrame in the list
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def create_group_scaled(dfs, num_col, cat_col):
    # This function groups by the cat_col and scales the num_col within each group
    for i, df in enumerate(dfs):
        df[num_col + '_scaled'] = df.groupby(cat_col)[num_col].transform(lambda x: (x - x.mean()) / x.std())
        dfs[i] = df

    return dfs[0], dfs[1], dfs[2]

def drop_uninteresting(dfs, cols):
    if len(dfs) != 3:
        raise ValueError("Expected exactly 3 DataFrames, but drop_uninteresting got {}".format(len(dfs)))
       
    for i, df in enumerate(dfs):
        for col in cols:
            dfs[i] = df.drop(cols, axis=1)

    return dfs[0], dfs[1], dfs[2]

def compare_models_with_without_engineered_features(data_map, model_map, runtime_map):
    print("Comparing model performance with and without engineered features..")
    # Set vars
    scoring, kfold = runtime_map['scoring'], runtime_map['kfold']

    for name in model_map:
        if name == 'XGBClassifier':
            model = XGBClassifier(enable_categorical = True)
            score_with_engineered = np.mean(cross_val_score(model, data_map['X_train'], data_map['y_train'], scoring=scoring, cv=kfold))
            score_without_engineered = np.mean(cross_val_score(model, data_map['X_train_no_engineered'], data_map['y_train'], scoring=scoring, cv=kfold))

            # Store the results in a dictionary
            performance_comparison = {
                'with_engineered': score_with_engineered,
                'without_engineered': score_without_engineered
            }

            # Plot the comparison
            categories = ['With Engineered Features', 'Without Engineered Features']
            scores = [performance_comparison['with_engineered'], performance_comparison['without_engineered']]

            plt.figure(figsize=(8, 6))
            plt.bar(categories, scores, color=['skyblue', 'salmon'])
            plt.title('Model Performance Comparison')
            plt.ylabel(scoring.replace('_', ' ').capitalize())  # Dynamic ylabel based on the scoring metric
            plt.ylim(0, 1)  # Assuming scoring metric is ROC AUC or similar (range 0-1)
            
            # Display the score values on the bars
            for i, score in enumerate(scores):
                plt.text(i, score + 0.02, f'{score:.5f}', ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'../eda/model/{name}_performance_comparison_{scoring}.png')
            plt.close()

            # Log the performance comparison
            log(f"Model performance with engineered features: {score_with_engineered}")
            log(f"Model performance without engineered features: {score_without_engineered}")
    
    del data_map['X_train_no_engineered']   # Cleanup for memory
    return data_map

def onehotencode(data_map):
    encoder = OneHotEncoder(drop=None, sparse_output=True, handle_unknown='ignore', dtype=np.float32)         # Sparse for memory management .. Use dense_array = sparse_matrix.todense() and pd.DataFrame(dense_array, columns=encoded_col_names) on demand when needed
    
    # Fit and transform the training data on categorical columns only
    data_map['X_train_encoded'] = encoder.fit_transform(data_map['X_train'][data_map['cat_cols_raw']])
    data_map['X_test_encoded'] = encoder.transform(data_map['X_test'][data_map['cat_cols_raw']])

    data_map['encoder'] = encoder
    data_map['encoded_columns'] = encoder.get_feature_names_out(data_map['cat_cols_raw']).tolist()

    return data_map

def feature_engineering(data_map):
    # Remove outliers from training set (not from test or pred)
    #X_train, y_train = remove_outliers_isolation_forest(X_train, y_train, data_map)

    # Update maps to include engineered columns and data
    data_map['cat_cols_engineered'] = data_map['X_train'].select_dtypes(include=['category']).columns.tolist()
    data_map['num_cols_engineered'] = data_map['X_train'].select_dtypes(include=[np.number]).columns.tolist()

    return data_map