import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, concat, get_dummies

import numpy as np

from os import listdir, remove
from os.path import isfile, join

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

from functions import log

# Model property insights
def plot_permutation_importances(data_map, model_map, runtime_map):
    # Investigate permutation importances of trained models
    print('EDA: Investigating permutation importances..')
    
    for name, model_info in model_map.items():
        if model_info['refit'] == 1 and "XGB" in name:
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

            # Log the top permutation importances
            log(f"The top permutation features for {name} are: {perm_top_features}")
            log(f"The top permutation importances for {name} are: {perm_top_importances}")

            # Plot permutation importances
            plt.figure(figsize=(30, 10))
            plt.barh(perm_top_features, perm_top_importances)
            plt.title(f'Permutation Feature Importance - {name}')
            plt.xlabel('Importance')
            plt.savefig(f'../eda/model/{name}_permutation_importance.png')
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
            
            # Logging the top features
            log(f"Top features for {name}: {top_features}")
            log(f"Top feature importances for {name}: {top_importances}")
            
            # Plotting feature importance
            plt.figure(figsize=(30, 10))
            plt.barh(top_features, top_importances)
            plt.title(f'Feature Importance - {name}')
            plt.xlabel('Importance')
            plt.savefig(f'../eda/model/{name}_feature_importance.png')
            plt.close()

# Feature insights
def plot_features_vs_target(data, num_cols, cat_cols, target_col, eda):
    print("Plotting features vs target..")

    # Helper function to create grouped plots and save them
    def save_plots(cols, plot_func, plot_type, file_idx):
        num_files = int(np.ceil(len(cols) / 20))
        for i in range(num_files):
            start_idx = i * 20
            end_idx = min(start_idx + 20, len(cols))
            subset_cols = cols[start_idx:end_idx]
            
            # Create the plots in a figure
            fig, axes = plt.subplots(len(subset_cols), 1, figsize=(10, 6 * len(subset_cols)))
            axes = axes if len(subset_cols) > 1 else [axes]  # Ensure axes is always iterable
            for ax, col in zip(axes, subset_cols):
                plot_func(ax, col)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(f'../eda/{eda}/{plot_type}_features_vs_target_{file_idx + i + 1}.png')
            plt.close()
    # Function to plot boxplots for numerical columns
    def plot_boxplot(ax, col):
        sns.boxplot(x=target_col, y=col, data=data, showmeans=True, showcaps=True, ax=ax)
        ax.set_title(f'Boxplot of {col} by {target_col}')
        plt.xticks(rotation=45)  # Rotate tick labels using plt.xticks
    # Function to plot countplots for categorical columns
    def plot_countplot(ax, col):
        sns.countplot(x=col, hue=target_col, data=data, palette="Set2", ax=ax)
        ax.set_title(f'Countplot of {col} by {target_col}')
        plt.xticks(rotation=45)  # Rotate tick labels using plt.xticks
    
    # Plot numerical features with boxplots
    save_plots(num_cols, plot_boxplot, 'numerical', 0)

    # Plot categorical features with countplots
    save_plots(cat_cols, plot_countplot, 'categorical', 0)

def plot_numerical_features(train_data, target_col, num_cols, eda, max_plots_per_file=20):    
    print("Plotting numerical..")
    # Determine how many plots to create based on max_plots
    num_plots = int(np.ceil(len(num_cols) / max_plots_per_file))

    for i in range(num_plots):
        # Determine which columns to include in this plot
        start_idx = i * max_plots_per_file
        end_idx = min(start_idx + max_plots_per_file, len(num_cols))
        subset_cols = num_cols[start_idx:end_idx]

        # Create the pairplot for the subset of columns
        print(f'EDA: Plotting numerical features ({eda}), Plot {i + 1}/{num_plots}..')
        fig = sns.pairplot(train_data[subset_cols + [target_col]], hue=target_col, diag_kind="kde", corner=True, plot_kws={'alpha': 0.7})
        
        # Set x-labels for each subplot
        for ax in fig.axes[-1]:  # Iterate over the last row's axes
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)  # Set x-labels with specified font size
        
        # Save the plot
        plt.savefig(f'../eda/{eda}/numerical_features_pairplot_{i + 1}.png')
        plt.close()

def plot_categorical_features(train_data, target_col, cat_cols, eda, max_plots_per_file=20):
    
    print(f'EDA: Plotting categorical features ({eda}) for columns: {cat_cols}..')

    # Determine the number of features to plot
    print(cat_cols)
    num_features = len(cat_cols)
    file_count = 1
    
    # Loop through categorical columns in batches
    for start in range(0, num_features, max_plots_per_file):
        end = min(start + max_plots_per_file, num_features)  # Determine the range for this batch
        plt.figure(figsize=(15, 5 * ((end - start) // 3 + 1)))  # Adjust figure size based on the number of plots

        for i, col in enumerate(cat_cols[start:end]):
            plt.subplot((end - start + 2) // 3, 3, i + 1)  # Create subplots in 3 columns
            sns.countplot(x=target_col, hue=col, data=train_data)
            plt.title(f'Distribution of {col} by {target_col}')
            plt.xlabel(target_col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.legend(title=col)

        # Save each batch of plots into a separate file
        plt.savefig(f'../eda/{eda}/categorical_countplot_part{file_count}.png')  
        plt.close()
        file_count += 1

def plot_categorical_numerical_interactions(train_data, target_col, cat_cols, num_cols, eda):
    # Detect categorical and numerical columns
    print(f'EDA: Plotting categorical/numerical features ({eda})..')
    
    # Determine the number of features to plot and create a grid layout
    max_plots_per_row = 2  # Number of plots per row
    num_rows = int(np.ceil(len(cat_cols) * len(num_cols) / max_plots_per_row))  # Grid layout for each plot type
    
    # Box Plots
    plt.figure(figsize=(15, 5 * num_rows))
    plot_idx = 1
    for i, cat_col in enumerate(cat_cols):
        for j, num_col in enumerate(num_cols):
            plt.subplot(num_rows, max_plots_per_row, plot_idx)
            sns.boxplot(x=cat_col, y=num_col, hue=target_col, data=train_data, order=sorted(train_data[cat_col].unique()))
            plt.title(f'Box Plot of {num_col} by {cat_col}')
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.xticks(rotation=45)
            plot_idx += 1
    plt.tight_layout()
    plt.savefig(f'../eda/{eda}/categorical_numerical_boxplot.png')  
    plt.close()

def plot_single_col_boxplot(data, col, target_col, eda):
    print('EDA: Plotting single plot..')
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_col, y=col, data=data, hue=target_col, legend=False, showmeans=True, showcaps=True)
    plt.title(f'Boxplot of {col} by {target_col}')
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.savefig(f'../eda/{eda}/boxplot_{col}', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    
def plot_skewness(X_train_data, eda):
    print(f'EDA: Plotting skewness ({eda})..')
    [remove(join(f'../eda/{eda}/col_dist/', f)) for f in listdir(f'../eda/{eda}/col_dist/') if isfile(join(f'../eda/{eda}/col_dist/', f))]
    for col in X_train_data.columns:
        if X_train_data[col].dtype in [np.int64, np.float64]:
            skewness = round(X_train_data[col].skew(), 2)
            sns.histplot(X_train_data[col], kde=True)
            plt.title(f'Distribution of {col} (skew = {skewness})')
            plt.savefig(f'../eda/{eda}/col_dist/skew={skewness}_col={col}.png')
            plt.close()

def plot_mi(X_train, y_train, eda):
    print(f'EDA: Plotting MI scores ({eda})..')
        
    # Ensure categorical variables are encoded
    X_train_encoded = get_dummies(X_train, drop_first=True)  # Assuming X_train_data includes categorical vars
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X_train_encoded, y_train, discrete_features='auto')
    mi_scores = Series(mi_scores, name="MI Scores", index=X_train_encoded.columns)
    
    for t in ['min', 'max']:
        if t == 'max':
            scores = mi_scores.nlargest(10)
        else:
            scores = mi_scores.nsmallest(10)

        plt.figure(dpi=100, figsize=(30, 7))
        plt.barh(scores.index, scores.values)
        plt.xlabel("Mutual Information Score")
        plt.title(f"Mutual Information Scores ({t})")
        plt.savefig(f'../eda/{eda}/mutual_information_{t}.png')
        plt.close()
    del X_train_encoded

def plot_corr(X_train_data, eda):
    print(f'EDA: Plotting correlations ({eda})..')
    X_train_encoded = get_dummies(X_train_data, drop_first=True)
    corr_matrix = X_train_encoded.corr()
    corr_matrix = corr_matrix.where(np.triu(np.abs(corr_matrix) > 0.5, k=1))
    plt.figure(figsize=(40, 20))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=1)
    plt.title('Significant correlation matrix heatmap')
    plt.savefig(f'../eda/{eda}/correlation_matrix.png')
    plt.close()

# Main EDA function
def eda(data_map, runtime_map):
    # Set variables based on maps
    X_train, y_train = data_map['X_train'], data_map['y_train']
    eda = 'processed' if data_map['num_cols_engineered'] else 'unprocessed'
    [num_plot, cat_plot, mixed_plot, single_plot, skew, mi, corr, imp] = runtime_map['plots']
    target_col = data_map['target_cols'][0]                                      # For readability
    eda_when = runtime_map['eda_when'] 

    if eda == 'unprocessed':
        num_cols, cat_cols = data_map['num_cols_raw'], data_map['cat_cols_raw']
        print(f"Numerical Cols: {num_cols}")
    else:
        num_cols, cat_cols = data_map['num_cols_engineered'], data_map['cat_cols_engineered']
    
    if eda_when == 'both' or (eda_when == 'after' and eda == 'processed') or (eda_when == 'before' and eda == 'unprocessed'):
        print(f'Running EDA for {eda}..')

        train_data = concat([X_train, y_train], axis=1)

        # plot_features_vs_target(train_data, num_cols, cat_cols, target_col, eda)
        

        if num_plot:
            plot_numerical_features(train_data, target_col, num_cols, eda)

        if cat_plot:
            plot_categorical_features(train_data, target_col, cat_cols, eda)

        if mixed_plot:
            plot_categorical_numerical_interactions(train_data, target_col, cat_cols, num_cols, eda)
        
        if skew: # Calculate skewness for all columns and visualize the column distributions
            plot_skewness(X_train, eda)
        
        if mi:  # Calculate mutual information scores
            plot_mi(X_train, y_train, eda) 

        if corr: # Investigate_correlations
            plot_corr(X_train, eda)
        
        del train_data

        print(f'EDA: eda for {eda} done.\n')

    else:
        print(f'Skipping EDA for {eda}..')
    
    

