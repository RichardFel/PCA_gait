'''
Settings

Data 2-min walk test:
if with/without aid only include with aid

Data physical actiity:
Only include parts with kmph >= 0.2

PCA: 
Explained variance of 1 (now 2), total explained variance of 80%
Varimax rotation, optinal is promax 

LMEM:
Only participants with at least 2 measurements

'''

#%%
import pandas as pd
import numpy as np
import scipy

from sklearn.decomposition import PCA
from sklearn import preprocessing

from factor_analyzer import Rotator
from factor_analyzer.factor_analyzer import calculate_kmo

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import cross_val_score
from statsmodels.stats.anova import anova_lm

import pingouin as pg

from Functions.load_files import *
from Functions.PCA import *
from Functions.LMM import *
from Functions.test_retest import *
from Functions.description import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import statsmodels.api as sm
from statsmodels.formula.api import logit
import statsmodels.formula.api as smf
import seaborn as sns
verbose = True

from sklearn.linear_model import LinearRegression
#%%

# load files
new_selection,selection  = load_file(file = '0.95', verbose = verbose)

# Describe data
describe_data(new_selection, selection)

# Calculates sphericity
new_selection = sphericity_kmo(new_selection)

# Rescales data 
scaledData, scaledData_df, tmp_scaler = scaler(new_selection)

# Creates PCA to determine number of components
number_of_components = compute_pca(new_selection, scaledData, verbose = True, visualise = False)

# Creates second PCA with max number of components
pca, loadings, pca_data = compute_pca2(number_of_components, scaledData, new_selection)

# Used to determine the loadings and features that load strongly
features, loadings_corr = cor_comp_var(pca, new_selection)
loadings_corr = pca.components_.T

# Rotates the PCA
rotated_loadings_df, rotated_loadings, rotator = rotate_pca(loadings_corr, new_selection)

# Make a bi plot
# bi_plot(rotated_loadings_df)

# calculate the icc of the test-retest 
pca_test_hertest_df = load_rotate_test_hertest(new_selection, tmp_scaler, pca)
icc_mae(pca_test_hertest_df, number_of_components, verbose = True)


# Add kmph and prepare for 
pca_kmph = add_kmph(scaledData_df, rotated_loadings, new_selection, selection)
totaal = pca_kmph.merge(selection, how = 'left', on = ['Subject number', 'T moment', 'aid'])
tmp = totaal.corr()
tmp = tmp.abs()
for column in tmp.columns:
    tmp.loc[tmp[column] < 0.3, column] = -1

tmp.to_excel('Excel files/correlation/correlation_totaal.xlsx')


#%%
# Association
pca_daily_life = pca_kmph.merge(selection[['Subject number', 'T moment', 'aid',
                                           'KMPH R']], how = 'left', on = ['Subject number', 'T moment', 'aid'])
pca_daily_life.to_excel('Excel files/results/pca_daily_life.xlsx')
pca_daily_life.rename(columns = {
    'mean_gait_speed_DL' : 'Average gait speed',
    'max_gait_speed_DL' : 'Maximum gait speed', 
    'steps_per_day' : 'Number of steps (daily)', 
    'PCA_0' : 'PC0: Tempo', 
    'PCA_1' : 'PC1: Asymmetry', 
    'PCA_2' : 'PC2: Postural stability',
    'PCA_3' : 'PC3: Trunk movement', 
    'PCA_4' : 'PC4: Variability', 
    'PCA_5' : 'PC5: Rhythm', 
    'PCA_6' : 'PC6: Intensity', 
    'PCA_7' : 'PC7: Stride distance', 
    'PCA_8' : 'PC8: Regularity', 
    'KMPH R': 'Gait speed'},
    inplace = True)

mask = np.triu(np.ones_like(pca_daily_life.corr(), dtype=np.bool))
pca_daily_life.corr().round(2).to_excel('Excel files/correlation/correlation_association.xlsx')

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(18, 9))
sns.heatmap(pca_daily_life.corr().iloc[1:,:-1], vmin=-1, vmax=1, mask=mask[1:,:-1], annot=True, cmap='seismic', ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment="right") 
ax.tick_params(axis='both', which='major', labelsize=23)
plt.tight_layout()
plt.savefig('Images/heatmap.pdf', dpi=300, format='pdf')


def root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    return round(rmse,3)
    
def mean_absolute_error(predictions, targets):
    mae = np.average(np.abs(predictions - targets))   
    return round(mae,3)
#%%
# Linear regression model
mse = []
rmse_list = []
mae_list = []
n = len(totaal)
results = {}

# General settings
data = pca_kmph
data =  data.merge(totaal[['Subject number', 'T moment', 'aid', 'KMPH R']], how = 'left', on = ['Subject number', 'T moment', 'aid'])
data['aid'].replace(['Nee', 'Ja'],[0, 1], inplace=True)
columns = ['steps_per_day', 'mean_gait_speed_DL', 'max_gait_speed_DL'] # 'mean_gait_speed_DL',
subj_list = pca_kmph['Subject number'].unique()
per_k = 5

correlation = data.corr().abs()
tmp_corr = correlation['KMPH R'][4:-1].loc[correlation['KMPH R'][4:-1] > 0.9]
correlation = correlation.loc[~correlation.index.isin(tmp_corr.index.values)]

# Define difinitive models
base_model = ['KMPH R', 'aid']

#%%
results = {}
for columnToPredict in columns:
    if columnToPredict == 'steps_per_day':
        data[[columnToPredict]] = data[[columnToPredict]].astype(int)
    for column in correlation.index[4:-1]:
        incl_components = base_model + [column]
        for run in range(10):
            actual_list_c = []
            predicted_list_c = []

            # Randomise the order of the data
            data = data.sample(frac=1) 
            np.random.shuffle(subj_list)       

            for i in range(int(len(subj_list) / 5)):
                # Select subjests 
                subj = subj_list[i*5:i*5+5]

                # Train/test split
                test = data.loc[data['Subject number'].isin(subj)]
                train = data.loc[~data['Subject number'].isin(test['Subject number'])]

                # Uses all data of the test set and max 1 measurrement per subject per moment of the train set
                train = train.drop_duplicates(subset = ['Subject number', 'T moment'])

                ### Definitive combined model ###
                # Prepare train data set
                x = train.loc[:,incl_components]
                # _, x, tmp_scaler = scaler(x)
                y = train[[columnToPredict]]

                # PLS on train dataset
                model = LinearRegression().fit(x, y)

                # Prepare test data set
                x_test = test.loc[:,incl_components]
                # x_test = tmp_scaler.transform(x_test)
                y_test = test[[columnToPredict]]

                # Predict results 
                predicted = model.predict(x_test)
                actual = y_test

                # Add values to list
                for val_1, val_2 in zip(actual.values, predicted):
                    actual_list_c.append(val_1[0])
                    predicted_list_c.append(val_2[0])
            actual_list_c = np.array(actual_list_c)
            predicted_list_c = np.array(predicted_list_c)

            RMSE_c = root_mean_square_error(actual_list_c, predicted_list_c)
            MAE_c = mean_absolute_error(actual_list_c, predicted_list_c)

            # NRMSE = norm_root_mean_square_error(actual_list, predicted_list)
            key = f'{columnToPredict}_{run}_{column}'
            results[key] = [columnToPredict, run, column, RMSE_c, MAE_c ]

test_results_df = pd.DataFrame.from_dict(results,orient='index', columns = ['columnToPredict', 'run', 'column','RMSE_c', 
                                        'MAE_c',])
print(test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].mean())
print(test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].std())

outcomes_1 = test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].mean()
outcomes_2 = test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].std()
baseline = [0.49, 0.31, 1516]
columns = ['max_gait_speed_DL', 'mean_gait_speed_DL', 'steps_per_day']

for i in range(7):
    print(f'component {i} &&')
    for column in columns:
        if column == 'steps_per_day':
            print(f' {round(outcomes_1[column][i],1)} ({round(outcomes_2[column][i],1)}) & {round(baseline[2] - outcomes_1[column][i],1)} [{round((baseline[2] - outcomes_1[column][i]) / baseline[2] * 100,1)}%] &&')
        elif column == 'mean_gait_speed_DL':
            print(f' {round(outcomes_1[column][i],3)} ({round(outcomes_2[column][i],3)}) & {round(baseline[1] - outcomes_1[column][i],3)} [{round((baseline[1] - outcomes_1[column][i]) / baseline[1] * 100,1)}%] &&')
        else:
            print(f' {round(outcomes_1[column][i],3)} ({round(outcomes_2[column][i],3)}) & {round(baseline[0] - outcomes_1[column][i],3)} [{round((baseline[0] - outcomes_1[column][i]) / baseline[0] * 100,1)}%] &&')


#%%
# Definitive models, all data

def add_identity(axes, *line_args, **line_kwargs):
    # Adds a diagonal line to a figure
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

for num, columnToPredict in enumerate(columns):
    actual_list_c = []
    predicted_list_c = []
    actual_list_i = []
    predicted_list_i = []
    actual_list_g = []
    predicted_list_g = []

    if columnToPredict == 'steps_per_day':
        data[[columnToPredict]] = data[[columnToPredict]].astype(int)
        fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(16, 6))
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)
        ax[2].set_box_aspect(1)
    else:
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)

    incl_components = ['KMPH R', 'aid', 'PCA_4']

    ### Combined model ###
    x = data.loc[:,incl_components]
    # _, x, tmp_scaler = scaler(x)
    y = data[[columnToPredict]]

    # PLS on train dataset
    model = LinearRegression().fit(x, y)

    # Prepare test data set
    x_test = data.loc[:,incl_components]
    # x_test = tmp_scaler.transform(x_test)
    y_test = data[[columnToPredict]]

    # Predict results 
    predicted = model.predict(x_test)
    score_c = model.score(x_test, y_test)
    n = len(y_test)
    p = len(model.coef_)
    Adj_score_c = 1-(1-score_g)*(n-1)/(n-p-1)
    actual = y_test

    # Add values to list
    for val_1, val_2 in zip(actual.values, predicted):
        actual_list_c.append(val_1[0])
        predicted_list_c.append(val_2[0])

    ### Intercept only model ###
    y = data[[columnToPredict]].mean()

    # # Prepare test data set
    actual = data[[columnToPredict]]

    predicted = np.ones(len(actual)) * y[0]
    for val_1, val_2 in zip(actual.values, predicted):
        actual_list_i.append(val_1[0])
        predicted_list_i.append(val_2)

    ### Gait speed only ###
    # Prepare train data set
    x = data[['KMPH R', 'aid']]
    # _, x, tmp_scaler = scaler(x)
    y = data[[columnToPredict]]

    # PLS on train dataset
    model = LinearRegression().fit(x, y)

    # Prepare test data set
    x_test = data[['KMPH R', 'aid']]
    # x_test = tmp_scaler.transform(x_test)
    y_test = data[[columnToPredict]]

    # Predict results 
    predicted = model.predict(x_test)
    score_g = model.score(x_test, y_test)
    n = len(y_test)
    p = len(model.coef_)
    Adj_score_g = 1-(1-score_g)*(n-1)/(n-p-1)
    actual = y_test

    # Add values to list
    for val_1, val_2 in zip(actual.values, predicted):
        actual_list_g.append(val_1[0])
        predicted_list_g.append(val_2[0])

    actual_list_c = np.array(actual_list_c)
    actual_list_i = np.array(actual_list_i)
    actual_list_g = np.array(actual_list_g)
    predicted_list_c = np.array(predicted_list_c)
    predicted_list_i = np.array(predicted_list_i)
    predicted_list_g = np.array(predicted_list_g)

    RMSE_c = root_mean_square_error(actual_list_c, predicted_list_c)
    RMSE_i = root_mean_square_error(actual_list_i, predicted_list_i)
    RMSE_g = root_mean_square_error(actual_list_g, predicted_list_g)
    MAE_c = mean_absolute_error(actual_list_c, predicted_list_c)
    MAE_i = mean_absolute_error(actual_list_i, predicted_list_i)
    MAE_g = mean_absolute_error(actual_list_g, predicted_list_g)
    NRMSE_c = root_mean_square_error(actual_list_c, predicted_list_c) / np.mean(actual_list_c)
    NRMSE_i = root_mean_square_error(actual_list_i, predicted_list_i) / np.mean(actual_list_c)
    NRMSE_g = root_mean_square_error(actual_list_g, predicted_list_g) / np.mean(actual_list_c)

    ax[0].plot(actual_list_i, predicted_list_i, marker='.', linestyle='')
    ax[0].set_xlabel('Actual scores')
    ax[0].set_ylabel('Predicted scores')
    ax[0].set_title('Intercept only')

    ax[1].plot(actual_list_g, predicted_list_g, marker='.', linestyle='')
    ax[1].set_xlabel('Actual scores')
    ax[1].set_ylabel('Predicted scores')
    ax[1].set_title('Gait speed only')
    add_identity(ax[0], color='black', ls=':')
    add_identity(ax[1], color='black', ls=':')

    print(f'Column {columnToPredict}')
    print(f'RMSE gs {RMSE_g}')
    print(f'MAE gs {MAE_g}')
    print(f'NRMSE gs {NRMSE_g}')
    print(f'R^2 gs {score_g}')
    print(f'adj R^2 gs {Adj_score_g}')
    print('\n')

    if columnToPredict == 'steps_per_day':
        ax[2].plot(actual_list_c, predicted_list_c, marker='.', linestyle='')
        ax[2].set_xlabel('Actual scores')
        ax[2].set_ylabel('Predicted scores')
        ax[2].set_title('Combined model')
        ax[0].set_xlim(0,12000)
        ax[0].set_ylim(0,12000)
        ax[1].set_xlim(0,12000)
        ax[1].set_ylim(0,12000)
        ax[2].set_xlim(0,12000)
        ax[2].set_ylim(0,12000)
        add_identity(ax[2], color='black', ls=':')

        print(f'RMSE C {RMSE_c}')
        print(f'MAE C {MAE_c}')
        print(f'NRMSE C {NRMSE_c}')
        print(f'R^2 C {score_c}')
        print(f'adj R^2 C {Adj_score_g}')
        print('\n')
    elif columnToPredict == 'mean_gait_speed_DL':
        ax[0].set_xlim(0,3)
        ax[0].set_ylim(0,3)
        ax[1].set_xlim(0,3)
        ax[1].set_ylim(0,3)
    else:
        ax[0].set_xlim(1,4.5)
        ax[0].set_ylim(1,4.5)
        ax[1].set_xlim(1,4.5)
        ax[1].set_ylim(1,4.5)

    plt.savefig(f'Images/prediction_{columnToPredict}.pdf', dpi=300, format='pdf')


#%% 
# Loop over different dependent variables




for num, columnToPredict in enumerate(columns):
    results = {}

    if columnToPredict == 'steps_per_day':
        data[[columnToPredict]] = data[[columnToPredict]].astype(int)
        fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(16, 6))
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)
        ax[2].set_box_aspect(1)
    else:
        fig, ax = plt.subplots(ncols=2, sharey=True)
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)

    incl_components = ['KMPH R', 'aid', 'PCA_4']
    # Repeat this loop 100 times as there is some random factor in the data
    for run in range(100):
        actual_list_c = []
        predicted_list_c = []
        actual_list_i = []
        predicted_list_i = []
        actual_list_g = []
        predicted_list_g = []

        # Randomise the order of the data
        data = data.sample(frac=1) 
        np.random.shuffle(subj_list)

        for i in range(int(len(subj_list) / 5)):
            # Select subjests 
            subj = subj_list[i*5:i*5+5]

            # Train/test split
            test = data.loc[data['Subject number'].isin(subj)]
            train = data.loc[~data['Subject number'].isin(test['Subject number'])]

            # Uses all data of the test set and max 1 measurrement per subject of the train set
            train = train.drop_duplicates(subset = ['Subject number', 'T moment'])

            ### Definitive combined model ###
            # Prepare train data set
            x = train.loc[:,incl_components]
            # _, x, tmp_scaler = scaler(x)
            y = train[[columnToPredict]]

            # PLS on train dataset
            model = LinearRegression().fit(x, y)

            # Prepare test data set
            x_test = test.loc[:,incl_components]
            # x_test = tmp_scaler.transform(x_test)
            y_test = test[[columnToPredict]]

            # Predict results 
            predicted = model.predict(x_test)
            actual = y_test

            # Add values to list
            for val_1, val_2 in zip(actual.values, predicted):
                actual_list_c.append(val_1[0])
                predicted_list_c.append(val_2[0])

            ### Intercept-only model ###
            # # Prepare train data set
            y = train[[columnToPredict]].mean()

            # # Prepare test data set
            actual = test[[columnToPredict]]

            predicted = np.ones(len(actual)) * y[0]
            for val_1, val_2 in zip(actual.values, predicted):
                actual_list_i.append(val_1[0])
                predicted_list_i.append(val_2)

            ### Gait speed only ###
            # Prepare train data set
            x = train[['KMPH R', 'aid']]
            # _, x, tmp_scaler = scaler(x)
            y = train[[columnToPredict]]

            # PLS on train dataset
            model = LinearRegression().fit(x, y)

            # Prepare test data set
            x_test = test[['KMPH R', 'aid']]
            # x_test = tmp_scaler.transform(x_test)
            y_test = test[[columnToPredict]]

            # Predict results 
            predicted = model.predict(x_test)
            actual = y_test

            # Add values to list
            for val_1, val_2 in zip(actual.values, predicted):
                actual_list_g.append(val_1[0])
                predicted_list_g.append(val_2[0])

        actual_list_c = np.array(actual_list_c)
        actual_list_i = np.array(actual_list_i)
        actual_list_g = np.array(actual_list_g)
        predicted_list_c = np.array(predicted_list_c)
        predicted_list_i = np.array(predicted_list_i)
        predicted_list_g = np.array(predicted_list_g)

        RMSE_c = root_mean_square_error(actual_list_c, predicted_list_c)
        RMSE_i = root_mean_square_error(actual_list_i, predicted_list_i)
        RMSE_g = root_mean_square_error(actual_list_g, predicted_list_g)
        MAE_c = mean_absolute_error(actual_list_c, predicted_list_c)
        MAE_i = mean_absolute_error(actual_list_i, predicted_list_i)
        MAE_g = mean_absolute_error(actual_list_g, predicted_list_g)
        NRMSE_c = root_mean_square_error(actual_list_c, predicted_list_c) / np.mean(actual_list_c)
        NRMSE_i = root_mean_square_error(actual_list_i, predicted_list_i) / np.mean(actual_list_c)
        NRMSE_g = root_mean_square_error(actual_list_g, predicted_list_g) / np.mean(actual_list_c)

        # NRMSE = norm_root_mean_square_error(actual_list, predicted_list)
        key = f'{columnToPredict}_{run}'
        results[key] = [columnToPredict, run, RMSE_c, RMSE_i, RMSE_g,MAE_c, MAE_i, MAE_g, NRMSE_c, NRMSE_i, NRMSE_g ]

    ax[0].plot(actual_list_i, predicted_list_i, marker='o', linestyle='')
    ax[0].set_xlabel('Actual scores')
    ax[0].set_ylabel('Predicted scores')
    ax[0].set_title('Intercept only')

    ax[1].plot(actual_list_g, predicted_list_g, marker='o', linestyle='')
    ax[1].set_xlabel('Actual scores')
    ax[1].set_ylabel('Predicted scores')
    ax[1].set_title('Gait speed only')

    if columnToPredict == 'steps_per_day':
        ax[2].plot(actual_list_c, predicted_list_c, marker='o', linestyle='')
        ax[2].set_xlabel('Actual scores')
        ax[2].set_ylabel('Predicted scores')
        ax[2].set_title('Combined model')
        ax[0].set_xlim(0,12000)
        ax[0].set_ylim(0,12000)
        ax[1].set_xlim(0,12000)
        ax[1].set_ylim(0,12000)
        ax[1].set_xlim(0,12000)
        ax[1].set_ylim(0,12000)
    elif columnToPredict == 'mean_gait_speed_DL':
        ax[0].set_xlim(0,3)
        ax[0].set_ylim(0,3)
        ax[1].set_xlim(0,3)
        ax[1].set_ylim(0,3)
    else:
        ax[0].set_xlim(1,4.5)
        ax[0].set_ylim(1,4.5)
        ax[1].set_xlim(1,4.5)
        ax[1].set_ylim(1,4.5)

# results_df = pd.DataFrame.from_dict(results,orient='index', columns = ['columnToPredict', 'run', 'RMSE_c', 
#                                         'RMSE_i','RMSE_g','MAE_c', 'MAE_i', 'MAE_g', 'NRMSE_c', 'NRMSE_i', 'NRMSE_g'])
# print(results_df.groupby(['columnToPredict'])['RMSE_c'].mean())
# print(results_df.groupby(['columnToPredict'])['RMSE_g'].mean())
# print(results_df.groupby(['columnToPredict'])['RMSE_i'].mean())

# print(results_df.groupby(['columnToPredict'])['MAE_c'].mean())
# print(results_df.groupby(['columnToPredict'])['MAE_g'].mean())
# print(results_df.groupby(['columnToPredict'])['MAE_i'].mean())

# print(results_df.groupby(['columnToPredict'])['NRMSE_c'].mean())
# print(results_df.groupby(['columnToPredict'])['NRMSE_g'].mean())
# print(results_df.groupby(['columnToPredict'])['NRMSE_i'].mean())
# # results_df.to_csv('test_results.csv')
# fig.


#%%
results = {}
base_model = ['KMPH R', 'aid', 'PCA_4']
for columnToPredict in columns:
    if columnToPredict == 'steps_per_day':
        data[[columnToPredict]] = data[[columnToPredict]].astype(int)
    for column in correlation.index[4:-1]:
        incl_components = base_model + [column]
        for run in range(100):
            actual_list_c = []
            predicted_list_c = []

            # Randomise the order of the data
            data = data.sample(frac=1) 
            np.random.shuffle(subj_list)       

            for i in range(int(len(subj_list) / 5)):
                # Select subjests 
                subj = subj_list[i*5:i*5+5]

                # Train/test split
                test = data.loc[data['Subject number'].isin(subj)]
                train = data.loc[~data['Subject number'].isin(test['Subject number'])]

                # Uses all data of the test set and max 1 measurrement per subject per moment of the train set
                train = train.drop_duplicates(subset = ['Subject number', 'T moment'])

                ### Definitive combined model ###
                # Prepare train data set
                x = train.loc[:,incl_components]
                # _, x, tmp_scaler = scaler(x)
                y = train[[columnToPredict]]

                # PLS on train dataset
                model = LinearRegression().fit(x, y)

                # Prepare test data set
                x_test = test.loc[:,incl_components]
                # x_test = tmp_scaler.transform(x_test)
                y_test = test[[columnToPredict]]

                # Predict results 
                predicted = model.predict(x_test)
                actual = y_test

                # Add values to list
                for val_1, val_2 in zip(actual.values, predicted):
                    actual_list_c.append(val_1[0])
                    predicted_list_c.append(val_2[0])
            actual_list_c = np.array(actual_list_c)
            predicted_list_c = np.array(predicted_list_c)

            RMSE_c = root_mean_square_error(actual_list_c, predicted_list_c)
            MAE_c = mean_absolute_error(actual_list_c, predicted_list_c)

            # NRMSE = norm_root_mean_square_error(actual_list, predicted_list)
            key = f'{columnToPredict}_{run}_{column}'
            results[key] = [columnToPredict, run, column, RMSE_c, MAE_c ]

test_results_df = pd.DataFrame.from_dict(results,orient='index', columns = ['columnToPredict', 'run', 'column','RMSE_c', 
                                        'MAE_c',])
print(test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].mean())
print(test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].std())

outcomes_1 = test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].mean()
outcomes_2 = test_results_df.groupby(['columnToPredict','column'])['RMSE_c'].std()
baseline = [0.49, 0.31, 1454]
columns = ['max_gait_speed_DL', 'mean_gait_speed_DL', 'steps_per_day']

for i in range(7):
    print(f'component {i} &&')
    for column in columns:
        if column == 'steps_per_day':
            print(f' {round(outcomes_1[column][i],1)} ({round(outcomes_2[column][i],1)}) & {round(baseline[2] - outcomes_1[column][i],1)} [{round((baseline[2] - outcomes_1[column][i]) / baseline[2] * 100,1)}%] &&')
        elif column == 'mean_gait_speed_DL':
            print(f'&&&')
        else:
            print(f'&&&')


#%%


# print results
columns = ['steps_per_day_y', 'mean_gait_speed_DL_y', 'max_gait_speed_DL_y']

# results_df = pd.read_csv('test_results.csv', index_col=0)
outcome_1 = results_df.groupby(['columnToPredict', 'n_components'])['RMSE'].mean()
outcome_2 =  results_df.groupby(['columnToPredict', 'n_components'])['RMSE'].std()

outcome_3 = results_df.groupby(['columnToPredict', 'n_components'])['NRMSE'].mean()
outcome_4 =  results_df.groupby(['columnToPredict', 'n_components'])['NRMSE'].std()

for i in range(5):
    print(f'component {i}')
    for j in columns:
        if j == 'steps_per_day_y':
            print(f' {round(outcome_1[j][i+1],1)} & {round(outcome_2[j][i+1],1)} &&')
        else:
            print(f' {round(outcome_1[j][i+1],3)} & {round(outcome_2[j][i+1],3)} &&')



# Gait speed only model
subj_list = pca_kmph['Subject number'].unique()
two_measurement = subj_list[pca_kmph.groupby(['Subject number'])['mean_gait_speed_DL'].count() > 0]

# Columns: mean_gait_speed_DL, max_gait_speed_DL, mode_gait_speed_DL, steps_per_day, minutes_per_day
columnToPredict = 'steps_per_day'
randomEffect= 'Subject number'

# Liniar mixed effects models to predict kmph in daily life from gait speed 2min
pca_daily_life = pca_daily_life.dropna(subset = [columnToPredict])
pca_daily_life.rename(columns = {'KMPH R': 'kmph'}, inplace = True)
pca_daily_life = pca_daily_life.loc[pca_daily_life['Subject number'].isin(two_measurement)]


# Exclude C0 & C4
# high_icc = high_icc[1:4]

# df = pca_daily_life.sample(frac=1).drop_duplicates(subset = ['Subject number'])
df = pca_daily_life
X_train = df[['kmph']]
X_train = sm.add_constant(X_train)
y_train = df[columnToPredict]
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

actual_1 = model.predict(X_train)
predicted_1 = y_train
print(f'RMSE: {root_mean_square_error(predicted_1, actual_1)}')
print(f'MAE: {mean_absolute_error(predicted_1, actual_1)}')

# Default : X_train = df[['PCA_1', 'PCA_2','PCA_3', 'kmph']]
# Max gait speed: X_train = df[['PCA_3', 'kmph']]
# Mean gait speed = X_train = df[['PCA_2','PCA_3', 'kmph']]
# Steps per day = 

X_train = df[['PCA_3', 'kmph']]
X_train = sm.add_constant(X_train)
y_train = df[columnToPredict]
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
predicted = model.predict(X_train)
actual = y_train
print(f'RMSE: {root_mean_square_error(predicted, actual)}')
print(f'MAE: {mean_absolute_error(predicted, actual)}')


if columnToPredict == 'max_gait_speed_DL':
    min = 1
    max = 4.5 
elif columnToPredict == 'mean_gait_speed_DL':
    min = 0
    max = 2.5 
else:
    min = 0
    max = 12000  

fig_f, ax_f = plt.subplots()
ax_f.plot(actual_1,predicted_1 , linestyle = 'None', marker = 'o',)
ax_f.set_title('True vs predicted: Gait speed only model')
ax_f.set_xlabel('Predicted')
ax_f.set_ylabel('Actual ')
ax_f.set_ylim(min,max)
ax_f.set_xlim(min,max)

fig_f, ax_f = plt.subplots()
ax_f.plot(actual,predicted , linestyle = 'None', marker = 'o',)
ax_f.set_title('True vs predicted: Combined model')
ax_f.set_xlabel('Actual')
ax_f.set_ylabel('Predicted')
ax_f.set_ylim(min,max)
ax_f.set_xlim(min,max)

#%%
# Partial least squares
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics

mse = []
n = len(totaal)

totaal2 = totaal.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
       'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
       'PCA_3', 'PCA_4'])


# Columns: mean_gait_speed_DL, max_gait_speed_DL, mode_gait_speed_DL, steps_per_day, minutes_per_day
columnToPredict = 'steps_per_day_y'
x = totaal2.iloc[:,:70]
_, x, _ = scaler(x)
y = totaal2[[columnToPredict]].astype(float)

r_2 = []
for i in range(1,20):
    pls = PLSRegression(n_components=i)
    pls.fit(x,y)
    r_2.append(pls.score(x,y))

fig_f, ax_f = plt.subplots()
x_axis = np.arange(1,20)
ax_f.plot(x_axis, r_2 ,  linestyle = 'None', marker = 'o',)
ax_f.set_title('R^2 per number of components')
ax_f.set_xlabel('Number of components')
ax_f.set_ylabel('R^2')

pls = PLSRegression(n_components=10)
pls.fit(x,y)

predicted = pls.predict(x)
actual = y

if columnToPredict == 'max_gait_speed_DL_y':
    min = 1
    max = 4.5 
elif columnToPredict == 'mean_gait_speed_DL_y':
    min = 0
    max = 2.5 
else:
    min = 0
    max = 12000  

fig_f, ax_f = plt.subplots()
ax_f.plot(actual,predicted , linestyle = 'None', marker = 'o',)
ax_f.set_title('True vs predicted: PLS 10 components')
ax_f.set_xlabel('Actual')
ax_f.set_ylabel('Predicted')
ax_f.set_ylim(min,max)
ax_f.set_xlim(min,max)

r2 = pls.score(x,y)
n = len(y)
p = pls.n_components
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

print('\n')
print('PLS combined model')
print(f'R^2 : {r2}')
print(f'ADJ R^2 : {Adj_r2}')
print(f'RMSE: {root_mean_square_error(predicted, actual.values)}')
print(f'MAE: {mean_absolute_error(predicted, actual)}')
print('\n')


x = x.loc[:,'KMPH R'].values
x = np.reshape(x,(len(x),1))
y = totaal2[[columnToPredict]].astype(float)
pls = PLSRegression(n_components=1)
pls.fit(x,y)
print(f'PLS r^2 score {pls.score(x,y)}')
predicted = pls.predict(x)
actual = y
fig_f, ax_f = plt.subplots()
ax_f.plot(actual,predicted , linestyle = 'None', marker = 'o',)
ax_f.set_title('True vs predicted: PLS gait speed only')
ax_f.set_xlabel('Actual')
ax_f.set_ylabel('Predicted')
ax_f.set_ylim(min,max)
ax_f.set_xlim(min,max)

r2 = pls.score(x,y)
n = len(y)
p = len(model.coef_)
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

print('\n')
print('PLS Gait speed only model')
print(f'R^2 : {r2}')
print(f'ADJ R^2 : {Adj_r2}')
print(f'RMSE: {root_mean_square_error(predicted, actual.values)}')
print(f'MAE: {mean_absolute_error(predicted, actual)}')
print('\n')

#%%
r2_list = []
r2a_list = []
rmse_list = []
mae_list = []
mse_list =[]
actual_list = []
predicted_list = []

subj_list = pca_kmph['Subject number'].unique()
columnToPredict = 'max_gait_speed_DL_y'
if columnToPredict == 'steps_per_day_y':
    totaal[[columnToPredict]] = totaal[[columnToPredict]].astype(int)

per_k = 5

for i in range(1,11):
    actual_list = []
    predicted_list = []
    count = 0
    for _ in range(1):
        totaal = totaal.sample(frac=1)
        for subj in subj_list:
            count += 5
            # Train/test split
            test = totaal.loc[totaal['Subject number'].isin(subj)]
            train = totaal.loc[~totaal['Subject number'].isin(test['Subject number'])]
            train = train.drop_duplicates(subset = ['Subject number'])
            train = train.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
                'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
                'PCA_3', 'PCA_4'])
            test = test.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
                'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
                'PCA_3', 'PCA_4'])

            # Prepare train data set
            x = train.iloc[:,:70]
            _, x, tmp_scaler = scaler(x)
            y = train[[columnToPredict]]

            # PLS on train dataset
            pls = PLSRegression(n_components=i, tol = 1e-06)
            pls.fit(x,y)
            #pls.score(x,y)

            # Prepare test data set
            x_test = test.iloc[:,:70]
            x_test = tmp_scaler.transform(x_test)
            y_test = test[[columnToPredict]]

            # Predict results 
            if columnToPredict == 'steps_per_day_y':
                predicted = pls.predict(x_test).astype(int)
            else:
                predicted = pls.predict(x_test)
            actual = y_test
            # r2 = pls.score(x_test,y_test)
            # n = len(y)
            # p = pls.n_components
            # Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
            for val_1, val_2 in zip(actual.values, predicted):
                actual_list.append(val_1[0])
                predicted_list.append(val_2[0])

    actual_list = np.array(actual_list)
    predicted_list = np.array(predicted_list)
    RMSE = root_mean_square_error(actual_list, predicted_list)
    MAE = mean_absolute_error(actual_list, predicted_list)

    print(f'Components {i} RMSE: {RMSE}')
    print(f'Components {i} MAE: {MAE}')

plt.plot(actual_list, predicted_list)
# # mse = metrics.mean_squared_error(predicted,actual.values)    
# # r2_list.append(r2)
# # r2a_list.append(Adj_r2)
# rmse_list.append(RMSE)
# mae_list.append(MAE)
# # mse_list.append(mse)

# # print(r2_list)
# # print(r2a_list)
# print(rmse_list)
# print(mae_list)
# 
# for i in range(1,10):
#     score = -1*model_selection.cross_val_score(PLSRegression(n_components=i),
#             x, y, scoring='neg_mean_squared_error').mean()    
#     mse.append(score)
# plt.plot(mse)
# # totaal


#%%
# Linear mixed models
# subj_list = pca_kmph['Subject number'].unique()
# two_measurement = subj_list[pca_kmph.groupby(['Subject number'])['mean_gait_speed_DL'].count() > 1]

# # Columns: mean_gait_speed_DL, max_gait_speed_DL, mode_gait_speed_DL, steps_per_day, minutes_per_day
# columnToPredict = 'mean_gait_speed_DL'
# randomEffect= 'Subject number'

# # Liniar mixed effects models to predict kmph in daily life from gait speed 2min
# pca_daily_life = pca_daily_life.dropna(subset = [columnToPredict])
# pca_daily_life.rename(columns = {'KMPH R': 'kmph'}, inplace = True)
# pca_daily_life = pca_daily_life.loc[pca_daily_life['Subject number'].isin(two_measurement)]

# icc = pd.read_excel('Excel files/ICC/ICC_MAE.xlsx', index_col=0)
# high_icc = icc.columns[icc.loc['ICC'] > 0.75]
# # Exclude C0 & C4
# high_icc = high_icc[1:4]

# Gait speed only model
# columns = np.array(['kmph'])
# model_1 = create_lmm(pca_daily_life, columnToPredict, randomEffect, columns)
# fitted_model_1 = model_1.fit(reml=False)
# print(fitted_model_1.summary())

# # Combines model
# columns = np.append(['kmph'], high_icc.values )
# model_2 = create_lmm(pca_daily_life, columnToPredict, randomEffect, columns)
# fitted_model_2 = model_2.fit(reml=False)
# print(fitted_model_2.summary())

# log_model_1 = fitted_model_1.llf
# log_model_2 = fitted_model_2.llf

# LR_statistic = -2*(-log_model_2 - -log_model_1)
# print(LR_statistic)

# p_val = scipy.stats.chi2.sf(LR_statistic, 3)
# print(p_val)

# def bic(self):
#     """Bayesian information criterion"""
#     if self.reml:
#         return np.nan
#     if self.freepat is not None:
#         df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
#     else:
#         df = self.params.size + 1
#     return -2 * self.llf + np.log(self.nobs) * df

# def aic(self):
#     """Akaike information criterion"""
#     if self.reml:
#         return np.nan
#     if self.freepat is not None:
#         df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
#     else:
#         df = self.params.size + 1
#     return -2.0 * self.llf + 2.0 * df

# def root_mean_square_error(predictions, targets):
#     rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
#     return round(rmse,3)
    
# def mean_absolute_error(predictions, targets):
#     mae = np.average(np.abs(predictions - targets))   
#     return round(mae,3)

# print(f'Dependent variable: {columnToPredict}')
# print('Model 1:')
# print(f'LL:{fitted_model_1.llf}')
# print(f'AIc: {aic(fitted_model_1)}')
# print(f'BIC: {bic(fitted_model_1)}')
# print(f'RMSE: {root_mean_square_error(fitted_model_1.fittedvalues.values,pca_daily_life[columnToPredict].values)}')
# print(f'MAE: {mean_absolute_error(fitted_model_1.fittedvalues.values,pca_daily_life[columnToPredict].values)}')

# print('Model 2:')
# print(f'LL:{fitted_model_2.llf}')
# print(f'AIc: {aic(fitted_model_2)}')
# print(f'BIC: {bic(fitted_model_2)}')
# print(f'RMSE: {root_mean_square_error(fitted_model_2.fittedvalues.values,pca_daily_life[columnToPredict].values)}')
# print(f'MAE: {mean_absolute_error(fitted_model_2.fittedvalues.values,pca_daily_life[columnToPredict].values)}')

# print('Comparison')

# print(LR_statistic)
# print(p_val)

#%%

# Step 1 summary of the model:

# stop = False
# try:
#     while stop == False:
#         highest_p_value = fitted_model.pvalues.iloc[1:-1].max()
#         highest_p_idx = fitted_model.pvalues.iloc[1:-1].idxmax()
#         if highest_p_value > 0.05:
#             columns = np.delete(columns, np.where(columns == highest_p_idx)[0])
#         else:
#             stop = True
#         model = create_lmm(data, columnToPredict, randomEffect, fixedEffects, columns)
#         fitted_model = model.fit()
# except:
#         model = create_lmm(data, columnToPredict, randomEffect, fixedEffects, columns)
#         fitted_model = model.fit()
# print(fitted_model.summary())

# # Step 2 check residuals:
# fig_r, ax_r = plt.subplots()
# plt.plot(fitted_model_2.resid, linestyle = 'None', marker = 'o',)
# ax_r.set_title('Residuals')
# ax_r.set_xlabel('Measurements')
# ax_r.set_ylabel('residual')

# # Step 3 plot actual vs predicted
# fig_f, ax_f = plt.subplots()
# plt.plot(fitted_model.fittedvalues.values,data[columnToPredict].values , linestyle = 'None', marker = 'o',)
# ax_f.set_title('True vs predicted')
# ax_f.set_xlabel('Measurements')
# ax_f.set_ylabel('Gait speed')
# ax_f.axis('equal')

# # Step 4 QQ-plot
# random_effects = fitted_model_2.random_effects
# mean_random_effects = [value[0] for key, value in random_effects.items()]
# mean_random_effects = np.array(mean_random_effects)
# sm.qqplot(mean_random_effects, fit=True, line = '45')


# #%%
# pca_kmph_two = pca_kmph.loc[pca_kmph['Subject number'].isin(two_measurement)]
# pca_kmph_two.reset_index(drop = True, inplace = True)
# gait_speed_two.reset_index(drop = True, inplace = True)
# pca_kmph_two = gait_speed_two.join(pca_kmph_two.iloc[:,8:])

# # Liniar mixed effects models to predict kmph in daily life from pcaa
# icc = pd.read_excel('Excel files/ICC/ICC_MAE.xlsx', index_col=0)
# high_icc = icc.columns[icc.loc['ICC'] > 0.75]

# correlaation = pca_kmph_two.corr()
# correlaation.to_excel('Excel files/correlation/correlation_pca_kmph.xlsx')

# data = pca_kmph_two
# fixedEffects = ''
# columns = np.append(['kmph'], high_icc.values )
# # columns = ['PCA_0', 'PCA_8']

# model = create_lmm(data, columnToPredict, randomEffect, fixedEffects, columns)
# fitted_model = model.fit()

# # Step 1 summary of the model:
# # print(fitted_model.summary())

# # step 2: loop to delete the component with the highest p-value
# stop = False
# while stop == False:
#     highest_p_value = fitted_model.pvalues.iloc[2:-1].max()
#     highest_p_idx = fitted_model.pvalues.iloc[2:-1].idxmax()
#     if highest_p_value > 0.05:
#         columns = np.delete(columns, np.where(columns == highest_p_idx)[0])
#     else:
#         stop = True
#     model = create_lmm(data, columnToPredict, randomEffect, fixedEffects, columns)
#     fitted_model = model.fit()
# print(fitted_model.summary())

# # # Step 2 check residuals:
# # fig_r, ax_r = plt.subplots()
# # plt.plot(fitted_model.resid, linestyle = 'None', marker = 'o',)
# # ax_r.set_title('Residuals')
# # ax_r.set_xlabel('Measurements')
# # ax_r.set_ylabel('residual')

# # # Step 3 plot actual vs predicted
# fig_f, ax_f = plt.subplots()
# plt.plot(fitted_model.fittedvalues.values,data[columnToPredict].values , linestyle = 'None', marker = 'o',)
# ax_f.set_title('True vs predicted')
# ax_f.set_xlabel('Measurements')
# ax_f.set_ylabel('Gait speed')
# ax_f.set_ylim(0,5)
# ax_f.set_xlim(0,5)

# # # Step 4 QQ-plot
# # random_effects = fitted_model.random_effects
# # mean_random_effects = [value[0] for key, value in random_effects.items()]
# # mean_random_effects = np.array(mean_random_effects)
# # sm.qqplot(mean_random_effects, fit=True, line = '45')



# # %%
# # Compare the two models
# log_model_1 = fitted_model.llf
# log_model_2 = fitted_model_2.llf

# LR_statistic = -2*(log_model_1-log_model_2)
# print(LR_statistic)

# p_val = scipy.stats.chi2.sf(LR_statistic, 2)
# print(p_val)

# # %%

