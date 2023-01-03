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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
totaal.corr().to_excel('Excel files/correlation/correlation_totaal.xlsx')


#%%
# Association
pca_daily_life = pca_kmph.merge(selection[['Subject number', 'T moment', 'aid',
                                           'KMPH R']], how = 'left', on = ['Subject number', 'T moment', 'aid'])
pca_daily_life.corr().round(2).to_excel('Excel files/correlation/correlation_association.xlsx')

def root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    return round(rmse,3)

def norm_root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    nrmse = rmse / (targets.max() - targets.min())
    return round(nrmse,3)

def mean_absolute_error(predictions, targets):
    mae = np.average(np.abs(predictions - targets))   
    return round(mae,3)

#%%
# Partial least squares determine nunmber of components 

import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics

mse = []
rmse_list = []
mae_list = []
n = len(totaal)
results = {}

# General settings
totaal2 = totaal.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
       'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
       'PCA_3', 'PCA_4'])
columns = ['steps_per_day_y', 'mean_gait_speed_DL_y', 'max_gait_speed_DL_y']
subj_list = pca_kmph['Subject number'].unique()
per_k = 5

# Loop over different dependent variables
for columnToPredict in columns:
    if columnToPredict == 'steps_per_day_y':
        totaal[[columnToPredict]] = totaal[[columnToPredict]].astype(int)

    # Create a loop with different number of components
    for n_components in range(1,6):

        # Repeat this loop 100 times as there is some random factor in the data
        for run in range(10):
            actual_list = []
            predicted_list = []
            # Randomise the order of the data
            totaal = totaal.sample(frac=1) 
            np.random.shuffle(subj_list)

            for i in range(int(len(subj_list) / 5)):
                # Select subjests 
                subj = subj_list[i*5:i*5+5]

                # Train/test split
                test = totaal.loc[totaal['Subject number'].isin(subj)]
                train = totaal.loc[~totaal['Subject number'].isin(test['Subject number'])]

                # Uses all data of the test set and max 1 measurrement per subject of the train set
                # train = train.drop_duplicates(subset = ['Subject number'])
                train = train.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
                    'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
                    'PCA_3', 'PCA_4'])
                test = test.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
                    'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
                    'PCA_3', 'PCA_4'])

                # Prepare train data set
                x = train.iloc[:,:5]
                _, x, tmp_scaler = scaler(x)
                y = train[[columnToPredict]]

                # PLS on train dataset
                pls = PLSRegression(n_components=i, tol = 0.001)
                pls.fit(x,y)
                #pls.score(x,y)

                # Prepare test data set
                x_test = test.iloc[:,:5]
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
            NRMSE = norm_root_mean_square_error(actual_list, predicted_list)
            MAE = mean_absolute_error(actual_list, predicted_list)

            # print(f'Variable: {columnToPredict} Run: {run} Components {n_components} RMSE: {RMSE}')
            # print(f'Variable: {columnToPredict} Run: {run} Components {n_components} MAE: {MAE}')

            key = f'{columnToPredict}_{run}_{n_components}'
            results[key] = [columnToPredict, run, n_components, RMSE, MAE, NRMSE]
            
results_df = pd.DataFrame.from_dict(results,orient='index', columns = ['columnToPredict', 'run', 'n_components', 'RMSE', 'MAE','NRMSE'])
# results_df.to_csv('test_results.csv')

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


#%%
# Compare the different models
results = {}

# intercept only
for columnToPredict in columns:
    if columnToPredict == 'steps_per_day_y':
        totaal[[columnToPredict]] = totaal[[columnToPredict]].astype(int)

    # Repeat this loop 100 times as there is some random factor in the data
    for run in range(100):
        actual_list = []
        predicted_list = []
        # Randomise the order of the data
        totaal = totaal.sample(frac=1) 
        np.random.shuffle(subj_list)

        for i in range(int(len(subj_list) / 5)):
            # Select subjests 
            subj = subj_list[i*5:i*5+5]

            # Train/test split
            test = totaal.loc[totaal['Subject number'].isin(subj)]
            train = totaal.loc[~totaal['Subject number'].isin(test['Subject number'])]

            # Uses all data of the test set and max 1 measurrement per subject of the train set
            train = train.drop_duplicates(subset = ['Subject number'])

            # Prepare train data set
            y = train[[columnToPredict]].mean()

            # Prepare test data set
            y_test = test[[columnToPredict]]

            actual = y_test
            predicted = np.ones(len(actual)) * y[0]
            for val_1, val_2 in zip(actual.values, predicted):
                actual_list.append(val_1[0])
                predicted_list.append(val_2)

        actual_list = np.array(actual_list)
        predicted_list = np.array(predicted_list)
        RMSE = root_mean_square_error(actual_list, predicted_list)
        NRMSE = norm_root_mean_square_error(actual_list, predicted_list)
        MAE = mean_absolute_error(actual_list, predicted_list)

        print(f'Variable: {columnToPredict} Run: {run}  RMSE: {RMSE}')
        print(f'Variable: {columnToPredict} Run: {run}  MAE: {MAE}')

        key = f'{columnToPredict}_{run}'
        results[key] = [columnToPredict, run, RMSE, MAE, NRMSE]
intercept_only_df = pd.DataFrame.from_dict(results,orient='index', columns = ['columnToPredict', 'run','RMSE', 'MAE','NRMSE'])

outcome_1 = intercept_only_df.groupby(['columnToPredict'])['RMSE'].mean()
outcome_2 =  intercept_only_df.groupby(['columnToPredict'])['RMSE'].std()
outcome_3 = intercept_only_df.groupby(['columnToPredict'])['NRMSE'].mean()
outcome_4 =  intercept_only_df.groupby(['columnToPredict'])['NRMSE'].std()

for j in columns:
    if j == 'steps_per_day_y':
        print(f' {round(outcome_1[j],1)} & {round(outcome_2[j],1)} &&')
    else:
        print(f' {round(outcome_1[j],3)} & {round(outcome_2[j],3)} &&')

#%%
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
p = pls.n_components
Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)

print('\n')
print('PLS Gait speed only model')
print(f'R^2 : {r2}')
print(f'ADJ R^2 : {Adj_r2}')
print(f'RMSE: {root_mean_square_error(predicted, actual.values)}')
print(f'MAE: {mean_absolute_error(predicted, actual)}')
print('\n')

#%%
# r2_list = []
# r2a_list = []
# rmse_list = []
# mae_list = []
# mse_list =[]
# actual_list = []
# predicted_list = []

# subj_list = pca_kmph['Subject number'].unique()
# columnToPredict = 'max_gait_speed_DL_y'
# if columnToPredict == 'steps_per_day_y':
#     totaal[[columnToPredict]] = totaal[[columnToPredict]].astype(int)

# per_k = 5

# for i in range(1,11):
#     actual_list = []
#     predicted_list = []
#     count = 0
#     for _ in range(1):
#         totaal = totaal.sample(frac=1)
#         for subj in subj_list:
#             count += 5
#             # Train/test split
#             test = totaal.loc[totaal['Subject number'].isin(subj)]
#             train = totaal.loc[~totaal['Subject number'].isin(test['Subject number'])]
#             train = train.drop_duplicates(subset = ['Subject number'])
#             train = train.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
#                 'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
#                 'PCA_3', 'PCA_4']
#             test = test.drop(columns=['Subject number', 'T moment', 'aid', 'mean_gait_speed_DL_x',
#                 'max_gait_speed_DL_x', 'steps_per_day_x', 'PCA_0', 'PCA_1', 'PCA_2',
#                 'PCA_3', 'PCA_4'])

#             # Prepare train data set
#             x = train.iloc[:,:70]
#             _, x, tmp_scaler = scaler(x)
#             y = train[[columnToPredict]]

#             # PLS on train dataset
#             pls = PLSRegression(n_components=i, tol = 1e-06)
#             pls.fit(x,y)
#             #pls.score(x,y)

#             # Prepare test data set
#             x_test = test.iloc[:,:70]
#             x_test = tmp_scaler.transform(x_test)
#             y_test = test[[columnToPredict]]

#             # Predict results 
#             if columnToPredict == 'steps_per_day_y':
#                 predicted = pls.predict(x_test).astype(int)
#             else:
#                 predicted = pls.predict(x_test)
#             actual = y_test
#             # r2 = pls.score(x_test,y_test)
#             # n = len(y)
#             # p = pls.n_components
#             # Adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
#             for val_1, val_2 in zip(actual.values, predicted):
#                 actual_list.append(val_1[0])
#                 predicted_list.append(val_2[0])

#     actual_list = np.array(actual_list)
#     predicted_list = np.array(predicted_list)
#     RMSE = root_mean_square_error(actual_list, predicted_list)
#     MAE = mean_absolute_error(actual_list, predicted_list)

#     print(f'Components {i} RMSE: {RMSE}')
#     print(f'Components {i} MAE: {MAE}')

# plt.plot(actual_list, predicted_list)
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

