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

from scipy import stats
#%%

# load files
new_selection,selection  = load_file(file = '0.95', verbose = verbose)

# Describe data
describe_data(new_selection, selection)

# Calculates sphericity
new_selection = sphericity_kmo(new_selection)

# Rescales data 
scaledData, scaledData_df, scaler = scaler(new_selection)

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
pca_test_hertest_df = load_rotate_test_hertest(new_selection, scaler, pca)
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


#%%
# Gait speed only model
subj_list = pca_kmph['Subject number'].unique()
two_measurement = subj_list[pca_kmph.groupby(['Subject number'])['steps_per_day'].count() > 1]

# Columns: mean_gait_speed_DL, max_gait_speed_DL, mode_gait_speed_DL, steps_per_day, minutes_per_day
columnToPredict = 'max_gait_speed_DL'
randomEffect= 'Subject number'

# Liniar mixed effects models to predict kmph in daily life from gait speed 2min
pca_daily_life = pca_daily_life.dropna(subset = [columnToPredict])
pca_daily_life.rename(columns = {'KMPH R': 'kmph'}, inplace = True)
pca_daily_life = pca_daily_life.loc[pca_daily_life['Subject number'].isin(two_measurement)]

icc = pd.read_excel('Excel files/ICC/ICC_MAE.xlsx', index_col=0)
high_icc = icc.columns[icc.loc['ICC'] > 0.75]
# Exclude C0 & C4
high_icc = high_icc[1:3]

# Gait speed only model
columns = np.array(['kmph'])
model_1 = create_lmm(pca_daily_life, columnToPredict, randomEffect, columns)
fitted_model_1 = model_1.fit(reml=False)
print(fitted_model_1.summary())

# Combines model
columns = np.append(['kmph'], high_icc.values )
model_2 = create_lmm(pca_daily_life, columnToPredict, randomEffect, columns)
fitted_model_2 = model_2.fit(reml=False)
print(fitted_model_2.summary())

log_model_1 = fitted_model_1.llf
log_model_2 = fitted_model_2.llf

LR_statistic = -2*(-log_model_2 - -log_model_1)
print(LR_statistic)

p_val = scipy.stats.chi2.sf(LR_statistic, 3)
print(p_val)

def bic(self):
    """Bayesian information criterion"""
    if self.reml:
        return np.nan
    if self.freepat is not None:
        df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
    else:
        df = self.params.size + 1
    return -2 * self.llf + np.log(self.nobs) * df

def aic(llf, nobs, df_modelwc):
    """
    Akaike information criterion

    Parameters
    ----------
    llf : {float, array_like}
        value of the loglikelihood
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    aic : float
        information criterion
    """
    return -2.0 * llf + 2.0 * df_modelwc
bic(fitted_model_1)

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

