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

import statsmodels.api as sm
from statsmodels.formula.api import logit
import statsmodels.formula.api as smf



def add_kmph(scaledData_df, rotated_loadings, new_selection, selection):
    rotated_data = np.dot(scaledData_df.iloc[:,:], rotated_loadings) 
    pca_fac = pd.DataFrame(rotated_data, index = [new_selection.index, selection['T moment'],  selection['aid'], selection['mean_gait_speed_DL'],
                                                  selection['max_gait_speed_DL'],  selection['steps_per_day']
                                                  ])
    pca_fac.reset_index(inplace = True)
    pca_fac.dropna(subset = ['mean_gait_speed_DL'], inplace = True)
    # sns.histplot(pca_fac['mean_gait_speed_DL'])
    pca_fac = pca_fac.reset_index(drop=True)
    
    for column in pca_fac.iloc[:,6:]:  
        pca_fac.rename(columns = {column: f"PCA_{column}"}, inplace = True) 
    return pca_fac

def create_lmm_random_slope(data, columnToPredict, randomEffect, fixedEffects, columns):
    if len(columns) > 1:
        for column in columns:
            fixedEffects += (str(column)) if column == columns[-1] else f'{str(column)}+'
    else:
        fixedEffects = columns[0]
    fixedEffects += '+T_moment'
    Formula = f'{str(columnToPredict)}~{fixedEffects}'
    data.rename(columns={'T moment': 'T_moment'}, inplace = True)
    data["T_moment"] = data["T_moment"].str.replace("T","")
    # fitted_model = model.fit()
    # print(fitted_model.summary())
    return smf.mixedlm(Formula, data, groups=data[randomEffect])

def create_lmm(data, columnToPredict, randomEffect, columns):
    fixedEffects = ''
    if len(columns) > 1:
        for column in columns:
            fixedEffects += (str(column)) if column == columns[-1] else f'{str(column)}+'
    else:
        fixedEffects = columns[0]
    Formula = f'{str(columnToPredict)}~{fixedEffects}'
    # fitted_model = model.fit()
    # print(fitted_model.summary())
    return smf.mixedlm(Formula, data, groups=data[randomEffect])