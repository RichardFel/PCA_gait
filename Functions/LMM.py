import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.api as sm
sns.set_style("whitegrid", {'axes.grid' : False})

def prepare_data_lmm(pca_kmph, totaal, selection, plot = False):
    '''
    Description:
    Prepare data to use for the linear mixed models, including converting KM/H into M/s
    '''

    data = pca_kmph
    data =  data.merge(totaal[['Subject number', 'T moment', 'aid', 'KMPH R']], how = 'left', on = ['Subject number', 'T moment', 'aid'])
    data['aid'].replace(['Nee', 'Ja'],[0, 1], inplace=True)
    data['Rehab_center'] = data['Subject number'].str[-1:]
    data.drop_duplicates(subset = ['Subject number', 'T moment', 'aid'], inplace = True)
    # data = data.loc[~data['Subject number'].isin(data['Subject number'].value_counts()[data['Subject number'].value_counts() == 1].index)]
    
    # Drop high correlatted variables
    data.rename(columns = {'KMPH R': 'kmph'}, inplace = True)
    data['ms_2m'] = round(data['kmph'] / 3.6 * 10,2)  
    data['ms_2m'] -= data['ms_2m'].mean() 
    data['mean_gait_speed_DL'] = round(data['mean_gait_speed_DL'] / 3.6 * 10 ,2)
    data['max_gait_speed_DL'] = round(data['max_gait_speed_DL'] / 3.6 * 10 ,2)
    correlation = data.corr().abs()
    tmp_corr = correlation['ms_2m'][4:-1].loc[correlation['ms_2m'][4:-1] > 0.8]
    correlation = correlation.loc[~correlation.index.isin(tmp_corr.index.values)]

    if plot:
        var = 'KMPH R'
        spaghetti = selection.loc[:, ['Subject number','T moment',var]]
        spaghetti = spaghetti.loc[~spaghetti['Subject number'].isin(spaghetti['Subject number'].value_counts()[spaghetti['Subject number'].value_counts() == 1].index)]
        spaghetti.drop_duplicates(subset = ['Subject number', 'T moment'], inplace = True)
        spaghetti_wide = pd.pivot(spaghetti, index='Subject number', columns='T moment', values=var)
        for subj in spaghetti_wide.index:
            x = np.where(~np.isnan(spaghetti_wide.loc[subj].values))[0]
            values = spaghetti_wide.loc[subj].values[~np.isnan(spaghetti_wide.loc[subj].values)]
            plt.plot(x, values)

    return correlation, data


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
    '''
    Description:
    Creates a linear mixed model with slopes
    '''

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

def create_lmm(data, columnToPredict, randomEffect, columns, slope = None):
    '''
    Description:
    Creates a linear mixed model
    '''

    fixedEffects = ''
    if len(columns) > 1:
        for column in columns:
            fixedEffects += (str(column)) if column == columns[-1] else f'{str(column)}+'
    else:
        fixedEffects = columns[0]
    Formula = f'{str(columnToPredict)}~{fixedEffects}'
    # fitted_model = model.fit()
    # print(fitted_model.summary())
    if slope:
        return smf.mixedlm(Formula, data, groups=data[randomEffect], re_formula=f"~{slope}")
    else:
        return smf.mixedlm(Formula, data, groups=data[randomEffect])

def lmm_gs_only(data, columnToPredict, randomEffect):
    '''
    Description:
    Creates a linear mixed model using only gait speed
    '''
    columns = np.array(['ms_2m'])
    model_1 = create_lmm(data, columnToPredict, randomEffect, columns)
    fitted_model_1 = model_1.fit(reml=False)
    print(fitted_model_1.summary())
    print(f'LL:{fitted_model_1.llf}')
    print(f'AIc: {aic(fitted_model_1)}')
    print(f'BIC: {bic(fitted_model_1)}')
    print(f'RMSE: {root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}') 
    print(f'nRMSE: {n_root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}')

def lmm_interaction(base_model, data, columnToPredict, randomEffect):
    '''
    Description:
    Creates a linear mixed model using gait speed and an interaction term
    '''
    model_1 = create_lmm(data, columnToPredict, randomEffect, base_model)
    fitted_model_1 = model_1.fit(reml=False)
    print(fitted_model_1.summary())
    print(f'LL:{fitted_model_1.llf}')
    print(f'AIc: {aic(fitted_model_1)}')
    print(f'BIC: {bic(fitted_model_1)}')
    print(f'RMSE: {root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}') 
    print(f'nRMSE: {n_root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}')
    return fitted_model_1

def increase_font(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

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

def lmm_plot(fitted_model_1, observed_values, name):
    '''
    Description:
    Visualise the results of the models
    '''
    if name == 'mean_gait_speed_DL':
        xlim1 = [-0.15,0.15]
        xlim2 = [-0.2,0.2]
        xlim3 = [0.1,0.7]
        resid = fitted_model_1.resid / 10
        fit = fitted_model_1.fittedvalues / 10
        obs = observed_values / 10
        title = '3A: Average gait speed'
        appendix_num = 1
    elif name == 'max_gait_speed_DL':
        xlim1 = [-0.4,0.4]
        xlim2 = [-0.3,0.3]
        xlim3 = [0.2,1.2] 
        resid = fitted_model_1.resid / 10
        fit = fitted_model_1.fittedvalues / 10
        obs = observed_values / 10
        title = '3B: Maximum gait speed'
        appendix_num = 2
    else:
        xlim1 = [-2500,2500]
        xlim2 = [-2500,2500]
        xlim3 = [-200,8000]
        resid = fitted_model_1.resid 
        fit = fitted_model_1.fittedvalues 
        obs = observed_values 
        title = '3C: Number of steps per day'
        appendix_num = 3

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.histplot(resid, ax = ax)
    ax.set_xlim(xlim1)
    ax.set_title(f'{appendix_num}A: Distribution residuals')
    increase_font(ax)
    plt.savefig(f'Images/resid_{name}.pdf', dpi=300, format='pdf')
    
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    sm.qqplot(resid, line ='45', ax = ax2, fit = True)
    ax2.set_title(f'{appendix_num}B: Q-Q plot')
    increase_font(ax2)

    plt.savefig(f'Images/qqplot_{name}.pdf', dpi=300, format='pdf')
    
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    ax3.plot(fit,resid, '.',  markersize=20)
    ax3.set_ylim(xlim2)
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'{appendix_num}C: Residual vs fitted')
    increase_font(ax3)

    plt.savefig(f'Images/res_fit_{name}.pdf', dpi=300, format='pdf')

    fig4, ax4 = plt.subplots(figsize=(9, 9))
    ax4.plot(fit,obs, '.',  markersize=20)
    ax4.set_xlabel('Estimated values')
    ax4.set_ylabel('Observed values')

    ax4.set_ylim(xlim3)
    ax4.set_xlim(xlim3)
    ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax4.yaxis.set_major_locator(plt.MaxNLocator(4))
    add_identity(ax4)
    # ax4.set_title(title)
    increase_font(ax4)
    plt.tight_layout()
    
    plt.savefig(f'Images/fit_obs_{name}.pdf', dpi=300, format='pdf')




def lmm_combined(base_model, data, column, columnToPredict, randomEffect):
    incl_components = np.append(base_model, [column])
    model_1 = create_lmm(data, columnToPredict, randomEffect, incl_components)
    fitted_model_1 = model_1.fit(reml=False)
    print(fitted_model_1.summary())
    
    print(f'LL:{fitted_model_1.llf}')
    print(f'AIc: {aic(fitted_model_1)}')
    print(f'BIC: {bic(fitted_model_1)}')
    print(f'RMSE: {root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}') 
    print(f'nRMSE: {n_root_mean_square_error(fitted_model_1.fittedvalues, data[columnToPredict])}')

def bic(self):
    """Bayesian information criterion"""
    if self.reml:
        return np.nan
    if self.freepat is not None:
        df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
    else:
        df = self.params.size + 1
    return -2 * self.llf + np.log(self.nobs) * df

def aic(self):
    """Akaike information criterion"""
    if self.reml:
        return np.nan
    if self.freepat is not None:
        df = self.freepat.get_packed(use_sqrt=False, has_fe=True).sum() + 1
    else:
        df = self.params.size + 1
    return -2.0 * self.llf + 2.0 * df

def n_root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    nrmse = rmse / np.std(targets)
    return round(nrmse,3)
    
def root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    return round(rmse,3)
    