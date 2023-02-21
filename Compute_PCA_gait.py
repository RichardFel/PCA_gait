'''
Description:
Code used to compute a principal component analysis on gait features and calculate the association 
between gait features measured with a two-minute walk test and measures of walking behavior.

Input:
1. File with reliable gait features and removed highly correlated / uncorrelated features. 
2. File with information regarding meaeseures of walking behavior in daily life. 

Output:
Results of three linear mixed models for the meaeseures of walking behavior in daily life. 
These include, the number of steps, and the average and maximal gait speed in daily lfie. 
'''


#%%
import numpy as np
import matplotlib.pyplot as plt
from Functions.load_files import *
from Functions.PCA import *
from Functions.LMM import *
from Functions.test_retest import *
from Functions.description import *
from Functions.Association import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import seaborn as sns
verbose = True
plot = True

#%%
def main(verbose, plot):
    ########### load files ###########
    new_selection, selection  = load_file(file = '0.95', verbose = verbose)

    # Describe data
    describe_data(new_selection, selection)

    # Calculates sphericity
    new_selection = sphericity_kmo(new_selection)

    # Rescales data 
    scaledData, scaledData_df, tmp_scaler = scaler(new_selection)

    ########### PCA ###########
    # Creates PCA to determine number of components
    number_of_components = compute_pca(new_selection, scaledData, verbose = True, visualise = False)

    # Creates second PCA with max number of components
    pca, _, _ = compute_pca2(number_of_components, scaledData, new_selection)

    # Used to determine the loadings and features that load strongly
    _, loadings_corr = cor_comp_var(pca, new_selection)
    loadings_corr = pca.components_.T

    # Rotates the PCA
    rotated_loadings_df, rotated_loadings, _ = rotate_pca(loadings_corr, new_selection)

    # Make a bi plot
    if plot:
        bi_plot(rotated_loadings_df)

    ########### Reliability ###########
    # calculate the icc of the test-retest 
    pca_test_hertest_df = load_rotate_test_hertest(new_selection, tmp_scaler, pca)
    icc_mae(pca_test_hertest_df, number_of_components, verbose = True)

    # Add kmph and prepare for LMM
    pca_kmph = add_kmph(scaledData_df, rotated_loadings, new_selection, selection)
    totaal = pca_kmph.merge(selection, how = 'left', on = ['Subject number', 'T moment', 'aid'])
    tmp = totaal.corr()
    tmp = tmp.abs()
    for column in tmp.columns:
        tmp.loc[tmp[column] < 0.3, column] = -1
    tmp.to_excel('Excel files/correlation/correlation_totaal.xlsx')

    ########### Association ###########
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

    # Create a heatmap
    heatmap(pca_daily_life)

    ########### Linear mixed models ###########
    # General settings
    correlation, data = prepare_data_lmm(pca_kmph, totaal, selection)

    # # Liniar mixed effects models to predict kmph in daily life from gait speed two-minute walk test
    columnToPredict = 'steps_per_day'
    randomEffect= 'Subject number'
    PCs = correlation.index[4:-1].values
    data = data.dropna(subset = [columnToPredict])

    # Gait speed only model
    lmm_gs_only(data, columnToPredict, randomEffect)

    # Combined
    base_model = np.array(['ms_2m'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)
    base_model = np.array(['ms_2m', 'PCA_2'])
    PCs = np.array(['PCA_1', 'PCA_6', 'PCA_3', 'PCA_5', 'PCA_8'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)

    # interaction
    base_model = np.array(['ms_2m','PCA_6','ms_2m*PCA_6'])
    fitted_model_1 = lmm_interaction(base_model, data, columnToPredict, randomEffect)
    lmm_plot(fitted_model_1, observed_values = data[columnToPredict], name = columnToPredict)

    ## Max gait speed ##
    columnToPredict = 'max_gait_speed_DL'
    randomEffect= 'Subject number'
    PCs = correlation.index[4:-1].values
    data = data.dropna(subset = [columnToPredict])

    # Gait speed only model
    lmm_gs_only(data, columnToPredict, randomEffect)

    # Combined
    base_model = np.array(['ms_2m'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)
    base_model = np.array(['ms_2m', 'PCA_2'])
    PCs = np.array(['PCA_1', 'PCA_6', 'PCA_3', 'PCA_5', 'PCA_8'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)

    # interaction
    base_model = np.array(['ms_2m','PCA_2'])
    fitted_model_1 = lmm_interaction(base_model, data, columnToPredict, randomEffect)
    lmm_plot(fitted_model_1, observed_values = data[columnToPredict], name = columnToPredict)


    ## Mean gait speed ##
    columnToPredict = 'mean_gait_speed_DL'
    randomEffect= 'Subject number'
    PCs = correlation.index[4:-1].values
    data = data.dropna(subset = [columnToPredict])

    # Gait speed only model
    lmm_gs_only(data, columnToPredict, randomEffect)

    # Combined
    base_model = np.array(['ms_2m'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)

    # interaction
    base_model = np.array(['kmph','PCA_6','kmph*PCA_6'])
    lmm_interaction(base_model, data, columnToPredict, randomEffect)
    base_model = np.array(['kmph', 'PCA_6'])
    PCs = np.array(['PCA_1', 'PCA_2', 'PCA_3', 'PCA_5', 'PCA_8'])
    for column in PCs:
        lmm_combined(base_model, data, column, columnToPredict, randomEffect)
    base_model = np.array(['kmph', 'PCA_6',]) 
    fitted_model_1 = lmm_interaction(base_model, data, columnToPredict, randomEffect)
    lmm_plot(fitted_model_1, data[columnToPredict], columnToPredict)

if __name__ == "__main__":
    main(verbose, plot)

    
#%%