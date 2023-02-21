import pandas as pd
import numpy as np
import pingouin as pg

def load_rotate_test_hertest(new_selection, scaler, pca):
    '''
    Description:
    1. Compute the principal components based on test-retest data
    2. Calculate ICC scores
    '''

    test_hertest = pd.read_excel('Excel files/Raw_files/gait_features_test_retest.xlsx', index_col=0)
    nFilesPP = 2
    countValues = test_hertest['Subject number'].value_counts()
    inComplete = countValues.loc[countValues != nFilesPP].index
    test_hertest.drop(test_hertest.loc[test_hertest['Subject number'].isin(inComplete)].index, inplace = True)
    test_hertest_selection = test_hertest.loc[:,test_hertest.columns.isin(new_selection.columns)]
    scaled_data_test_hertest = scaler.transform(test_hertest_selection)
    pca_data_test_hertest = pca.transform(scaled_data_test_hertest)
    pce_test_hertest_df = pd.DataFrame(pca_data_test_hertest, index = test_hertest[['Subject number','Test Type']],
        columns = [f'PCA_{x}' for x in range(pca_data_test_hertest.shape[1])])
    pce_test_hertest_df['Subject number'], pce_test_hertest_df['Test Type'] = zip(*pce_test_hertest_df.index)
    pce_test_hertest_df.reset_index(drop = True, inplace = True)
    return pce_test_hertest_df

# Validity and reliability

## calculate accuracy
def root_mean_square_error(predictions, targets):
    rmse = np.sqrt(np.mean(((predictions - targets) ** 2)))
    return round(rmse,3)
    
def root_mea_square_error_min_max(predictions, targets):
    minimum = round(np.sqrt(((predictions - targets) ** 2)).min(),3)
    maximum = round(np.sqrt(((predictions - targets) ** 2)   ).max()   ,3) 
    return minimum, maximum
    
def mean_absolute_error(predictions, targets):
    mae = np.average(np.abs(predictions - targets))   
    return round(mae,3)

def absolute_error_std(predictions, targets):
    mae = np.std(abs(predictions - targets) )
    return round(mae,3)

def absolute_error_min_max(predictions, targets):
    minimum = round(abs(predictions - targets).min(),3)
    maximum = round(abs(predictions - targets).max(),3)
    return minimum, maximum

def rel_mean_absolute_error(predictions, targets):
    sd = (np.std(predictions) + np.std(targets))/2
    rmae = np.sqrt(np.mean(((predictions - targets) ** 2))) /sd
    return round(rmae,3)

def icc_mae(pca_test_hertest_df, number_of_components, verbose = None):
    '''
    Description:
    Calculate ICC, MDS, etc. for the test-retest Principal components
    '''

    results = pd.DataFrame(index = ['ICC', 'MDC', 'MAE', 'MAE_std', 'MAE_min',
    'MEA_max','MAPE', 'RMSE', 'RMSE_STD', 'RMSE_min', 'RMSE_max'],
                    columns = [f'PCA_{x}' for x in range(number_of_components)]).astype(float)

    for variable in pca_test_hertest_df.columns[:number_of_components]:
        variableDF = pca_test_hertest_df[['Subject number', 'Test Type',  variable]]
        icc = pg.intraclass_corr(data=variableDF, targets='Subject number', raters = 'Test Type',
                                ratings=variable).round(10)
        ICC = icc['ICC'].loc[1]
        CI = icc['CI95%'].loc[1]
        SEM = (np.std(variableDF.loc[:,variable]) * np.sqrt(1 - ICC))
        MDC = (1.96 * SEM * np.sqrt(2))
        SEM = SEM.round(3)
        ICC_CI = f'{str(ICC.round(3))} [{str(CI[0])},{str(CI[1])}]'
        MDC_SEM = f'{str(MDC.round(3))} ({str(SEM)})'
        results.loc['ICC',variable] = round(ICC,3)
        results.loc['ICC_CI',variable] = ICC_CI
        results.loc['MDC',variable] = round(MDC,3)
        results.loc['MDC_SEM',variable] = MDC_SEM


        test = variableDF.loc[variableDF['Test Type'] == 'Test']
        hertest = variableDF.loc[variableDF['Test Type'] == 'Hertest']
        results.loc['RMSE',variable] = root_mean_square_error(test[variable].values, hertest[variable].values)
        results.loc['MAE',variable] = mean_absolute_error(test[variable].values, hertest[variable].values)
        results.loc['MAE_min',variable], results.loc['MEA_max',variable] = absolute_error_min_max(test[variable].values, hertest[variable].values)
        results.loc['rMAE',variable] = rel_mean_absolute_error(test[variable].values, hertest[variable].values)
        mae_min_max = f"{results.loc['MAE',variable]}"
        if verbose:
            print(f"{ICC_CI} & {MDC_SEM} & {mae_min_max} & {results.loc['rMAE',variable]} & {results.loc['RMSE',variable]} \ \ ")
    results.to_excel('Excel files/ICC/ICC_MAE.xlsx')