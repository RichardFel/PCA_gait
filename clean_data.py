#%%
import pandas as pd


def main(Activity, correlation_value):
    raw_outcomes = pd.read_excel('Excel files/Cleaned_files/clean_gait.xlsx')
    icc_values = pd.read_excel('Excel files/ICC/gait_features_ICC.xlsx', index_col = 0, 
                               usecols = [0,1])
    high_icc_values = icc_values.loc[icc_values['ICC'] > 0.75]
    raw_outcomes.set_index(['Subject number', 'T moment', 'aid'], inplace = True)
    selection = raw_outcomes[high_icc_values.index]
    correlation = check_correlation(selection, correlation_value, Activity)
    print(correlation.columns)
    print(len(correlation))
    clean_selection = selection[correlation.columns] 
    clean_selection = clean_selection.reset_index()
    cleaned_selection = clean_selection.sort_values(by = ['Subject number', 'T moment'])
    saveAs = f'Excel files/Cleaned_files/clean_gait_{correlation_value}.xlsx'
    cleaned_selection.to_excel(saveAs, index=False )
       
def check_correlation(selection, corr_value, Activity):
    correlation = selection.corr()            
    correlation.to_excel(f'Excel files/Cleaned_files/correlation_{Activity}.xlsx', index=True )
    for column, _ in correlation.iterrows():
        if (correlation.loc[column].abs() < 0.3).all():
            print(f'{column} is not correlated')
            correlation.drop(columns = [column], inplace = True)
            correlation.drop(index = [column], inplace = True)
            
    for variable1, variable2 in correlation.iterrows():
        for variable2, value in variable2.iteritems():
            if variable1 == variable2:
                continue
            if abs(value) > corr_value:
                # print(f'{variable1} and {variable2} are highly correlated')
                try:
                    if correlation[variable1].sum() > correlation[variable2].sum():
                       correlation.drop(columns = [variable1], inplace = True)
                       correlation.drop(index = [variable1], inplace = True)
                    else:
                        correlation.drop(columns = [variable2], inplace = True)
                        correlation.drop(index = [variable2], inplace = True)
                except KeyError: 
                    continue
    return correlation
    
#%%
if __name__ == "__main__":
    # unique_id = pd.read_excel('Excel files/Overzicht_pt/Unique IDs.xlsx', index_col = 0)
    raw_outcomes = pd.read_excel('Excel files/Raw_files/looptest_2022_10_24.xlsx', index_col = 0)
    
    # raw_outcomes = raw_outcomes.loc[raw_outcomes['Subject number'].isin(unique_id.index)]
    duplicates = raw_outcomes.loc[raw_outcomes.duplicated(['Stride time mean L', 'Stride time std L'], keep = False)]
    duplicates.to_excel('Excel files/Duplicates/Duplicates.xlsx')
    raw_outcomes_nodups = raw_outcomes.drop_duplicates(['Stride time mean L', 'Stride time std L'], )
    raw_outcomes_nodups.to_excel('Excel files/Cleaned_files/clean_gait.xlsx', index= False )
    
    main(Activity = 'Gait', correlation_value = 0.95)



# %%
