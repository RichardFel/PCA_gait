# # Independent sample t-test
# rotated_data = np.dot(scaledData_df.iloc[:,:], rotated_loadings) 
# pca_fac = pd.DataFrame(rotated_data, index = [new_selection.index, selection['T moment'],  selection['aid'], selection['FAC_score']])
# pca_fac.reset_index(inplace = True)
# pca_fac.dropna(subset = ['FAC_score'], inplace = True)

# pca_fac = pca_fac.loc[(pca_fac.FAC_score == 3) | (pca_fac.FAC_score == 5)]
# def FAC_determination(x):
#     return 0 if x == 3 else 1

# pca_fac['FAC_high_low'] = pca_fac['FAC_score'].apply(FAC_determination)

# pca_fac = pca_fac.iloc[:,4:]
# for column in pca_fac.iloc[:,:-1]:  
#     pca_fac.rename(columns = {column: f"PCA_{column}"}, inplace = True)  
# low_fac = pca_fac.loc[pca_fac['FAC_high_low'] == 0]
# high_fac = pca_fac.loc[pca_fac['FAC_high_low'] == 1]

# for column in pca_fac.iloc[:,:-1].columns:
#     mean_low = round(np.mean(low_fac[column]),1)
#     sd_low = round(np.std(low_fac[column]),1)
#     min_low = round(min(low_fac[column]),1)
#     max_low = round(max(low_fac[column]),1)
#     mean_high = round(np.mean(high_fac[column]),1)
#     sd_high = round(np.std(high_fac[column]),1)
#     min_high = round(min(high_fac[column]),1)
#     max_high = round(max(high_fac[column]),1)
#     _,p_value = scipy.stats.ttest_ind(low_fac[column],high_fac[column])

#     if p_value < 0.01:
#         p_value = '<0.01'
#     else:   
#         p_value = round(p_value,3)
#     print(f'{column} & {mean_low} ({sd_low})[{min_low},{max_low}] & {mean_high} ({sd_high})[{min_high},{max_high}] & {p_value}')

# #%%
# Load and rotate test-retest data