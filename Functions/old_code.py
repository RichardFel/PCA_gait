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
sns.set_style("whitegrid")
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
      
fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[2].set_box_aspect(1)

for num, columnToPredict in enumerate(columns):
    actual_list_c = []
    predicted_list_c = []
    actual_list_i = []
    predicted_list_i = []
    actual_list_g = []
    predicted_list_g = []

    if columnToPredict == 'steps_per_day':
        data[[columnToPredict]] = data[[columnToPredict]].astype(int)


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
    Adj_score_c = 1-(1-score_c)*(n-1)/(n-p-1)
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

    print(f'Column {columnToPredict}')
    print(f'RMSE gs {RMSE_g}')
    print(f'MAE gs {MAE_g}')
    print(f'NRMSE gs {NRMSE_g}')
    print(f'R^2 gs {score_g}')
    print(f'adj R^2 gs {Adj_score_g}')
    print('\n')

    sns.regplot(x = actual_list_i, y = predicted_list_i, marker='o', ax = ax[num], scatter_kws={'s':8}, fit_reg = False)
    sns.regplot(x = actual_list_g, y = predicted_list_g, marker='^', ax = ax[num], scatter_kws={'s':7}, fit_reg = False)


    # ax[num].plot(actual_list_g, predicted_list_g, marker='^', linestyle='', markersize=4)

    ax[num].set_xlabel('Actual scores')
    ax[num].set_ylabel('Predicted scores')
    if num == 0:
        ax[num].set_title('3A: Number of steps')
        sns.regplot(x = actual_list_c, y = predicted_list_c, marker='s', ax = ax[num], scatter_kws={'s':6}, fit_reg = False)
        # ax[num].plot(actual_list_c, predicted_list_c, marker='s', linestyle='', markersize=3)
        ax[num].set_xlim(0,12000)
        ax[num].set_ylim(0,12000)
        ax[num].legend(labels = ['Intercept only', 'Gait speed', 'Combined'])
        # print stuff
        print(f'RMSE C {RMSE_c}')
        print(f'MAE C {MAE_c}')
        print(f'NRMSE C {NRMSE_c}')
        print(f'R^2 C {score_c}')
        print(f'adj R^2 C {Adj_score_c}')
        print('\n')
    elif num == 1:
        ax[num].set_title('3B: Average gait speed')
        ax[num].set_xlim(0,3)
        ax[num].set_ylim(0,3)
        ax[num].legend(labels = ['Intercept only', 'Gait speed'])
    else:
        ax[num].set_title('3C: Maximum gait speed')
        ax[num].set_xlim(1,4.5)
        ax[num].set_ylim(1,4.5)
        ax[num].legend(labels = ['Intercept only', 'Gait speed'])
    add_identity(ax[num], color='black', ls=':')
plt.savefig(f'Images/prediction_actual.pdf', dpi=300, format='pdf')


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
