import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def load_file(file = '0.95', verbose = None):
    '''
    Description:
    1. Load the raw gait files
    2. Load daily life gait characteristics & process to get a score per participant
    '''
    # Load the file
    selection = pd.read_excel(f'Excel files/Cleaned_files/clean_gait_{file}.xlsx', index_col = 0)
    selection = selection.reset_index()
    # selection.drop_duplicates(subset=['Subject number', 'T moment'], keep = 'last', inplace = True)
    new_selection = selection.drop(columns = ['T moment', 'aid'])
    new_selection.set_index('Subject number', inplace = True)
    
    # load the gait speed file
    gait_daily_life = pd.read_excel('Excel files/Raw_files/gait_features_new_30s_lag5_22_10_26.xlsx', index_col = 0, header = 0)
    
    # Exlude everything below 0.2 speed
    gait_daily_life = gait_daily_life.loc[gait_daily_life['KMPH'] >= 0.2]
    
    # At least 50 gait events
    tmp = gait_daily_life.groupby(['Subject', 'T_moment'])['n strides'].agg(['count'])
    tmp = tmp.loc[tmp['count'] > 25]
    tmp.reset_index(inplace = True)
    gait_daily_life = gait_daily_life[gait_daily_life.set_index(['Subject','T_moment']).index.isin(tmp.set_index(['Subject','T_moment']).index)]

    # Figure of data
    gait_daily_life['Gait speed [m/s]'] = gait_daily_life['KMPH'] / 3.6
    fig_f, ax_f = plt.subplots(1,2)
    sns.histplot(data=gait_daily_life, x='Gait speed [m/s]', binwidth= 1/37, ax = ax_f[0])
    ax_f[0].set_title('1A')
    ax_f[0].grid(False)
    example = gait_daily_life.loc[gait_daily_life['Subject'] == 'S5398H']
    example.rename(columns = {'T_moment': 'T moment'}, inplace = True)
    sns.histplot(data=example, x='Gait speed [m/s]', binwidth= 1/37, ax = ax_f[1],
    hue = 'T moment', multiple="stack")
    ax_f[1].set_title('1B')
    ax_f[1].grid(False)
    ax_f[1].set_ylabel('Count')

    plt.tight_layout()
    fig_f.savefig('Images/distribution.pdf', dpi=300, format='pdf')
    # ax_f.set_title('Distribution gait speed in daily life')


    # Calculate features from this file that we want
    mean_gait_speed = gait_daily_life.groupby(['Subject', 'T_moment'])['KMPH'].mean()
    
    # Max gait speed, at least 3 occurences at this speed
    n_gait_speed = gait_daily_life.groupby(['Subject', 'T_moment'])['KMPH'].value_counts()
    test = n_gait_speed.loc[n_gait_speed > 2 ]
    test.name = 'frequency'
    test2 = test.reset_index()
    max_gait_speed = test2.groupby(['Subject', 'T_moment'])['KMPH'].max()
    
    # # Modus of the gait speed
    mode_gait_speed = gait_daily_life.groupby(['Subject', 'T_moment'])['KMPH'].agg(pd.Series.mode)
    for row, value in enumerate(mode_gait_speed):
        if not isinstance(value, float):
            mode_gait_speed[row] = value[-1]
    mode_gait_speed = mode_gait_speed.astype('float64')
    
    # Time wearing the sensor
    gait_daily_life[['First','Last']] = gait_daily_life.Episode_ID.str.split("_",expand=True,)
    gait_daily_life['First'] = pd.to_numeric(gait_daily_life['First'])
    gait_daily_life['Last'] = pd.to_numeric(gait_daily_life['Last'])
    first_last = gait_daily_life.groupby(['Subject', 'T_moment','Measurement_num'])['First'].agg(['min','max'])
    first_last.reset_index(inplace = True)
    
    # single pa measurements
    single = first_last.drop_duplicates(subset = ['Subject', 'T_moment'], keep = False)
    single.loc[:,'difference'] = single.loc[:,'max'] - single.loc[:,'min']

    # multiple pa measurements    
    multiple = first_last.loc[~first_last.index.isin(single.index)]
    multiple_2 = multiple.drop_duplicates(subset = ['Subject', 'T_moment'], keep = 'first')
    summed_max = multiple.groupby(['Subject', 'T_moment'])['max'].agg(['sum'])
    summed_max.reset_index(inplace = True)
    multiple_2 = multiple_2.merge(summed_max, how = 'left', on = ['Subject', 'T_moment'])
    multiple_2['sum'] += 3600 
    multiple_2.loc[:,'difference'] = multiple_2.loc[:,'sum'] - multiple_2.loc[:,'min']
    wearing_time = pd.concat((single, multiple_2), ignore_index = True)
    wearing_time = wearing_time.loc[wearing_time['difference'] > 3800]
    wearing_time.loc[wearing_time['difference'] < 8640, 'difference'] += 3800
    wearing_time['difference'] = wearing_time['difference'] / 8640
    
    # Calculate steps per day
    number_of_steps = gait_daily_life.groupby(['Subject', 'T_moment'])['n strides'].agg(['sum']) 
    number_of_steps.reset_index(inplace = True)
    number_of_steps = number_of_steps.merge(wearing_time[['Subject', 'T_moment','difference']], on = ['Subject', 'T_moment'], how = 'left')
    number_of_steps['steps_per_day'] = round(number_of_steps['sum'] / number_of_steps['difference'],0)
    number_of_steps.rename(columns = {'Subject' : 'Subject number',
                                      'T_moment': 'T moment'}, inplace = True)
    
    # Minutes per day
    step_time = gait_daily_life.groupby(['Subject', 'T_moment'])['n strides'].agg(['count']) 
    step_time.reset_index(inplace = True)
    step_time['count'] = step_time['count'] / 2 # per minute

    step_time = step_time.merge(wearing_time[['Subject', 'T_moment','difference']], on = ['Subject', 'T_moment'], how = 'left')
    step_time['minutes_per_day'] = round(step_time['count'] / step_time['difference'],0)
    step_time.rename(columns = {'Subject' : 'Subject number',
                                      'T_moment': 'T moment'}, inplace = True)

    
    def merge_data(df, column, name, selection):
        df = df.to_frame()
        df.reset_index(inplace = True)
        df.rename(columns = {column: name,
                                          'Subject' : 'Subject number',
                                          'T_moment': 'T moment'}, inplace = True)
        selection = selection.merge(df[['Subject number','T moment', name]], how = 'left', 
                                    on = ['Subject number','T moment'])
        return selection
    
    selection = merge_data(mean_gait_speed, 'KMPH','mean_gait_speed_DL', selection)
    selection = merge_data(max_gait_speed, 'KMPH','max_gait_speed_DL', selection)
    selection = merge_data(mode_gait_speed, 'KMPH', 'mode_gait_speed_DL', selection)
    selection = selection.merge(number_of_steps[['Subject number','T moment','steps_per_day']], 
                                on = ['Subject number','T moment'],
                                how = 'left')
    selection = selection.merge(step_time[['Subject number','T moment','minutes_per_day']], 
                                on = ['Subject number','T moment'],
                                how = 'left')
    return new_selection, selection
