import pandas as pd
import numpy as np

def describe_data(new_selection, selection):
    pt_data = pd.read_excel('Excel files/Overzicht_pt/MS_Karateristieken_2022_10_18.xlsx',index_col = 0)
    pt_data = pt_data.loc[pt_data.index.isin(new_selection.index)]
    
    # amount of rows
    print(f'N participants = {len(pt_data)}')
    print(f'N 2-min walk test = {len(selection)}')
    print(f'N physical activity = {len(selection.dropna(subset = ["mean_gait_speed_DL"]))}')
    print('\n')
    
    
    # Sex
    print(f'aantal man: {len(pt_data.loc[pt_data["Geslacht [0=vrouw / 1 =man]"] == 0])}, '
          f'aantal vrouw: {len(pt_data.loc[pt_data["Geslacht [0=vrouw / 1 =man]"] == 1])}')
    gait_speed = new_selection['KMPH R'].describe()
    gait_speed = gait_speed / 3.6
    print(f'gait speed mean: {round(gait_speed["mean"],2)} std {round(gait_speed["std"],2)} ')
    gait_speed = selection['mean_gait_speed_DL'].describe()
    gait_speed = gait_speed / 3.6
    print(f'mean gait speed daily life mean: {round(gait_speed["mean"],2)} std {round(gait_speed["std"],2)} ')
    
    age = pt_data['Leeftijd [jaren]'].describe()
    print(f'Leeftijd gem. {round(age["mean"],1)} en SD {round(age["std"],1)}.')
    stride_time = new_selection['Stride time mean R'].describe()
    print(f'stride_time mean: {round(stride_time["mean"],1)} std {round(stride_time["std"],1)} ')
    gait_speed = selection['max_gait_speed_DL'].describe()
    gait_speed = gait_speed / 3.6
    print(f'max gait speed daily life mean: {round(gait_speed["mean"],2)} std {round(gait_speed["std"],2)} ')
    
    beroerte_type = pt_data['soort CVA [ischemisch = 0/hemorragisch = 1/subarachnoïdale bloeding = 2]'].value_counts()
    print(beroerte_type)
    stride_lenth = new_selection['Stride dist mean R'].describe()
    print(f'stride_lenth mean: {round(stride_lenth["mean"],1)} std {round(stride_lenth["std"],1)} ')
    gait_speed = selection['mode_gait_speed_DL'].describe()
    gait_speed = gait_speed / 3.6
    print(f'modus gait speed daily life mean: {round(gait_speed["mean"],2)} std {round(gait_speed["std"],2)} ')
    
    beroerte_zijde = pt_data['zijde CVA'].value_counts()
    print(beroerte_zijde)
    cadence = new_selection['Cadence L'].describe()
    print(f'cadence mean: {round(cadence["mean"],1)} std {round(cadence["std"],1)} ')
    strides_per_day = selection['steps_per_day'].describe()
    print(f'steps_per_day  mean: {round(strides_per_day["mean"],0)} std {round(strides_per_day["std"],0)} ')
    
    BI = pt_data['BI'].describe()
    print(f'BI mean: {round(BI["mean"],1)} std {round(BI["std"],1)} ')
    time_per_day = selection['minutes_per_day'].describe()
    print(f'time_per_day  mean: {round(time_per_day["mean"],0)} std {round(time_per_day["std"],0)} ')
    
    BSS = pd.to_numeric(pt_data['BBS [0 - 56] ']).describe(include=[np.number])
    print(f'BSS mean: {round(BSS["mean"],1)} std {round(BSS["std"],1)} ')
