#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

gait_daily_life = pd.read_excel('Excel files/Raw_files/gait_features_new_60s_lag5.xlsx', index_col = 0, header = 0)



# %%

fig, ax = plt.subplots()
sns.histplot(data=gait_daily_life, x="KMPH", ax = ax, binwidth = 0.1, binrange = [0.2,5])
ax.set_title('Gait speed in daily life')


S9903H = gait_daily_life.loc[gait_daily_life['Subject'] == 'S9903H']
fig, ax = plt.subplots()
sns.histplot(data=gait_daily_life, x="KMPH", ax = ax, binwidth = 0.1, binrange = [0.2,5])
ax.set_title('Gait speed in daily life')