import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap(pca_daily_life):
    '''
    Description:
    creates a heatmap of the correlations 
    '''

    mask = np.triu(np.ones_like(pca_daily_life.corr(), dtype=np.bool))
    pca_daily_life.corr().round(2).to_excel('Excel files/correlation/correlation_association.xlsx')
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.heatmap(pca_daily_life.corr().iloc[1:,:-1], vmin=-1, vmax=1, mask=mask[1:,:-1], annot=True, cmap='seismic', ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment="right") 
    ax.tick_params(axis='both', which='major', labelsize=23)
    plt.tight_layout()
    plt.savefig('Images/heatmap.pdf', dpi=300, format='pdf')