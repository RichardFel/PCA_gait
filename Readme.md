# Principal component analysis and linear mixed models 

## Description

This algorithm was created for the paper: 'beyond gait speed' in which we evaluated the association between gait speed obtained from a two-minute walk test and measures of walking behavior in daily life. 
The following steps are conducted in this algorithm:
1. Data is 'cleaned' by only including reliable gait features (ICC > 0.75) which are correlated with at least 0.3. If multiple features had a correlation > 0.95, only one of t hese was included in further analysis.
2. A principal component analysis (PCA) is applied to the data 
3. The corresponding loadings are tested by transforming test-retest data (unused in the computation of the PCA) into principal components and calculating the reliability. 
4. Last, per measures of walking ability a baseline gait-speed-only linear mixed model (LMM) is created. The PC are added via a forward selection procedure. 

## Getting Started

The data required to run the code will be made availible after publication of the study.

1. Proces the raw data using the clean_data.py file.
2. Calculate the PCA and the LMM by running the compute_pca_gait.py file.

### Libraries
* Numpy
* Matplotlib
* Seaborn
* Pandas
* Statsmodels
* Sklearn
* factor_analyzer
* pingouin


## Help
For help or information contact: richard.felius@hu.nl.

## Authors
Richard Felius; Utrecht University of applied sciences; email: richard.felius@hu.nl.


## Version History
* 0.1
    * Initial Release


## Acknowledgments
