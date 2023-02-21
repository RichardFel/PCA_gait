import pandas as pd
import numpy as np
import scipy

from sklearn.decomposition import PCA
from sklearn import preprocessing

from factor_analyzer import Rotator
from factor_analyzer.factor_analyzer import calculate_kmo

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import cross_val_score


def sphericity_kmo(new_selection, verbose = None):
    '''
    Description:
    Calculate Kaiser-Meyer-Olkin
    '''
    ## Bartlett sphericity
    #from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    #chi_square_value,p_value=calculate_bartlett_sphericity(new_selection)
    #chi_square_value, p_value
    
    ## Kaiser-Meyer-Olkin (KMO)
    kmo_all,kmo_model=calculate_kmo(new_selection)
    kmo_outcome = pd.DataFrame(kmo_all, new_selection.columns, columns = ['KMO'])
    kmo_new_selection = kmo_outcome.loc[kmo_outcome['KMO'] > 0.5]
    print(f'Variables excluded because KMO {len(new_selection.columns) - len(kmo_new_selection)}')
    print(f'Average kaiser-Meyer-Olkin: {kmo_new_selection.mean()}')
    new_selection = new_selection.loc[:,new_selection.columns.isin(kmo_new_selection.index)]
    return new_selection


# Pre processing
def scaler(new_selection):
    '''
    Description:
    Convert data to z-scores
    '''

    scaler = preprocessing.StandardScaler() 
    scaler.fit(new_selection)
    scaledData = scaler.transform(new_selection) # Use s.transform on other data to scale other data
    
    scaledData_df = pd.DataFrame(scaledData, columns = new_selection.columns)
    # scaledData_df.to_excel('Scaled_data.xlsx')
    return scaledData, scaledData_df, scaler


def scree_plot(pca):
    ax = figure().gca()
    ax.plot(pca.explained_variance_)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Scree Plot of PCA: Component Eigenvalues')
    show()
    
def var_explained(number_of_components, pca):
    import numpy as np
    from matplotlib.pyplot import figure, show
    from matplotlib.ticker import MaxNLocator

    ax = figure().gca()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axvline(x=number_of_components, linewidth=1, color='r', alpha=0.5)
    plt.title('Explained Variance of PCA by Component')
    show()
    
def compute_pca(new_selection, scaledData, verbose = None, visualise = None):
    '''
    Description:
    Create a Principal component analysis
    '''
    n_components = min(len(new_selection.columns), len(new_selection))
    pca = PCA(n_components= n_components)
    pca.fit(scaledData)
    pca_data = pca.transform(scaledData)
    per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
    number_of_components = len(np.where(pca.explained_variance_ > 1.0)[0])

    if visualise:
        scree_plot(pca)

    if verbose:
        print(f'Explained variance ratio {per_var}%')
        print(f'Number of components required; {number_of_components}')
        var_explained(number_of_components, pca)
        explained_variance = np.sum(pca.explained_variance_ratio_[:number_of_components])
        print(f'Explained variance: {explained_variance}')

    return number_of_components


def compute_pca2(number_of_components, scaledData, new_selection):
    '''
    Description:
    Create a Principal component analysis with less variables
    '''
    # Again PCA but now with less variables
    pca = PCA(n_components= number_of_components)
    pca.fit(scaledData)
    pca_data = pca.transform(scaledData)
    per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)  
    print(f'Explained variance ratio {per_var}%')
    
    loadings = pd.DataFrame(pca.components_)
    loadings.columns = new_selection.columns
    return pca, loadings, pca_data



def cor_comp_var(pca, new_selection):
    '''
    Description:
    Determine whcih features load strongly on which components
    '''
    loadings_corr =  np.sqrt(pca.explained_variance_)*pca.components_.T 
    loadings_corr_df = pd.DataFrame(loadings_corr.transpose(), columns =new_selection.columns )
    
    features = {}
    for column, row in loadings_corr_df.iterrows():
        top_features = dict(row.abs().sort_values(ascending = False)[:10])
        label = f'PCA_{column}'
        features[label] = top_features
    return features, loadings_corr


# # Promax rotation of the components

def rotate_pca(loadings_corr, new_selection):
    '''
    Description:
    Rotate the components based on the loadings
    '''

    rotator = Rotator(method='varimax', normalize = True)
    rotated_loadings = rotator.fit_transform(loadings_corr)
    labels = [f'PC{str(x)}' for x in range(1, loadings_corr.shape[1]+1)]
    rotated_loadings_df= pd.DataFrame(rotated_loadings, columns = labels, index = new_selection.columns)
    return rotated_loadings_df,rotated_loadings,  rotator


# Visualise 
def bi_plot(rotated_loadings_df, coef1 = 'PC1', coef2 = 'PC2'):
    '''
    Description:
    Create a biplot of the data
    '''
    coeff = rotated_loadings_df.loc[:,coef1:coef2]
    def myplot(coeff,labels=None):
        n = coeff.shape[0]
        for i in range(n):
            plt.arrow(0, 0, coeff.iloc[i,0], coeff.iloc[i,1],color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff.iloc[i, 0] * 1.15, coeff.iloc[i, 1] * 1.15, f"Var{str(i + 1)}", color='g', ha='center', va='center')

            else:
                plt.text(coeff.iloc[i,0]* 1.15, coeff.iloc[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.grid()

    #Call the function. Use only the 2 PCs.
    myplot(coeff, labels = coeff.index)
    plt.show()
    



# Rotate the PCA
def varimax(loadings, normalize = True, max_iter = 500, tol=1e-5):
    """
    Perform varimax (orthogonal) rotation, with optional
    Kaiser normalization.

    Parameters
    ----------
    loadings : array-like
        The loading matrix

    Returns
    -------
    loadings : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_mtx : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    """
    X = loadings.copy()
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X

    # normalize the loadings matrix
    # using sqrt of the sum of squares (Kaiser)
    if normalize:
        normalized_mtx = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, X.copy())
        X = (X.T / normalized_mtx).T

    # initialize the rotation matrix
    # to N x N identity matrix
    rotation_mtx = np.eye(n_cols)

    d = 0
    for _ in range(max_iter):

        old_d = d

        # take inner product of loading matrix
        # and rotation matrix
        basis = np.dot(X, rotation_mtx)

        # transform data for singular value decomposition using updated formula :
        # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
        diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
        transformed = X.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

        # perform SVD on
        # the transformed matrix
        U, S, V = np.linalg.svd(transformed)

        # take inner product of U and V, and sum of S
        rotation_mtx = np.dot(U, V)
        d = np.sum(S)

        # check convergence
        if d < old_d * (1 + tol):
            break

    # take inner product of loading matrix
    # and rotation matrix
    X = np.dot(X, rotation_mtx)

    # de-normalize the data
    if normalize:
        X = X.T * normalized_mtx
    else:
        X = X.T

    # convert loadings matrix to data frame
    loadings = X.T.copy()
    return loadings, rotation_mtx

def promax(loadings, normalize = True, max_iter = 500, tol=1e-5, power=4):
    """
    Perform promax (oblique) rotation, with optional
    Kaiser normalization.
    Parameters
    ----------
    loadings : array-like
        The loading matrix
    Returns
    -------
    loadings : numpy array, shape (n_features, n_factors)
        The loadings matrix
    rotation_mtx : numpy array, shape (n_factors, n_factors)
        The rotation matrix
    psi : numpy array or None, shape (n_factors, n_factors)
        The factor correlations
        matrix. This only exists
        if the rotation is oblique.
    """
    X = loadings.copy()
    n_rows, n_cols = X.shape
    if n_cols < 2:
        return X

    if normalize:
        # pre-normalization is done in R's
        # `kaiser()` function when rotate='Promax'.
        array = X.copy()
        h2 = sp.diag(np.dot(array, array.T))
        h2 = np.reshape(h2, (h2.shape[0], 1))
        weights = array / sp.sqrt(h2)

    else:
        weights = X.copy()

    # first get varimax rotation
    X, rotation_mtx = varimax(weights)
    Y = X * np.abs(X)**(power - 1)

    # fit linear regression model
    coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

    # calculate diagonal of inverse square
    try:
        diag_inv = sp.diag(sp.linalg.inv(sp.dot(coef.T, coef)))
    except np.linalg.LinAlgError:
        diag_inv = sp.diag(sp.linalg.pinv(sp.dot(coef.T, coef)))

    # transform and calculate inner products
    coef = sp.dot(coef, sp.diag(sp.sqrt(diag_inv)))
    z = sp.dot(X, coef)

    if normalize:
        # post-normalization is done in R's
        # `kaiser()` function when rotate='Promax'
        z = z * sp.sqrt(h2)

    rotation_mtx = sp.dot(rotation_mtx, coef)

    coef_inv = np.linalg.inv(coef)
    phi = np.dot(coef_inv, coef_inv.T)

    # convert loadings matrix to data frame
    loadings = z.copy()
    return loadings, rotation_mtx, phi




