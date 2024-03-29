"""
Main high-level functions for computing the estimator.
The main function is estimator_unknown_ranks.

Created on Wed Nov 11 12:02:50 2020

@author : jeremylhour
"""
import numpy as np
from numba import njit

from statsmodels.distributions.empirical_distribution import ECDF

from kernelDensityEstimator import kernel_density_estimator
from empiricalCDF import smoothed_ecdf

# ------------------------------------------------------------------------------------
# INVERSE DENSITY ESTIMATORS
# ------------------------------------------------------------------------------------
@njit
def inv_density_LS(u_hat, y):
    """
    inv_density_LS :
        returns u_hat and inv_density using the Lewbel-Schennach (2007) method.
        Also changes the u_hat by removing duplicates
    
    Source :
        "A Simple Ordered Data Estimator For Inverse Density Weighted Functions,"
        Arthur Lewbel and Susanne Schennach, Journal of Econometrics, 2007, 186, 189-211.
    
    Args:
        u_hat (np.array): output of counterfactual_ranks function.
        y (np.array): outcome.

    Return:
        u_hat, inv_density (np.array)
    """
    u_hat = np.unique(np.sort(u_hat)) # order and remove duplicates
    F_inverse = np.quantile(y, u_hat)
    inv_density = (F_inverse[1:] - F_inverse[:-1]) / (u_hat[1:] - u_hat[:-1])
    u_hat = (u_hat[1:] + u_hat[:-1]) / 2 # replaces u_hat by the average of two consecutive u_hat
    return u_hat, inv_density

@njit
def inv_density_Xavier(u_hat, y, spacing_2: bool = True):
    """
    inv_density_Xavier :
        returns inv_density using the Xavier method
    
    Args:
        u_hat (np.array): output of counterfactual_ranks function
        y (np.array): outcome
        spacing_2 (bool): If True, two spacings between data points as in the points used for U_i will be
            U_i+ and U_i-. If not, it's U_i+ and U_i
    
    Return:
        inv_density (np.array)
    """
    u_hat = np.sort(u_hat)
    # find distinct values just above and just below
    ub, lb  = [], []
    for u in u_hat:
        # ABOVE -- upper bound
        if u == np.max(u_hat):
            ub.append(u)
        else:
            ub.append(np.min(np.array([item for item in u_hat if item > u])))
        
        # BELOW -- lower bound
        if u == np.min(u_hat):
            lb.append(u)
        else:
            if spacing_2 or u == np.max(u_hat):
                lb.append(np.max(np.array([item for item in u_hat if item < u])))
            else:
                lb.append(u)
                
    ub, lb = np.array(ub), np.array(lb)
    inv_density = (np.quantile(y, ub) - np.quantile(y, lb)) / (ub - lb)
    return inv_density

# ------------------------------------------------------------------------------------
# LOWER-LEVEL FUNCTIONS
# ------------------------------------------------------------------------------------
@njit
def compute_zeta(u_hat, inv_density, size):
    """
    compute_zeta :
        function to compute zeta, similar to Q in Athey and Imbens (2006).
        In our paper, it might be called 'eta' instead.
    
    Args:
        u_hat (np.array): output of counterfactual_ranks function
        inv_density (np.array): output of any of the inv_density functions
        size (int): size of the support
    
    Return:
        zeta (np.array)
    """
    support = np.linspace(1 / size, 1, size) # = F_y(Y)
    zeta = np.empty(size)
    for i in range(size):
        indicator = (support[i] <= u_hat)
        inside_integral = -(indicator - u_hat) * inv_density
        zeta[i] = np.mean(inside_integral)
    return zeta

@njit
def compute_phi(u_hat, inv_density, size):
    """
    compute_phi :
        function to compute phi, similar to P in Athey and Imbens (2006).
    
    Args:
        u_hat (np.array): output of counterfactual_ranks function
        inv_density (np.array): output of any of the inv_density functions
        size (int): size of the support
    
    Return:
        phi (np.array)
    """
    support = np.linspace(1 / size, 1, size) # = F_z(Z)
    phi = np.empty(size)
    for i in range(size):
        indicator = (support[i] <= u_hat)
        inside_integral = (indicator - u_hat) * inv_density
        phi[i] = np.mean(inside_integral)
    return phi

@njit
def compute_theta(u_hat, y):
    """
    compute_theta :
        returns theta_hat and counterfactual outcome
    
    Args:
        u_hat (np.array): output of counterfactual_ranks function
        y (np.array): outcome
    
    Return:
        counterfactual_y, theta_hat (np.array)
    """
    counterfactual_y = np.quantile(y, u_hat)
    theta_hat = np.mean(counterfactual_y)
    return counterfactual_y, theta_hat

def counterfactual_ranks(points_to_predict, points_for_distribution, method: str = "smoothed"):
    """
    counterfactual_ranks :
        compute \widehat U the value of the CDF at each element of points_to_predict,
        using the empirical CDF defined by 'points_for_distribution'.
    
    Args:
        points_to_predict (np.array): points for wich to get the rank in the distribution
        points_for_distribution (np.array): points for which to compute the CDF
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    
    Return:
        u_hat (np.array)
    """
    if method == "smoothed":
        u_hat = smoothed_ecdf(new_points=points_to_predict, data=points_for_distribution)
    elif method == "standard":
        ecdf = ECDF(points_for_distribution)
        u_hat = ecdf(points_to_predict)
    else:
        raise ValueError("'method' argument for counterfactual ranks needs to be either 'smoothed' or 'standard'.")
    return u_hat

# ------------------------------------------------------------------------------------
# UNKNOWN RANKS
# ------------------------------------------------------------------------------------
def estimator_unknown_ranks(y, x, z, method="smoothed", se_method="kernel", bootstrap_quantile=None):
    """
    estimator_unknown_ranks :
        computes the estimator (1), i.e. average of quantiles of the outcome for each estimated rank,
        and corresponding standard error. "lewbel-schennach" implement estimation of s.e. based on Lewbel and Schennach's paper
    
    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome ot untreated group at date 1.
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
        se_method (str): can be "kernel", "lewbel-schennach", or "xavier" depending on the type of method for computing 1/f(F^{-1}(u_hat)).
        bootstrap_quantile (list): Bootstrap quantiles to compute, if None, skip this
    
    Return:
        estimator and standard errors (np.array)
    """
    u_hat = counterfactual_ranks(points_to_predict=x, points_for_distribution=z, method=method)
        
    """
    Estimator of theta
    """
    counterfactual_y, theta_hat = compute_theta(u_hat=u_hat, y=y)
    
    """
    Compute quantiles and stop, if needed.
    """
    if bootstrap_quantile:
        theta_bootstrap = bootstrap_sample(y, x, z, method=method)
        return theta_hat, np.quantile(theta_bootstrap, q=bootstrap_quantile)
    
    """
    Compute inv_density depending on the method of choice.
    """ 
    if se_method == "kernel":
        inv_density = 1 / kernel_density_estimator(x=np.quantile(y, u_hat), data=y)
    elif se_method == "lewbel-schennach":
        u_hat, inv_density = inv_density_LS(u_hat=u_hat, y=y)
    elif se_method == "xavier":
        inv_density = inv_density_Xavier(u_hat=u_hat, y=y)
    else:
        raise ValueError("'se_method' argument should be 'kernel', 'lewbel-schennach' or 'xavier'.")
        
    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    zeta = compute_zeta(u_hat=u_hat, inv_density=inv_density, size=len(y))
    
    """
    compute_phi:
        compute vector phi_i as in out paper,
        similar to P in Athey and Imbens (2006).
    """
    phi = compute_phi(u_hat=u_hat, inv_density=inv_density, size=len(z))

    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = theta_hat - counterfactual_y
    
    """
    compute standard error
    """
    se = np.sqrt(np.mean(zeta ** 2) + np.mean(phi ** 2) + np.mean(epsilon ** 2))

    return theta_hat, se / np.sqrt(len(y))

# ------------------------------------------------------------------------------------
# KNOWN RANKS
# ------------------------------------------------------------------------------------
def estimator_known_ranks(y, u):
    """
    estimator_known_ranks :
        computes the estimator (1), i.e. average of quantiles of the outcome for each rank
    
    WARNING : 
        This function is a particular case of the previous one,
        it can probably be included with the right options, but I haven't took the time to do it.
    
    Args:
        y: np.array of the outcome -- corresponds to outcome of untreated group at date 1.
        u: np.array of the ranks
    """
    inv_density = 1/kernel_density_estimator(x=np.quantile(y, u), data=y) 
    counterfactual_y, theta_hat = compute_theta(u_hat=u, y=y)
            
    """
    compute_zeta:
        compute vector zeta_i as in out paper,
        similar to Q in Athey and Imbens (2006).
    """
    zeta = compute_zeta(u_hat=u, inv_density=inv_density, size=len(y))
    
    """
    compute_epsilon:
        Formula of Athey and Imbens (2006)
    """
    epsilon = -(counterfactual_y - theta_hat)
    
    """
    compute standard error
    """
    se = np.sqrt(np.mean(zeta**2) + np.mean(epsilon**2))
    
    return theta_hat, se/np.sqrt(len(y))

# ------------------------------------------------------------------------------------
# BOOTSTRAP PROCEDURE
# ------------------------------------------------------------------------------------
def bootstrap_sample(y, x, z, B=999, method="smoothed"):
    """
    bootstrap_sample :
        computes the bootstrapped sample
    
    Args:
        y (np.array): the outcome -- corresponds to outcome of untreated group at date 1.
        x (np.array): the points to project -- corresponds to outcome of treated group at date 0.
        z (np.array): the points for distribution -- corresponds to outcome ot untreated group at date 1.
        B (int): number of bootstrap samples
        method (str): can be "smoothed" or "standard" dependant on the type of method for computation of the CDF
    
    Return:
        (np.array)
    """
    theta_bootstrap = np.empty(shape=(B,))
    
    if method == "smoothed":
        for b in range(B):
            u_hat = counterfactual_ranks(
                points_to_predict=np.random.choice(x, size=len(x), replace=True),
                points_for_distribution=np.unique(np.random.choice(z, size=len(z), replace=True)),
                method="smoothed"
                )
            _, theta_hat = compute_theta(
                u_hat=u_hat,
                y=np.random.choice(y, size=len(y), replace=True)
                )
            theta_bootstrap[b] = theta_hat
    
    elif method == "standard":
        for b in range(B):
            u_hat = counterfactual_ranks(
                points_to_predict=np.random.choice(x, size=len(x), replace=True),
                points_for_distribution=np.unique(np.random.choice(z, size=len(z), replace=True)),
                method="standard"
                )
            _, theta_hat = compute_theta(
                u_hat=u_hat,
                y=np.random.choice(y, size=len(y), replace=True)
                )
            theta_bootstrap[b] = theta_hat
            
    else:
        raise ValueError("'method' argument for counterfactual ranks needs to be either 'smoothed' or 'standard'.")
            
    return theta_bootstrap

if __name__ == '__main__':
    pass