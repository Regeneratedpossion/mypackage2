
#import required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
import warnings
import random

warnings.filterwarnings('ignore')

def generate_gmm_2d(K, n_samples, means, covariances, weights, random_state=None):
    """
    Generate 2D Gaussian Mixture Model (GMM) dataset
    
    Parameters:
    
        K (int): Number of Gaussian components
        
        n_samples (int): Total number of samples
        
        means (list of array-like):  List of K mean vectors, each of shape (2,)
        
        covariances (list of array-like): List of K covariance matrices, each of shape (2,2)
        
        weights (array-like): Mixing proportions, shape (K,), must sum to 1
        
        random_state (int):Random seed 
        
    Returns:
        X (np.ndarray):Generated data points, shape (n_samples, 2)

        y (np.ndarray): True class labels, shape (n_samples,)
    """

    
    # === Parameter validation ===
    assert len(means) == K, "Number of means must match K"
    assert len(covariances) == K, "Number of covariances must match K"
    assert np.isclose(np.sum(weights), 1.0), "Sum of weights must be 1"
    np.random.seed(random_state) ,"Fix random seed"

    
    # === 1. Generate class labels from mixing proportions ===
    # Randomly assign component labels based on weights
    
    y = np.random.choice(K, size=n_samples, p=weights)
    
    # 2. === 2. Generate data points from corresponding Gaussians ===
    
    X = np.zeros((n_samples, 2))  # Initialize data matrix
    for k in range(K): #Find indices of samples belonging to component k
        
        indices = np.where(y == k)[0]
        if len(indices) > 0: # Generate data from the k-th Gaussian
           X[indices] = multivariate_normal.rvs(
    mean=means[k], 
    cov=covariances[k], 
    size=len(indices)
)  
    return X, y
