import numpy as np

def lin_reg(X, y, noise_var):
    # returns the coefficient vector beta for y = X * beta + epsilon
    # prior on beta and a N(0,I noise_var) on the noise
    beta = np.linalg.solve((noise_var * np.eye(X.shape[1]) + np.dot(X.T, X)), np.dot(X.T, y))
    return beta

def fit_kernel(K_pred, K, y, noise_var):
    return np.dot(K_pred, np.linalg.solve(noise_var * np.eye(K.shape[0]) + K, y))

