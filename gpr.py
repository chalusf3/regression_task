import numpy as np

def posterior_mean_var(K, K_pred, K_new, y, noise_var):
    # returns the posterior mean and variance of y^* = f(x^*) + epsilon where f is GP(0, k), epsilon ~ N(0, noise_var I ) 
    # y is the observation resulting from having observed X and K = k(X, X), K_pred = k(x^*, X), k_new = K(x^*, x^*)
    # K_pred.shape = (x^*.shape[0], x.shape[0])
    K_inv = np.linalg.inv(K)
    mean = np.dot(K_pred, np.dot(K_inv, y))
    var = K_new - np.dot(K_pred, np.dot(K_inv, K_pred.T))
    return mean, var

