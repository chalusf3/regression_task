import numpy as np

def lin_reg(X, y, noise_var):
    # returns the coefficient vector beta for y = X * beta + epsilon
    # N(0,1) prior on beta and a N(0,I noise_var) on the noise
    beta = np.linalg.solve((noise_var * np.eye(X.shape[1]) + np.dot(X.T, X)), np.dot(X.T, y))
    return beta

def fit_kernel(K_pred, K, y, noise_var):
    # returns the MLE fit of the x_pred under the model y = loc * beta + N(0, noise_var) with a N(0,1) prior on beta
    # K_pred[i,j] = k(x_pred_i, loc_j) (loc are the points used to create features / at training time)
    # K[i,j] = k(loc_i, loc_j) 
    # y_i = output from loc_i
    return np.dot(K_pred, np.linalg.solve(noise_var * np.eye(K.shape[0]) + K, y))

