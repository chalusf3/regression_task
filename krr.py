import numpy as np

def lin_reg(X, y, noise_var):
    # returns the coefficient vector beta for y = X * beta + epsilon
    # N(0,1) prior on beta and a N(0,I noise_var) on the noise
    beta = np.linalg.solve((noise_var * np.eye(X.shape[1]) + np.dot(X.T, X)), np.dot(X.T, y))
    return beta

"""
def fit_kernel(K_pred, K, y, noise_var):
    # returns the MLE fit of the x_pred under the model y = loc * beta + N(0, noise_var) with a N(0,1) prior on beta
    # K_pred[i,j] = k(x_pred_i, loc_j) (loc are the points used to create features / at training time)
    # K[i,j] = k(loc_i, loc_j) 
    # y_i = output from loc_i
    return np.dot(K_pred, np.linalg.solve(noise_var * np.eye(K.shape[0]) + K, y))
"""

def fit_from_feature_gen(X_train, y_train, X_pred, noise_var, feature_gen):
    # fits X_pred against the model y_train = feature_gen(X_train) * beta + N(0, noise_var) with a N(0,I) prior on beta
    PhiX_train = feature_gen(X_train)
    PhiX_pred = feature_gen(X_pred)

    if PhiX_train.shape[1] < PhiX_train.shape[0]:
        y_pred = np.dot(PhiX_pred, np.linalg.solve(noise_var * np.eye(PhiX_train.shape[1]) + np.dot(np.conj(PhiX_train.T), PhiX_train), np.dot(np.conj(PhiX_train.T), y_train)))
    else:
        y_pred = np.dot(np.dot(PhiX_pred, np.conj(PhiX_train.T)), np.linalg.solve(noise_var * np.eye(PhiX_train.shape[0]) + np.dot(PhiX_train, np.conj(PhiX_train.T)), y_train))
    return np.real(y_pred)

def fit_from_kernel_gen(X_train, y_train, X_pred, noise_var, kernel_gen):
    # fits X_pred against the model y_train = feature_gen(X_train) * beta + N(0, noise_var) with a N(0,I) prior on beta
    # where kernel_gen(x,y) = <feature_gen(x), feature_gen(y)>
    K_train = kernel_gen(X_train, X_train)
    K_pred = kernel_gen(X_pred, X_train)

    y_pred = np.dot(K_pred, np.linalg.solve(noise_var * np.eye(K_train.shape[0]) + K_train, y_train))    
    return y_pred
