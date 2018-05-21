import numpy as np

def posterior(K, K_pred, K_new, y, noise_var):
    # returns the posterior mean and variance of y^* = f(x^*) + epsilon where f is GP(0, k), epsilon ~ N(0, noise_var I ) 
    # y is the observation resulting from having observed X and K = k(X, X), K_pred = k(x^*, X), k_new = K(x^*, x^*)
    # K_pred.shape = (x^*.shape[0], x.shape[0])
    # K_inv = np.linalg.inv(K)
    precomp = np.linalg.solve(K, K_pred.T).T # = K_pred * K^{-1}
    mean = np.dot(precomp, y) # np.dot(K_pred, np.dot(K_inv, y))
    var = K_new - np.dot(precomp, K_pred.T) # np.dot(K_pred, np.dot(K_inv, K_pred.T))
    return mean, var

def posterior_from_feature_gen(X_train, y_train, X_pred, noise_var, feature_gen):
    PhiX_train = feature_gen(X_train)
    PhiX_pred = feature_gen(X_pred)

    # poor performance, is there a better way to avoid computing inverses by exploiting the structure K = np.dot(Phi, Phi.T)?
    K_inv = np.dot(PhiX_train, PhiX_train.T)
    K_pred = np.dot(PhiX_pred, PhiX_train.T)
    K_new =   np.dot(PhiX_pred, PhiX_pred.T)
    return posterior(K_inv, K_pred, K_new, y_train, noise_var)

def posterior_from_kernel_gen(X_train, y_train, X_pred, noise_var, kernel_gen):
    K = kernel_gen(X_train, X_train)
    K_pred = kernel_gen(X_pred, X_train)
    K_new = kernel_gen(X_pred, X_pred)
    return posterior(K, K_pred, K_new, y_train, noise_var)
