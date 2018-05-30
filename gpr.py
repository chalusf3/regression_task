import numpy as np

def posterior(K, K_pred, K_new, y, noise_var):
    # returns the posterior mean and variance of y^* = f(x^*) + epsilon where f is GP(0, k), epsilon ~ N(0, noise_var I ) 
    # y is the observation resulting from having observed X and K = k(X, X), K_pred = k(x^*, X), k_new = K(x^*, x^*)
    # K_pred.shape = (x^*.shape[0], x.shape[0])

    precomp = np.linalg.solve(K + noise_var * np.eye(K.shape[0]), K_pred.T).T # = K_pred * K^{-1}
    mean = np.dot(precomp, y)
    var = K_new - np.dot(precomp, K_pred.T) # np.dot(K_pred, np.dot(K_inv, K_pred.T))
    return mean, var

def posterior_from_feature_gen(X_train, y_train, X_pred, noise_var, feature_gen):
    PhiX_train = feature_gen(X_train)
    PhiX_pred =  feature_gen(X_pred)

    if PhiX_train.shape[0] > PhiX_train.shape[1]:
        precomp = np.linalg.solve(np.dot(PhiX_train.T, PhiX_train) + noise_var * np.eye(PhiX_train.shape[1]), np.concatenate([np.dot(PhiX_train.T, y_train), 
                                                                                                                              np.dot(np.dot(PhiX_train.T, PhiX_train), PhiX_pred.T)], axis = 1))
        
        mean = np.dot(PhiX_pred, precomp[:, 0, np.newaxis])
        var = np.dot(PhiX_pred, PhiX_pred.T) - np.dot(PhiX_pred, precomp[:, 1:])
        return mean, var
    else:
        K = np.dot(PhiX_train, PhiX_train.T)
        K_pred = np.dot(PhiX_pred, PhiX_train.T)
        K_new = np.dot(PhiX_pred, PhiX_pred.T)
        return posterior(K, K_pred, K_new, y_train, noise_var)

    # # poor performance, is there a better way to avoid computing inverses by exploiting the structure K = np.dot(Phi, Phi.T)?
    # K =  np.dot(PhiX_train, PhiX_train.T)
    # K_pred = np.dot(PhiX_pred, PhiX_train.T)
    # K_new =  np.dot(PhiX_pred, PhiX_pred.T)
    # return posterior(K, K_pred, K_new, y_train, noise_var)

def posterior_from_kernel_gen(X_train, y_train, X_pred, noise_var, kernel_gen):
    K = kernel_gen(X_train, X_train)
    K_pred = kernel_gen(X_pred, X_train)
    K_new = kernel_gen(X_pred, X_pred)
    return posterior(K, K_pred, K_new, y_train, noise_var)
