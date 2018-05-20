import numpy as np
import csv, timeit
import krr, gpr, kernels

def load_wine_dataset():
    X = np.zeros(shape = (1599, 11))
    y = np.zeros((1599, 1))
    
    with open('datasets/winequality-red.csv', 'rb') as f:
        next(f)
        c = 0
        reader = csv.reader(f, delimiter = ';', quotechar = '"')
        for row in reader:
            X[c] = row[0:11]
            y[c] = row[11]
            c+=1
    return X, y

def shuffle(X, y):
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]

def split_data(X, y, ratio = 0.8):
    X, y = shuffle(X, y)
    
    train_ratio = 0.8
    train_count = int(train_ratio * X.shape[0])
    X_train = X[:train_count]
    y_train = y[:train_count]
    X_test = X[train_count:]
    y_test = y[train_count:]
    return X_train, y_train, X_test, y_test

def whiten_data(X):
    means = np.mean(X, axis = 0)
    stds = np.std(X, axis = 0)
    X = (X - means) / stds
    return X, means, stds

if __name__ == '__main__':
    np.random.seed(0)
    X, y = load_wine_dataset()
    X = whiten_data(X)[0]
    X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
    X_train, y_train, X_cv, y_cv = split_data(X_train, y_train, 7.0/8.0)

    noise_var = 1.0
    scale = 8
    n_rff = 64
    seed = 1
    # Fit with kernel trick
    print 'Start'
    K_train = kernels.gaussian_kernel_gram(X_train, scale)
    K_pred = kernels.gaussian_kernel(X_cv, X_train, scale)
    
    y_cv_fit_kernel = krr.fit_kernel(K_pred, K_train, y_train, noise_var)
    print 'Done'

    # Fit with feature regression
    print 'Start'
    PhiX_train = kernels.iid_gaussian_RFF(X_train, n_rff, seed, scale)
    beta_feature = krr.lin_reg(PhiX_train, y_train, noise_var)
    PhiX_cv = kernels.iid_gaussian_RFF(X_cv, n_rff, seed, scale)
    y_cv_fit_feature = np.dot(PhiX_cv, beta_feature)
    print 'Done'
    
    # Fit by computing the kernel from the RFF and doing kernelized regression -> slow
    print 'Start'
    PhiX_train = kernels.iid_gaussian_RFF(X_train, n_rff, seed, scale)
    PhiX_cv = kernels.iid_gaussian_RFF(X_cv, n_rff, seed, scale)
    K_train = np.dot(PhiX_train, PhiX_train.T)
    K_pred = np.dot(PhiX_cv, PhiX_train.T)

    y_cv_fit_feature_kernelized = krr.fit_kernel(K_pred, K_train, y_train, noise_var)
    print 'Done'

    print np.linalg.norm(y_cv - y_cv_fit_kernel) / y_cv.shape[0]
    print np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]
    print np.linalg.norm(y_cv - y_cv_fit_feature_kernelized) / y_cv.shape[0]