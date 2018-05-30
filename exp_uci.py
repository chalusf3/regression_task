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
    # y = whiten_data(y)[0]
    X_train, y_train, X_test, y_test = split_data(X, y, 0.7) 
    X_train, y_train, X_cv, y_cv = split_data(X_train, y_train, 0.667) 
    # X_train = 70 % of all data
    # X_test = 20 % of all data
    # X_cv = 10 % of all data

    noise_var = 1.0
    scale = 8.0
    n_rff = 64
    seed = 0
    # Fit a GP with kernel 
    print 'Start'
    y_cv_gp_kernel, _ = gpr.posterior_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    print 'Done'
    # print y_cv - y_cv_gp_kernel
    print np.linalg.norm(y_cv - y_cv_gp_kernel) / y_cv.shape[0]

    # Fit a GP with random features 
    print 'Start'
    y_cv_gp_feature, _ = gpr.posterior_from_feature_gen(X_train, y_train, X_cv, noise_var, lambda a: kernels.iid_gaussian_RFF(a, n_rff, seed, scale))
    print 'Done'
    print np.linalg.norm(y_cv - y_cv_gp_feature) / y_cv.shape[0]
    print np.linalg.norm(y_cv_gp_kernel - y_cv_gp_feature) / y_cv.shape[0]


    # Fit with kernel trick
    print 'Start'
    y_cv_fit_kernel = krr.fit_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    print 'Done'

    # Fit with feature regression
    print 'Start'
    angle = 0.3 * np.pi
    y_cv_fit_feature = krr.fit_from_feature_gen(X_train, y_train, X_cv, noise_var, lambda a: kernels.angled_gaussian_RFF(a, n_rff, seed, scale, angle))
    print 'Done'

    print np.linalg.norm(y_cv - y_cv_fit_kernel) / y_cv.shape[0]
    print np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]
    print np.linalg.norm(y_cv_fit_kernel - y_cv_fit_feature) / y_cv.shape[0]
    