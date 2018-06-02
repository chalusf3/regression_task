import numpy as np
import matplotlib.pyplot as plt
import csv, timeit, pickle
import krr, gpr, kernels
from datetime import timedelta, datetime

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
    # noise_var = 1.0, scale = 8.0
    
def load_air_quality_dataset():
    def format_row(input):
        input = filter(None, input)
        if len(input):
            date = datetime.strptime(','.join(input[0:2]), '%d/%m/%Y,%H.%M.%S')
            expl = [float(x.replace(',', '.')) for x in input[3:4]+input[5:]]
            response = float(input[2].replace(',', '.'))
            return date, expl, response
        else:
            return None, None, None
    
    X = [None]
    y = [None]

    with open('datasets/AirQualityUCI.csv', 'rb') as f:
        next(f)

        reader = csv.reader(f, delimiter = ';', quotechar = '"')
        start_date, expl, y[0] = format_row(reader.next())
        X[0] = [0]+expl
        # start_date = datetime.strptime(','.join(row[0:2]), '%d/%m/%Y,%H.%M.%S')

        # X.append([0] + [float(x.replace(',', '.')) for x in row[3:-1]])
        # y.append(float(row[2].replace(',', '.')))
        for row in reader:
            row_date, expl, response = format_row(row)
            # row = filter(None, row)
            # row = row[0:4]+row[5:]

            if not row_date or not len(row) or -200 in row or response == -200:
                continue
        
            X.append([(row_date - start_date).total_seconds()] + expl)
            y.append(response)
        
        X = np.matrix(X)
        y = np.matrix(y).T
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

def krr_tune(X_train, y_train, X_cv, y_cv):
    noise_vars = np.linspace(0.3, 2.0, 3)
    scales = np.linspace(3, 15, 3)
    errors = np.zeros((len(noise_vars), len(scales)))
    for i, noise_var in enumerate(noise_vars):
        for j, scale in enumerate(scales):
            y_cv_fit_kernel = krr.fit_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
            print 'noise_var = %f\tscale = %f\terror = %f' % (noise_var, scale, np.linalg.norm(y_cv_fit_kernel - y_cv) / y_cv.shape[0])
            errors[i,j] = np.linalg.norm(y_cv_fit_kernel - y_cv) / y_cv.shape[0]
    print np.round(errors, decimals = 4)

def regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale):
    algo_names = ['iid', 'iid_anti', 'ort', 'ort_anti', 'HD_1', 'HD_2', 'HD_3']
    feature_gen_handles = [lambda a: kernels.iid_gaussian_RFF(a, n_rff, 0, scale), \
                           lambda a: kernels.make_antithetic(kernels.iid_gaussian_RFF(a, n_rff, 0, scale)), \
                           lambda a: kernels.ort_gaussian_RFF(a, n_rff, 0, scale), \
                           lambda a: kernels.make_antithetic(kernels.ort_gaussian_RFF(a, n_rff, 0, scale)), \
                           lambda a: kernels.HD_gaussian_RFF(a, n_rff, 0, scale, 1), \
                           lambda a: kernels.HD_gaussian_RFF(a, n_rff, 0, scale, 2), \
                           lambda a: kernels.HD_gaussian_RFF(a, n_rff, 0, scale, 3)]
    
    n_rffs = [4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]
    for algo_name, feature_gen_handle in zip(algo_names, feature_gen_handles):
        errors = {}
        for n_rff in n_rffs:
            y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, feature_gen_handle)
            errors[n_rff] = np.linalg.norm(y_test_fit - y_test) / y_test.shape[0]
            print n_rff, errors[n_rff]
        with open('%s_%s_krr.pk' % (data_name, algo_name), 'wb') as f:
            pickle.dump(errors, f)
    
    errors = {}
    for n_rff in n_rffs[:1] + n_rffs[-1:]:
        y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
        errors[n_rff] = np.linalg.norm(y_test_fit - y_test) / y_test.shape[0]
        print n_rff, errors[n_rff]
    with open('%s_exact_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)

def plot_regression_errors(data_name, algo_names):
    for algo_name in algo_names:
        with open('%s_%s_krr.pk' % (data_name, algo_name), 'rb') as f:
            data = pickle.load(f)
        x = data.keys()
        x.sort()
        y = [data[k] for k in x]
        plt.loglog(x, y, '.-', label = algo_name)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    data_name = ['wine', 'airq'][0]
    if data_name == 'wine':
        X, y = load_wine_dataset()
    elif data_name == 'airq':
        X, y = load_air_quality_dataset()
    
    X = whiten_data(X)[0]
    X_train, y_train, X_test, y_test = split_data(X, y, 0.7) 
    X_train, y_train, X_cv, y_cv = split_data(X_train, y_train, 0.667) 
    # X_train = 70 % of all data
    # X_test =  20 % of all data
    # X_cv =    10 % of all data

    noise_var = 1.0
    scale = 8.0
    n_rff = 64
    seed = 0

    # regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale)
    # plot_regression_errors(data_name, ['exact', 'iid', 'iid_anti', 'ort', 'ort_anti', 'HD_1', 'HD_2', 'HD_3'])

    # Fit a GP with random features 
    print 'Start'
    y_cv_gp_feature, _ = gpr.posterior_from_feature_gen(X_train, y_train, X_cv, noise_var, 
                                                        lambda a: kernels.ort_gaussian_RFF(a, n_rff, seed, scale))
    print 'Done'
    print np.linalg.norm(y_cv - y_cv_gp_feature) / y_cv.shape[0]
    
    # Fit a GP with kernel 
    print 'Start'
    y_cv_gp_kernel, _ = gpr.posterior_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    print 'Done'
    print np.linalg.norm(y_cv - y_cv_gp_kernel) / y_cv.shape[0]
    print np.linalg.norm(y_cv_gp_kernel - y_cv_gp_feature) / y_cv.shape[0]

    # Fit with kernel trick
    print 'Start'
    y_cv_fit_kernel = krr.fit_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    print 'Done'
    print np.linalg.norm(y_cv - y_cv_fit_kernel) / y_cv.shape[0]

    y_cv_fit_feature = krr.fit_from_feature_gen(X_train, y_train, X_cv, noise_var, 
                                                lambda a: kernels.iid_gaussian_RFF(a, n_rff, seed, scale))
    y_cv_fit_feature = np.real(y_cv_fit_feature)
    print np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]

    # Fit with feature regression
    # print 'Start'
    # for x in np.linspace(0, 1, 11):
    #     angle = x * np.pi
        
    #     y_cv_fit_feature = krr.fit_from_feature_gen(X_train, y_train, X_cv, noise_var, 
    #                                                 lambda a: kernels.angled_gaussian_RFF(a, n_rff, seed, scale, angle))
    #     print x, np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]
    # print 'Done'
    # print np.linalg.norm(y_cv_fit_kernel - y_cv_fit_feature) / y_cv.shape[0]
       