import numpy as np
import matplotlib.pyplot as plt
import csv, timeit, pickle
import krr, gpr, kernels
from datetime import timedelta, datetime
from collections import defaultdict
import scipy.linalg as sp_la

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
    X, y = shuffle(X[:], y[:])
    
    train_ratio = 0.8
    train_count = int(train_ratio * X.shape[0])
    X_train = X[:train_count]
    y_train = y[:train_count]
    X_test = X[train_count:]
    y_test = y[train_count:]
    return X_train, y_train, X_test, y_test

# def whiten_data(X):
#     means = np.mean(X, axis = 0)
#     stds = np.std(X, axis = 0)
#     X = (X - means) / stds
#     return X, means, stds
def whiten_data(X):
    means = np.mean(X, axis = 0)
    #stds = np.std(X, axis = 0)
    cov_mat = np.cov(X.T)
    X = (X - means)
    X = np.linalg.solve(sp_la.sqrtm(cov_mat), X.T).T
    return X, means, cov_mat

def main_axes(X):
    whiteX, _, cov_mat = whiten_data(X)
    unit_whiteX = np.divide(whiteX, np.linalg.norm(whiteX, axis = 1)[:, None])
    d, U = np.linalg.eig(np.cov(unit_whiteX.T))
    U = U[:, np.argsort(-d)] # sort in descending order
    d = -np.sort(-d)
    unit_main_axis = U[:, 0]
    return np.dot(sp_la.sqrtm(cov_mat), unit_main_axis)

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

def kernel_error(data_name, X_train, noise_var, scale):
    feature_gen_handles = [lambda a, s:                         kernels.iid_gaussian_RFF(a, n_rff, s, scale), \
                           lambda a, s: kernels.make_antithetic(kernels.iid_gaussian_RFF(a, n_rff, s, scale)), \
                           lambda a, s:                         kernels.ort_gaussian_RFF(a, n_rff, s, scale), \
                           lambda a, s: kernels.make_antithetic(kernels.ort_gaussian_RFF(a, n_rff, s, scale)), \
                           lambda a, s:                         kernels.HD_gaussian_RFF( a, n_rff, s, scale, 1), \
                           lambda a, s:                         kernels.HD_gaussian_RFF( a, n_rff, s, scale, 2), \
                           lambda a, s:                         kernels.HD_gaussian_RFF( a, n_rff, s, scale, 3)]
    algo_names = ['iid', 'iid_anti', 'ort', 'ort_anti', 'HD_1', 'HD_2', 'HD_3']
    exact_kernel = kernels.gaussian_kernel(X_train, X_train, scale)
    n_seeds = 10
    n_rffs = [4,8,12,16,24,40,56,72,88,104,128]
    for algo_name, feature_gen_handle in zip(algo_names, feature_gen_handles):
        errors = defaultdict(list)
        for n_rff in n_rffs:
            for seed in range(n_seeds):
                PhiX_train = feature_gen_handle(a, seed)
                sample_gram = np.dot(PhiX_train, PhiX_train.T)
                errors[n_rff].append(np.linalg.norm(PhiX_train - sample_gram))
        with open('%s_%s_krr.pk' % (data_name, algo_name), 'wb') as f:
            pickle.dump(errors, f)

def regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale = 1.0, degree = 2.0, inhom_term = 1.0):
    n_seeds = 10
    algos = {'iid':      lambda a, s:                         kernels.iid_gaussian_RFF(a, n_rff, s, scale), \
            'iid_anti':  lambda a, s: kernels.make_antithetic(kernels.iid_gaussian_RFF(a, n_rff, s, scale)), \
            'ort':       lambda a, s:                         kernels.ort_gaussian_RFF(a, n_rff, s, scale), \
            'ort_anti':  lambda a, s: kernels.make_antithetic(kernels.ort_gaussian_RFF(a, n_rff, s, scale)), \
            'HD_1':      lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 1), \
            'HD_2':      lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 2), \
            'HD_3':      lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 3), \
            'angled_0.5':lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 0.5), \
            'angled_0.75':lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 0.75), \
            'angled_1.0':lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 1.0), \
            'angled_1.25':lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 1.25), \
            'angled_2.1':lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 2.1), \
            'greedy':    lambda a, s:                 kernels.greedy_unif_gaussian_RFF(a, n_rff, s, scale), \
            'greedy_dir':lambda a, s:                  kernels.greedy_dir_gaussian_RFF(a, n_rff, s, scale), \
            'fastfood':  lambda a, s:                             kernels.fastfood_RFF(a, n_rff, s, scale), \
            'iid_polynomial_sp': lambda a, s:   kernels.iid_polynomial_sp_random_features(a, n_rff, s, degree, inhom_term), \
            'iid_exponential_sp': lambda a, s: kernels.iid_exponential_sp_random_features(a, n_rff, s, scale), \
            'approx_antithetic_RFF':   lambda a, s:                    kernels.approx_antithetic_RFF(a, n_rff, s, scale, main_axis(X))}

    test_algos = ['iid_anti', 'ort_anti', 'attempt'] # algos.keys()
    algos = {k: algos[k] for k in test_algos}

    n_rffs = [4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,156]
    n_rffs = [4,8,16,24,40,56,88,104,128,156]
    for algo_name, feature_gen_handle in algos.items():
        errors = defaultdict(list)
        for n_rff in n_rffs:
            for seed in range(100, n_seeds + 100):
                y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda a: feature_gen_handle(a, seed))
                errors[n_rff].append(np.linalg.norm(y_test_fit - y_test) / y_test.shape[0])
            print '{} {} \t{} \t{}'.format(algo_name, n_rff, np.mean(errors[n_rff]), np.sqrt(np.var(errors[n_rff])))
        with open('%s_%s_krr.pk' % (data_name, algo_name), 'wb') as f:
            pickle.dump(errors, f)

def regression_error_kernel(data_name, X_train, y_train, X_test, y_test, noise_var, scale = 1.0, degree = 2.0, inhom_term = 1.0):  
    n_rffs = [4,156]

    errors = {}
    y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    with open('%s_exact_gauss_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)
    
    errors = {}
    y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.exponential_sp_kernel(a, b, scale))
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    with open('%s_exact_exp_sp_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)
    
    errors = {}
    y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.polynomial_sp_kernel(a, b, degree, inhom_term))
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    with open('%s_exact_poly_sp_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)

def plot_regression_errors(data_name, algo_names):
    for algo_name in algo_names:
        with open('%s_%s_krr.pk' % (data_name, algo_name), 'rb') as f:
            data = pickle.load(f)
        x = data.keys()
        x.sort()
        means = np.array([np.mean(data[k]) for k in x])
        std_dev = np.sqrt([np.mean(np.square(data[k] - np.mean(data[k]))) for k in x])
        p = plt.plot(x, means, '.-', label = algo_name)
        plt.fill_between(x, means - 1.96 * std_dev, means + 1.96 * std_dev, color = p[0].get_color(), alpha = 0.05)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    data_name = ['wine', 'airq'][1]
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
    degree = 2
    inhom_term = 1.0
    n_rff = 64
    seed = 0

    # regression_error_kernel(data_name, X_train, y_train, X_test, y_test, noise_var, scale, degree, inhom_term)

    regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale, degree, inhom_term)
    # plot_regression_errors(data_name, ['exact_gauss', 'iid', 'ort', 'angled_0.5', 'angled_0.75', 'angled_1.25', 'angled_1.0', 'angled_2.1'])
    # plot_regression_errors(data_name, ['exact_gauss', 'iid', 'iid_anti', 'ort', 'ort_anti', 'HD_1', 'HD_2', 'HD_3', 'fastfood'])
    plot_regression_errors(data_name, ['exact_gauss', 'iid', 'approx_antithetic_RFF'])
    # plot_regression_errors(data_name, ['exact_gauss', 'iid', 'ort', 'HD_1', 'HD_3', 'fastfood'])
    # plot_regression_errors(data_name, ['exact_gauss', 'iid', 'exact_exp_sp', 'exact_poly_sp', 'iid_exponential_sp', 'iid_polynomial_sp'])
    # plot_regression_errors(data_name, ['exact_gauss', 'iid', 'greedy', 'greedy_dir'])

    # # Fit a GP with random iid features 
    # print 'Start'
    # y_cv_gp_feature, _ = gpr.posterior_from_feature_gen(X_train, y_train, X_cv, noise_var, 
    #                                                     lambda a: kernels.ort_gaussian_RFF(a, n_rff, seed, scale))
    # print 'Done'
    # print np.linalg.norm(y_cv - y_cv_gp_feature) / y_cv.shape[0]
    
    # # Fit a GP with kernel 
    # print 'Start'
    # y_cv_gp_kernel, _ = gpr.posterior_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    # print 'Done'
    # print np.linalg.norm(y_cv - y_cv_gp_kernel) / y_cv.shape[0]
    # print np.linalg.norm(y_cv_gp_kernel - y_cv_gp_feature) / y_cv.shape[0]

    # # Fit with kernel trick
    # print 'Start'
    # y_cv_fit_kernel = krr.fit_from_kernel_gen(X_train, y_train, X_cv, noise_var, lambda a, b: kernels.exponential_kernel(a, b, scale))
    # print 'Done'
    # print np.linalg.norm(y_cv - y_cv_fit_kernel) / y_cv.shape[0]

    y_cv_fit_feature = krr.fit_from_feature_gen(X_train, y_train, X_cv, noise_var, 
                                                lambda a: kernels.angled_gaussian_neighbour_RFF(a, n_rff, seed, scale, 1.3))
    y_cv_fit_feature = np.real(y_cv_fit_feature)
    print np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]

    # y_cv_fit_feature = krr.fit_from_feature_gen(X_train, y_train, X_cv, noise_var, 
    #                                             lambda a: kernels.fastfood_RFF(a, n_rff, seed, scale))
    # y_cv_fit_feature = np.real(y_cv_fit_feature)
    # print np.linalg.norm(y_cv - y_cv_fit_feature) / y_cv.shape[0]
