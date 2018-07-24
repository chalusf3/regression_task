import numpy as np
import scipy.linalg as sp_la
import scipy.special as sp_sp
import matplotlib.pyplot as plt
import matplotlib.ticker

import csv, timeit, pickle, random, pdb, time, numbers
import krr, gpr, kernels

from datetime import timedelta, datetime
from collections import defaultdict

def load_MSD():
    X = []
    y = []

    with open('datasets/YearPredictionMSD.txt', 'rb') as f:
        reader = csv.reader(f, delimiter = ',', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            X.append(row[1:])
            y.append(row[0])
    X = np.matrix(X)
    y = np.matrix(y).T
    return X, y

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

    train_count = int(ratio * X.shape[0])
    X_train = X[:train_count]
    y_train = y[:train_count]
    X_test = X[train_count:]
    y_test = y[train_count:]
    return X_train, y_train, X_test, y_test

def whiten_data(X):
    means = np.mean(X, axis = 0)
    # stds = np.std(X, axis = 0)
    cov_mat = np.cov(X.T)
    X = (X - means)
    if np.linalg.matrix_rank(cov_mat) == cov_mat.shape[0]:
        X = np.linalg.solve(sp_la.sqrtm(cov_mat), X.T).T
    else:
        print('/!\\ whitening: covariance matrix is singular')
        X /= np.diag(cov_mat)
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

""" # not needed anymore, see exp_pw_err.py for a similar experiment
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
"""

def algos_generator(keys, scale = 1.0, degree = 2.0, inhom_term = 1.0):
    algos = {}
    # RBF kernels
    algos['iid'] =                   lambda raw_feature, n_rff, seed:                         kernels.iid_gaussian_RFF(raw_feature, n_rff, seed, scale)
    algos['ort'] =                   lambda raw_feature, n_rff, seed:                         kernels.ort_gaussian_RFF(raw_feature, n_rff, seed, scale, weighted = False)
    algos['ort_weighted'] =          lambda raw_feature, n_rff, seed:                         kernels.ort_gaussian_RFF(raw_feature, n_rff, seed, scale, weighted = True)
    algos['iid_fix_norm'] =          lambda raw_feature, n_rff, seed:                         kernels.iid_fix_norm_RFF(raw_feature, n_rff, seed, scale)
    algos['ort_fix_norm'] =          lambda raw_feature, n_rff, seed:                         kernels.ort_fix_norm_RFF(raw_feature, n_rff, seed, scale, weighted = False)
    algos['ort_fix_norm_weighted'] = lambda raw_feature, n_rff, seed:                         kernels.ort_fix_norm_RFF(raw_feature, n_rff, seed, scale, weighted = True)
    algos['ort_ss_all'] =            lambda raw_feature, n_rff, seed:                         kernels.ort_gaussian_RFF(raw_feature, n_rff, seed, scale, weighted = False, subsample_all = True)
    algos['iid_anti'] =              lambda raw_feature, n_rff, seed: kernels.make_antithetic(kernels.iid_gaussian_RFF(raw_feature, n_rff / 2, seed, scale))
    algos['ort_anti'] =              lambda raw_feature, n_rff, seed: kernels.make_antithetic(kernels.ort_gaussian_RFF(raw_feature, n_rff / 2, seed, scale, weighted = False))
    algos['HD_1'] =                  lambda raw_feature, n_rff, seed:                          kernels.HD_gaussian_RFF(raw_feature, n_rff, seed, scale, 1)
    algos['HD_2'] =                  lambda raw_feature, n_rff, seed:                          kernels.HD_gaussian_RFF(raw_feature, n_rff, seed, scale, 2)
    algos['HD_3'] =                  lambda raw_feature, n_rff, seed:                          kernels.HD_gaussian_RFF(raw_feature, n_rff, seed, scale, 3)
    algos['HD_1_fix_norm'] =         lambda raw_feature, n_rff, seed:                          kernels.HD_fix_norm_RFF(raw_feature, n_rff, seed, scale, 1)
    algos['HD_2_fix_norm'] =         lambda raw_feature, n_rff, seed:                          kernels.HD_fix_norm_RFF(raw_feature, n_rff, seed, scale, 2)
    algos['HD_3_fix_norm'] =         lambda raw_feature, n_rff, seed:                          kernels.HD_fix_norm_RFF(raw_feature, n_rff, seed, scale, 3)
    algos['angled_0.5'] =            lambda raw_feature, n_rff, seed:            kernels.angled_gaussian_neighbour_RFF(raw_feature, n_rff, seed, scale, 0.5)
    algos['angled_0.75'] =           lambda raw_feature, n_rff, seed:            kernels.angled_gaussian_neighbour_RFF(raw_feature, n_rff, seed, scale, 0.75)
    algos['angled_1.0'] =            lambda raw_feature, n_rff, seed:            kernels.angled_gaussian_neighbour_RFF(raw_feature, n_rff, seed, scale, 1.0)
    algos['angled_1.25'] =           lambda raw_feature, n_rff, seed:            kernels.angled_gaussian_neighbour_RFF(raw_feature, n_rff, seed, scale, 1.25)
    algos['angled_2.1'] =            lambda raw_feature, n_rff, seed:            kernels.angled_gaussian_neighbour_RFF(raw_feature, n_rff, seed, scale, 2.1)
    algos['greedy'] =                lambda raw_feature, n_rff, seed:                 kernels.greedy_unif_gaussian_RFF(raw_feature, n_rff, seed, scale)
    algos['greedy_dir'] =            lambda raw_feature, n_rff, seed:                  kernels.greedy_dir_gaussian_RFF(raw_feature, n_rff, seed, scale)
    algos['fastfood'] =              lambda raw_feature, n_rff, seed:                             kernels.fastfood_RFF(raw_feature, n_rff, seed, scale)

    # Dot product kernels
    algos['iid_polyn'] =             lambda raw_feature, n_rff, seed:        kernels.iid_polynomial_sp_random_features(raw_feature, n_rff, seed, degree, inhom_term)
    algos['ort_polyn'] =             lambda raw_feature, n_rff, seed:   kernels.ort_polynomial_sp_random_unit_features(raw_feature, n_rff, seed, degree, inhom_term)
    algos['HD_polyn'] =              lambda raw_feature, n_rff, seed:    kernels.HD_polynomial_sp_random_unit_features(raw_feature, n_rff, seed, degree, inhom_term)
    algos['iid_polynomial_sp'] =     lambda raw_feature, n_rff, seed:        kernels.iid_polynomial_sp_random_features(raw_feature, n_rff, seed, degree, inhom_term)
    algos['iid_exponential_sp'] =    lambda raw_feature, n_rff, seed:       kernels.iid_exponential_sp_random_features(raw_feature, n_rff, seed, scale)

    algos = {k: algos[k] for k in keys}
    return algos

def regression_error_n_rff(data_name, algos, X_train, y_train, X_test, y_test, noise_var):
    timing = False
    if timing:
        n_seeds = 1000
    else:    
        n_seeds = 100

    if data_name == 'airq':
        n_rffs = range(4,24 + 1,2) # for squared exponential kernels
    elif data_name == 'wine':
        n_rffs = range(4,32 + 1,2) # for squared exponential kernels
    # n_rffs = [4,8,16,24,40,56,88,104,128,156,188,220,256,320,384,448,512,640] # np.power(2, np.arange(2, 11)) # for polynomial kernels
    for algo_name, feature_gen_handle in algos.items():
        errors = defaultdict(list)
        errors['runtimes'] = defaultdict(list)
        for n_rff in n_rffs:
            errors[n_rff] = np.zeros(n_seeds)
            start_time = time.clock()
            for seed in range(n_seeds):
                y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda raw_feature: feature_gen_handle(raw_feature, n_rff, seed))
                # errors[n_rff].append(np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0])
                errors[n_rff][seed] = np.mean(np.abs(y_test - y_test_fit))
            errors['runtimes'][n_rff] = (time.clock() - start_time) / n_seeds
            print '{} {} \t{} \t{:.4}sec'.format(algo_name, n_rff, np.mean(errors[n_rff]), errors['runtimes'][n_rff])

        if timing:
            filename = 'output/timing/%s_%s_krr.pk' % (data_name, algo_name)
        else:
            filename = 'output/%s_%s_krr.pk' % (data_name, algo_name)

        try:
            with open(filename, 'rb') as f:
                old_errors = pickle.load(f)
                if 'runtimes' in old_errors.keys() and not timing:
                    errors['runtimes'] = old_errors['runtimes']
        except IOError:
            print '%s file did not previously exist' % filename
        except EOFError:
            print '%s was not a pickle file' % filename
            
        with open(filename, 'wb') as f:
            pickle.dump(errors, f)
            
    return algos.keys()

def print_average_regression_error(data_name, algos, X_train, y_train, X_test, y_test, noise_var):
    # Kernel performance (from pickle archive)    
    with open('output/%s_exact_gauss_krr.pk' % data_name, 'rb') as f:
        data = pickle.load(f)
    n_rff = [key for key in data.keys() if isinstance(key, numbers.Number) ][0]
    print data[n_rff]
    print '{0} & {1:.3} & [{1:.3}, {1:.3}] & {2:.5} \\\\'.format('exact SE kernel'.ljust(24), data[n_rff][0], data['runtimes'][n_rff])

    if data_name == 'airq':
        n_rff = 12
    elif data_name == 'wine':
        n_rff = 12

    n_seeds = 1000

    for algo_name, feature_gen_handle in algos.items():
        errors = np.zeros(n_seeds)
        start_time = time.clock()
        for seed in range(n_seeds):
            y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda raw_feature: feature_gen_handle(raw_feature, n_rff, seed))
            errors[seed] = np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]
        runtime = (time.clock() - start_time) / n_seeds
        del y_test_fit
        # print '{} n_rff = {} mean = {:.3f} 5% CI = [{:.3}, {:.3}]\t{:.4}sec'.format(algo_name.replace('_', ' ').ljust(24), n_rff, np.mean(errors[n_rff]), np.percentile(errors, 2.5), np.percentile(errors, 97.5), runtime)
        print '{} & {:.3} & [{:.3}, {:.3}] & {:.5} \\\\'.format(algo_name.replace('_', ' ').replace('fix', 'fixed').ljust(24), np.mean(errors[n_rff]), np.percentile(errors, 2.5), np.percentile(errors, 97.5), runtime)

def regression_error_kernel(data_name, X_train, y_train, X_test, y_test, noise_var, scale = 1.0, degree = 2.0, inhom_term = 1.0):  
    if data_name == 'airq':
        n_rffs = [4,24] # for squared exponential kernels
    elif data_name == 'wine':
        n_rffs = [4,32] # for squared exponential kernels
    errors = {}
    errors['runtimes'] = {}
    n_trials = 10
    start_time = time.clock()
    for _ in range(n_trials):
        y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print '{} \t{} \t{:.4}sec'.format('SE kernel', errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    if n_trials > 1:
        filename = 'output/timing/%s_exact_gauss_krr.pk' % data_name
    else:
        filename = 'output/%s_exact_gauss_krr.pk' % data_name

    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)
    
    n_rffs = [4,2048]
    errors = {}
    errors['runtimes'] = {}
    n_trials = 10
    start_time = time.clock()
    for _ in range(n_trials):
        y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.exponential_sp_kernel(a, b, scale))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print '{} \t{} \t{:.4}sec'.format('exponential scalar product kernel', errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    if n_trials > 1:
        filename = 'output/timing/%s_exact_exp_sp_krr.pk' % data_name
    else:
        filename = 'output/%s_exact_exp_sp_krr.pk' % data_name
    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)
    
    errors = {}
    errors['runtimes'] = {}
    n_trials = 10
    start_time = time.clock()
    for _ in range(n_trials):
        y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.polynomial_sp_kernel(a, b, degree, inhom_term))
    errors['runtimes'][n_rffs[0]] = (time.clock() - start_time) / n_trials
    errors['runtimes'][n_rffs[-1]] = errors['runtimes'][n_rffs[0]]
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    print '{} \t{} \t{:.4}sec'.format('polynomial scalar product kernel', errors[n_rffs[0]], errors['runtimes'][n_rffs[0]])
    if n_trials > 1:
        'output/timing/%s_exact_poly_sp_krr.pk' % data_name
    else:
        'output/%s_exact_poly_sp_krr.pk' % data_name
    with open(filename, 'wb+') as f:
        pickle.dump(errors, f)
    
def plot_regression_errors(data_name, algo_names, filename = 'regression'):
    plt.figure(figsize = (6,4))
    ylim_ticks = [1,0]
    for algo_name in algo_names:
        with open('output/%s_%s_krr.pk' % (data_name, algo_name), 'rb+') as f:
            data = pickle.load(f)
        x = filter(lambda k: isinstance(k, numbers.Number), data.keys())
        x.sort()
        means = np.array([np.mean(data[k]) for k in x])
        ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        ylim_ticks[1] = max(ylim_ticks[1], np.max(means))
        low_perc = np.array([np.percentile(data[k], 2.5) for k in x])
        high_perc = np.array([np.percentile(data[k], 97.5) for k in x])
        p = plt.semilogy(x, means, '.-', label = algo_name.replace('_', ' '), linewidth = 1)
        # plt.fill_between(x, low_perc, high_perc, color = p[0].get_color(), alpha = 0.05)
    
    plt.xlabel(r'\# random features')
    
    plt.ylabel(r'Average regression error')
    # plt.yscale('log')

    if data_name == 'wine':
        yticks_spacing = 2.5e-2 # space between y ticks
    elif data_name == 'airq':
        yticks_spacing = 5e-2 # space between y ticks

    yticks_lim_integer = (1 + int(ylim_ticks[0] / yticks_spacing), int(ylim_ticks[1] / yticks_spacing)) # floor and ceil
    plt.minorticks_off()
    plt.yticks(yticks_spacing * np.arange(yticks_lim_integer[0], 1 + yticks_lim_integer[1]))
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    xticks_lim = [(int(min(plt.xticks()[0]) / 2) + 1) * 2, int(max(plt.xticks()[0]) / 2) * 2]
    plt.xticks(range(xticks_lim[0], xticks_lim[1]+1, 2))

    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_%s.eps' % (data_name, filename))
    # plt.show()
    plt.clf()

def plot_runtimes(data_name, algo_names, filename = 'regression'):
    plt.figure(figsize = (8,6))
    # ylim_ticks = [1,0]
    for algo_name in algo_names:
        with open('output/timing/%s_%s_krr.pk' % (data_name, algo_name), 'rb') as f:
            data = pickle.load(f)['runtimes']
        n_rffs = data.keys()
        n_rffs.sort()
        runtimes = np.array([data[n_rff] for n_rff in n_rffs])
        # ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        # ylim_ticks[1] = max(ylim_ticks[0], np.max(means))
        plt.plot(n_rffs, runtimes, '.-', label = algo_name.replace('_', ' ') )
        
    plt.xlabel(r'\# random features')
    plt.ylabel(r'Average runtime [s]')
    
    xticks_lim = ((int(min(plt.xticks()[0]) / 2) + 1) * 2, int(max(plt.xticks()[0]) / 2) * 2)
    plt.xticks(range(xticks_lim[0], xticks_lim[1]+1, 2))
    
    plt.ylim(0)

    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_%s_%s.eps' % (data_name, filename))
    # plt.show()
    plt.clf()

def plot_efficiency(data_name, X_train, y_train, X_test, y_test, noise_var, algo_handle, algoname = 'regression'):
    n_rffs = [100]
    for n_rff in n_rffs:
        PhiX_train = algo_handle(X_train, n_rff)
        PhiX_test = algo_handle(X_test, n_rff)
        beta = krr.lin_reg(PhiX_train, y_train, noise_var)
        plt.hist(beta, 50)
        plt.show()

def dependence_n_datapoints_kernel(data_name, X, y, noise_var, scale):
    n_seeds = 10
    n_divisions = 10
    division_ratios = np.linspace(0.1, 1.0, n_divisions)
    # division_ratios += [0.15, 0.25]
    X_train_all, y_train_all, X_test, y_test = split_data(X, y, ratio = 0.8)
    errors = {k: 0.0 for k in division_ratios}
    errors['runtimes'] = {}
    for division_ratio in division_ratios:
        start_time = time.clock()
        for seed in range(n_seeds):
            random.seed(seed)
            X_train, y_train, _, _ = split_data(X_train_all, y_train_all, ratio = division_ratio)
            y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.gaussian_kernel(a, b, scale))
            errors[division_ratio] += np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]
        errors['runtimes'][division_ratio] = (time.clock() - start_time) / n_seeds
        errors[division_ratio] /= n_seeds
        print division_ratio, errors[division_ratio]
    with open('output_dep/%s_exact_gauss_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)

def dependence_n_datapoints_rff(data_name, X, y, noise_var, algos):
    dim = np.prod(X[0].shape)
    if data_name != 'MSD':
        n_rffs = dim / 2 * np.arange(1, 7)
        n_rffs = n_rffs[1::2]
        n_seeds = 5000
    else:
        n_seeds = 10
        n_rffs = np.arange(1000, 0, -500)
    print dim, n_rffs
    n_divisions = 10
    division_ratios = np.linspace(0.1, 1.0, n_divisions)

    X_train_all, y_train_all, X_test, y_test = split_data(X, y, ratio = 0.8)

    for algo_name, feature_gen_handle in algos.items():
        errors = {}
        errors['runtimes'] = {}
        for n_rff in n_rffs:
            errors[n_rff] = {k: 0.0 for k in division_ratios}
            errors['runtimes'][n_rff] = {}
            for division_ratio in division_ratios:
                start_time = time.clock()
                for seed in range(n_seeds):
                    random.seed(seed)
                    np.random.seed(seed)
                    X_train, y_train, _, _ = split_data(X_train_all, y_train_all, ratio = division_ratio)

                    y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda raw_feature: feature_gen_handle(raw_feature, n_rff, seed))
                    errors[n_rff][division_ratio] += np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]
                errors['runtimes'][n_rff][division_ratio] = (time.clock() - start_time) / n_seeds
                errors[n_rff][division_ratio] /= n_seeds
            print algo_name, n_rff, [errors[n_rff][dr] for dr in division_ratios], [errors['runtimes'][n_rff][dr] for dr in division_ratios]
            with open('output_dep/%s_%s_krr.pk' % (data_name, algo_name), 'wb') as f:
                pickle.dump(errors, f)

def plot_dependence_n_datapoints(data_name, algo_names):
    plt.figure(figsize = (6,4))
    if data_name != 'MSD':
        # plot the kernel's curve
        with open('output_dep/%s_exact_gauss_krr.pk' % data_name) as f:
            data = pickle.load(f)
        division_ratios = sorted(filter(lambda k: isinstance(k, numbers.Number), data.keys()))
        plt.plot(division_ratios, [data[dr] for dr in division_ratios], '*-', linewidth = 1, label = 'exact kernel')

    # plot the RFF algos curves
    color_dict = {}
    for algo_name, marker in zip(algo_names, matplotlib.markers.MarkerStyle.filled_markers[0:len(algo_names)]):
        with open('output_dep/%s_%s_krr.pk' % (data_name, algo_name)) as f:
            data = pickle.load(f)
        n_rffs = sorted(filter(lambda k: isinstance(k, numbers.Number), data.keys()))

        for n_rff in n_rffs:
            division_ratios = sorted(filter(lambda k: isinstance(k, numbers.Number), data[n_rff].keys()))
            if n_rff in color_dict.keys():
                plt.plot(    division_ratios, [data[n_rff][dr] for dr in division_ratios], marker = marker, markersize = 4, linewidth = 1, label = r'%s, \# RFF = %d' % (algo_name.replace('_', ' '), n_rff), color = color_dict[n_rff]) 
            else:
                p = plt.plot(division_ratios, [data[n_rff][dr] for dr in division_ratios], marker = marker, markersize = 4, linewidth = 1, label = r'%s, \# RFF = %d' % (algo_name.replace('_', ' '), n_rff))
                color_dict[n_rff] = p[0].get_color()

    plt.xlim([0, 1])
    plt.xlabel('Fraction of data used for training')
    plt.ylabel('Regression error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_dep_n_datapoints.eps' % data_name)
    # plt.show()

    # plot runtimes
    plt.figure(figsize = (6, 4))
    for algo_name, marker in zip(algo_names, matplotlib.markers.MarkerStyle.filled_markers[0:len(algo_names)]):
        with open('output_dep/%s_%s_krr.pk' % (data_name, algo_name)) as f:
            data = pickle.load(f)['runtimes']
        n_rffs = sorted(data.keys())
        for n_rff in n_rffs:#[1::2]:
            division_ratios = sorted(data[n_rff].keys())
            if n_rff in color_dict.keys():
                plt.plot(division_ratios, [data[n_rff][dr] for dr in division_ratios], marker = marker, markersize = 4, linewidth = 1, label = r'%s, \# RFF = %d' % (algo_name.replace('_', ' '), n_rff), color = color_dict[n_rff]) 
            else:
                print('Something wrong with the color rotation scheme in dependence datapoints runtime plotting')
    if data_name != 'MSD':
        with open('output_dep/%s_exact_gauss_krr.pk' % data_name) as f:
            data = pickle.load(f)['runtimes']
        division_ratios = sorted(data.keys())
        plt.plot(division_ratios, [data[dr] for dr in division_ratios], marker = marker, markersize = 4, linewidth = 1, label = r'exact SE kernel')
    plt.xlabel('Fraction of training data used for training')
    plt.ylabel('Runtime [s]')
    
    asymptotics_x = np.linspace(min(division_ratios), max(division_ratios))
    plt.plot(asymptotics_x, 0.2e4*np.power(asymptotics_x, 1), '.', linewidth = 1, markersize = 1, label = r'$O(x)$')
    # plt.plot(asymptotics_x, 1e3*np.power(asymptotics_x, 2), '.', linewidth = 1, markersize = 1, label = r'$O(x^2)$')

    # plt.ylim(0)
    plt.xlim([0.1,1])
    plt.yscale('log')
    # plt.xscale('log')
    if data_name != 'MSD':
        plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))
    else:
        plt.legend()
    plt.tight_layout()
    plt.savefig('%s_dep_n_datapoints_runtime.eps' % data_name)
    plt.show()

    """
    color_dict = {}
    plt.figure(figsize = (6, 4))
    for algo_name, marker in zip(algo_names, matplotlib.markers.MarkerStyle.filled_markers[0:len(algo_names)]):
        with open('output_dep/%s_%s_krr.pk' % (data_name, algo_name)) as f:
            data = pickle.load(f)['runtimes']
        n_rffs = sorted(data.keys())
        division_ratios = sorted(data[n_rff].keys())
        for dr in filter(lambda dr: np.allclose(np.round(dr / 0.2), dr / 0.2), division_ratios):
            if dr in color_dict.keys():
                plt.plot(n_rffs, [data[n_rff][dr] for n_rff in n_rffs], marker = marker, markersize = 4, linewidth = 1, label = r'%s, fraction data = %.3f' % (algo_name.replace('_', ' '), dr), color = color_dict[dr])
            else:
                p = plt.plot(n_rffs, [data[n_rff][dr] for n_rff in n_rffs], marker = marker, markersize = 4, linewidth = 1, label = r'%s, fraction data = %.3f' % (algo_name.replace('_', ' '), dr))
                color_dict[dr] = p[0].get_color()
    plt.xlabel(r'\# RFF')
    plt.ylabel('Runtime [s]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_dep_n_datapoints_runtime_rff.eps' % data_name)
    plt.show()
    """

def plotting_error(X_train, y_train, X_test, y_test, data_name, noise_var, scale, degree, inhom_term):
    keys = ['iid','iid_fix_norm','ort','ort_fix_norm','ort_weighted','ort_fix_norm_weighted','ort_ss_all','HD_1','HD_2','HD_3','HD_1_fix_norm','HD_2_fix_norm','HD_3_fix_norm']
    algos = algos_generator(keys, scale = scale, degree = degree, inhom_term = inhom_term)

    # regression_error_kernel(data_name, X_train, y_train, X_test, y_test, noise_var, scale, degree, inhom_term)
    # regression_error_n_rff(        data_name, algos, X_train, y_train, X_test, y_test, noise_var)
    print_average_regression_error(data_name, algos, X_train, y_train, X_test, y_test, noise_var)

    # keys = ['iid','iid_fix_norm','ort','ort_fix_norm','ort_weighted','ort_fix_norm_weighted','ort_ss_all','HD_1','HD_2','HD_3','HD_1_fix_norm','HD_2_fix_norm','HD_3_fix_norm']
    # plot_regression_errors(data_name, ['exact_gauss'] + keys)
    # plot_runtimes(data_name, keys)
    
    # keys = ['iid','ort','iid_fix_norm','ort_fix_norm','ort_ss_all','ort_weighted','ort_fix_norm_weighted']
    # plot_regression_errors(data_name, ['exact_gauss'] + keys, filename = 'iid_ort')
    # plot_runtimes(data_name, keys, filename = 'iid_ort')
    
    # keys = ['ort', 'ort_fix_norm','HD_1','HD_2','HD_3','HD_1_fix_norm','HD_2_fix_norm','HD_3_fix_norm']
    # plot_regression_errors(data_name, ['exact_gauss'] + keys, filename = 'HD')
    # plot_runtimes(data_name, keys, filename = 'HD')

def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    np.random.seed(0)
    data_name = ['wine', 'airq', 'MSD'][2]
    if data_name == 'wine':
        X, y = load_wine_dataset()
    elif data_name == 'airq':
        X, y = load_air_quality_dataset()
    elif data_name == 'MSD':
        X, y = load_MSD()
    
    print data_name, len(y), [np.percentile(y, [0, 2.5, 50, 97.5, 100])]
    X = whiten_data(X)[0]
    
    noise_var = 1.0
    if data_name == 'wine' or data_name == 'airq':
        scale = 16.0
        degree = 3
        inhom_term = 1.0
    elif data_name == 'MSD':
        scale = 300.0
        degree = 3
        inhom_term = 1.0

    algos = algos_generator(['iid', 'ort_weighted'], scale = scale, degree = degree, inhom_term = inhom_term)
    # dependence_n_datapoints_kernel(data_name, X, y, noise_var, scale)
    # dependence_n_datapoints_rff(data_name, X, y, noise_var, algos)
    plot_dependence_n_datapoints(data_name, algos.keys())

    if data_name == 'wine' or data_name == 'airq':
        X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
    elif data_name == 'MSD':
        X_test = X[-51630:]
        y_test = y[-51630:]
        X_train = X[:-51630]
        y_train = y[:-51630]
    # X_train = 80 % of all data
    # X_test =  20 % of all data
    
    # print('Dimension implicit feature space polynomial kernel = %d' % sp_sp.comb(X_train.shape[1] + degree, degree))
    
    # plotting_error(X_train, y_train, X_test, y_test, data_name, noise_var, scale, degree, inhom_term)

    # plot_efficiency(data_name, X_train, y_train, X_test, y_test, noise_var, lambda x, n_rff: kernels.ort_gaussian_RFF(x, n_rff, 0, scale), algoname = 'iid')

    # y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda a: kernels.iid_gaussian_RFF(a, 1024, 0, scale))
    # print noise_var, scale, np.mean(np.abs(y_test_fit - y_test))
    # y_fit = krr.fit_from_feature_gen(X_train, y_train, X, noise_var, lambda a: kernels.iid_gaussian_RFF(a, 1024, 0, scale))
    # print noise_var, scale, np.mean(np.abs(y_fit - y))

if __name__ == '__main__':
    main()
  