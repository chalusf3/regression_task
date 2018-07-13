import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import csv, timeit, pickle
import krr, gpr, kernels
from datetime import timedelta, datetime
from collections import defaultdict
import scipy.linalg as sp_la
import scipy.special as sp_sp
import time

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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

def whiten_data(X):
    means = np.mean(X, axis = 0)
    # stds = np.std(X, axis = 0)
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

def regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale = 1.0, degree = 2.0, inhom_term = 1.0):
    n_seeds = 50 # 10000
    algos = {}
    algos['iid'] =                   lambda a, s:                         kernels.iid_gaussian_RFF(a, n_rff, s, scale)
    algos['ort'] =                   lambda a, s:                         kernels.ort_gaussian_RFF(a, n_rff, s, scale)
    algos['iid_fix_norm'] =          lambda a, s:                         kernels.iid_fix_norm_RFF(a, n_rff, s, scale)
    algos['ort_fix_norm'] =          lambda a, s:                         kernels.ort_fix_norm_RFF(a, n_rff, s, scale)
    algos['ort_ss_last'] =           lambda a, s:                         kernels.ort_gaussian_RFF(a, n_rff, s, scale, subsample_all = False)
    # algos['iid_anti'] =              lambda a, s: kernels.make_antithetic(kernels.iid_gaussian_RFF(a, n_rff / 2, s, scale))
    # algos['ort_anti'] =              lambda a, s: kernels.make_antithetic(kernels.ort_gaussian_RFF(a, n_rff / 2, s, scale))
    algos['HD_1'] =                  lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 1)
    algos['HD_2'] =                  lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 2)
    algos['HD_3'] =                  lambda a, s:                          kernels.HD_gaussian_RFF(a, n_rff, s, scale, 3)
    algos['HD_1_fix_norm'] =         lambda a, s:                          kernels.HD_fix_norm_RFF(a, n_rff, s, scale, 1)
    algos['HD_2_fix_norm'] =         lambda a, s:                          kernels.HD_fix_norm_RFF(a, n_rff, s, scale, 2)
    algos['HD_3_fix_norm'] =         lambda a, s:                          kernels.HD_fix_norm_RFF(a, n_rff, s, scale, 3)
    # algos['angled_0.5'] =            lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 0.5)
    # algos['angled_0.75'] =           lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 0.75)
    # algos['angled_1.0'] =            lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 1.0)
    # algos['angled_1.25'] =           lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 1.25)
    # algos['angled_2.1'] =            lambda a, s:            kernels.angled_gaussian_neighbour_RFF(a, n_rff, s, scale, 2.1)
    # algos['greedy'] =                lambda a, s:                 kernels.greedy_unif_gaussian_RFF(a, n_rff, s, scale)
    # algos['greedy_dir'] =            lambda a, s:                  kernels.greedy_dir_gaussian_RFF(a, n_rff, s, scale)
    # algos['fastfood'] =              lambda a, s:                             kernels.fastfood_RFF(a, n_rff, s, scale)

    # algos['iid_polyn'] =             lambda a, s:        kernels.iid_polynomial_sp_random_features(a, n_rff, s, degree, inhom_term)
    # algos['ort_polyn'] =             lambda a, s:   kernels.ort_polynomial_sp_random_unit_features(a, n_rff, s, degree, inhom_term)
    # algos['HD_polyn'] =              lambda a, s:    kernels.HD_polynomial_sp_random_unit_features(a, n_rff, s, degree, inhom_term)
    # algos['iid_polynomial_sp'] =     lambda a, s:        kernels.iid_polynomial_sp_random_features(a, n_rff, s, degree, inhom_term)
    # algos['iid_exponential_sp'] =    lambda a, s:       kernels.iid_exponential_sp_random_features(a, n_rff, s, scale)

    test_algos = algos.keys()
    algos = {k: algos[k] for k in test_algos}

    if data_name == 'airq':
        n_rffs = range(4,24 + 1,2) # for squared exponential kernels
    elif data_name == 'wine':
        n_rffs = range(4,32 + 1,2) # for squared exponential kernels
    # n_rffs = [4,8,16,24,40,56,88,104,128,156,188,220,256,320,384,448,512,640] # np.power(2, np.arange(2, 11)) # for polynomial kernels
    for algo_name, feature_gen_handle in algos.items():
        errors = defaultdict(list)
        errors['runtimes'] = defaultdict(list)
        for n_rff in n_rffs:
            start_time = time.clock()
            for seed in range(n_seeds):
                y_test_fit = krr.fit_from_feature_gen(X_train, y_train, X_test, noise_var, lambda a: feature_gen_handle(a, seed))
                errors[n_rff].append(np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0])
            errors['runtimes'][n_rff] = (time.clock() - start_time) / n_seeds
            print '{} {} \t{} \t{:.4}sec'.format(algo_name, n_rff, np.mean(errors[n_rff]), errors['runtimes'][n_rff])
        with open('output/%s_%s_krr.pk' % (data_name, algo_name), 'wb') as f:
            pickle.dump(errors, f)
    return algos.keys()

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
    with open('output/%s_exact_gauss_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)
    
    n_rffs = [4,2048]
    errors = {}
    y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.exponential_sp_kernel(a, b, scale))
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    with open('output/%s_exact_exp_sp_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)
    
    errors = {}
    y_test_fit = krr.fit_from_kernel_gen(X_train, y_train, X_test, noise_var, lambda a, b: kernels.polynomial_sp_kernel(a, b, degree, inhom_term))
    errors[n_rffs[0]] = [np.linalg.norm(y_test_fit - y_test, ord = 1) / y_test.shape[0]]
    errors[n_rffs[-1]] = errors[n_rffs[0]]
    with open('output/%s_exact_poly_sp_krr.pk' % data_name, 'wb') as f:
        pickle.dump(errors, f)
    
def plot_regression_errors(data_name, algo_names, filename = 'regression'):
    plt.figure(figsize = (8,6))
    ylim_ticks = [1,0]
    for algo_name in algo_names:
        with open('output/%s_%s_krr.pk' % (data_name, algo_name), 'rb') as f:
            data = pickle.load(f)
        x = filter(lambda k: isinstance(k, (int, long)), data.keys())
        x.sort()
        means = np.array([np.mean(data[k]) for k in x])
        ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        ylim_ticks[1] = max(ylim_ticks[1], np.max(means))
        low_perc = np.array([np.percentile(data[k], 2.5) for k in x])
        high_perc = np.array([np.percentile(data[k], 97.5) for k in x])
        p = plt.semilogy(x, means, '.-', label = algo_name.replace('_', '\_') )
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
        with open('output/%s_%s_krr.pk' % (data_name, algo_name), 'rb') as f:
            data = pickle.load(f)['runtimes']
        n_rffs = data.keys()
        n_rffs.sort()
        runtimes = np.array([data[n_rff] for n_rff in n_rffs])
        # ylim_ticks[0] = min(ylim_ticks[0], np.min(means))
        # ylim_ticks[1] = max(ylim_ticks[0], np.max(means))
        plt.plot(n_rffs, runtimes, '.-', label = algo_name.replace('_', '\_') )
        
    plt.xlabel(r'\# random features')
    plt.ylabel(r'Average runtime')
    
    xticks_lim = ((int(min(plt.xticks()[0]) / 2) + 1) * 2, int(max(plt.xticks()[0]) / 2) * 2)
    plt.xticks(range(xticks_lim[0], xticks_lim[1]+1, 2))
    
    plt.ylim(0)

    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_%s_%s.eps' % (data_name, filename))
    # plt.show()
    plt.clf()

def main():
    np.random.seed(0)
    data_name = ['wine', 'airq'][1]
    if data_name == 'wine':
        X, y = load_wine_dataset()
    elif data_name == 'airq':
        X, y = load_air_quality_dataset()
    
    print data_name, [np.percentile(y, [0, 2.5, 50, 97.5, 100])]

    X = whiten_data(X)[0]
    X_train, y_train, X_test, y_test = split_data(X, y, 0.8) 
    # X_train = 80 % of all data
    # X_test =  20 % of all data
    
    # X_train, y_train, X_cv, y_cv = split_data(X_train, y_train, 0.8)
    # X_train = 64 % of all data
    # X_test =  20 % of all data
    # X_cv =    16 % of all data

    noise_var = 1.0
    scale = 16.0
    degree = 3
    inhom_term = 1.0
    
    print('Dimension implicit feature space polynomial kernel = %d' % sp_sp.comb(X.shape[1] + degree, degree))

    regression_error_kernel(data_name, X_train, y_train, X_test, y_test, noise_var, scale, degree, inhom_term)

    keys = regression_error_n_rff(data_name, X_train, y_train, X_test, y_test, noise_var, scale, degree, inhom_term)
    
    keys = ['iid','ort','iid_fix_norm','ort_fix_norm','ort_ss_last','HD_1','HD_2','HD_3','HD_1_fix_norm','HD_2_fix_norm','HD_3_fix_norm']
    plot_regression_errors(data_name, ['exact_gauss'] + keys)
    plot_runtimes(data_name, keys)
    
    keys = ['iid','ort','iid_fix_norm','ort_fix_norm','ort_ss_last']
    plot_regression_errors(data_name, ['exact_gauss'] + keys, filename = 'iid_ort')
    plot_runtimes(data_name, keys, filename = 'iid_ort')
    keys = ['ort', 'ort_fix_norm','HD_1','HD_2','HD_3','HD_1_fix_norm','HD_2_fix_norm','HD_3_fix_norm']
    plot_regression_errors(data_name, ['exact_gauss'] + keys, filename = 'HD')
    plot_runtimes(data_name, keys, filename = 'HD')

if __name__ == '__main__':
    main()