import numpy as np
import matplotlib.pyplot as plt
import kernels
from scipy.special import comb
from multiprocessing import Pool

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def squared_exponential_kernel():
    dim = 24
    scale = np.sqrt(dim) * 2
    n_seeds = 5000
    max_n_rff = 10 * dim + 1
    algos = {}
    algos['iid'] =           lambda X, n_rff, seed: kernels.iid_gaussian_RFF(X, n_rff, seed, scale)
    algos['iid_fixed_norm'] =  lambda X, n_rff, seed: kernels.iid_fix_norm_RFF(X, n_rff, seed, scale)
    # algos['iid_invnorm'] =   lambda X, n_rff, seed: kernels.iid_invnorm_gaussian_RFF(X, n_rff, seed, scale)
    # algos['ort_ss_all'] =           lambda X, n_rff, seed: kernels.ort_gaussian_RFF(X, n_rff, seed, scale, subsample_all = True, weighted = False)
    algos['ort'] =          lambda X, n_rff, seed: kernels.ort_gaussian_RFF(X, n_rff, seed, scale, subsample_all = False, weighted = False)
    algos['ort_weighted'] = lambda X, n_rff, seed: kernels.ort_gaussian_RFF(X, n_rff, seed, scale, subsample_all = False, weighted = True)
    # algos['ort_fix_norm_ss_all'] =  lambda X, n_rff, seed: kernels.ort_fix_norm_RFF(X, n_rff, seed, scale, subsample_all = True, weighted = False)
    algos['ort_fixed_norm'] = lambda X, n_rff, seed: kernels.ort_fix_norm_RFF(X, n_rff, seed, scale, subsample_all = False, weighted = False)
    algos['ort_fixed_norm_weighted'] = lambda X, n_rff, seed: kernels.ort_fix_norm_RFF(X, n_rff, seed, scale, subsample_all = False, weighted = True)
    # algos['iid_anti'] =      lambda X, n_rff, seed: kernels.make_antithetic(kernels.iid_gaussian_RFF(X, n_rff / 2, seed, scale))
    # algos['ort_anti'] =      lambda X, n_rff, seed: kernels.make_antithetic(kernels.ort_gaussian_RFF(X, n_rff / 2, seed, scale))
    # algos['HD_stack_power'] = lambda X, n_rff, seed: kernels.HD_stack_power_gaussian_RFF(X, n_rff, seed, scale)
    # algos['HD_1'] =           lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 1)
    # algos['HD_fix_norm_1'] =  lambda X, n_rff, seed: kernels.HD_fix_norm_RFF(X, n_rff, seed, scale, 1)
    # algos['HD_fix_norm_1_subsample'] = lambda X, n_rff, seed: kernels.HD_fix_norm_subsample_RFF(X, n_rff, seed, scale, 1)
    # algos['HD_2'] =           lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 2)
    # algos['HD_fix_norm_2'] =  lambda X, n_rff, seed: kernels.HD_fix_norm_RFF(X, n_rff, seed, scale, 2)
    # algos['HD_3'] =           lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 3)
    # algos['HD_fix_norm_3'] =  lambda X, n_rff, seed: kernels.HD_fix_norm_RFF(X, n_rff, seed, scale, 3)
    # algos['fastfood'] =       lambda X, n_rff, seed: kernels.fastfood_RFF(X, n_rff, seed, scale)
    # algos['greedy'] =    lambda X, n_rff, seed: kernels.greedy_unif_gaussian_RFF(X, n_rff, seed, scale)
    # algos['angled_%.3f' % (0.9)]   = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*0.9)
    # algos['angled_%.3f' % (0.95)]  = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*0.95)
    # algos['angled_%.3f' % (1)]     = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1)
    # algos['angled_%.3f' % (1.013)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.013)
    # algos['angled_%.3f' % (1.025)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.025)
    # algos['angled_%.3f' % (1.038)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.038)
    # algos['angled_%.3f' % (1.050)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.050)
    # algos['angled_%.3f' % (1.063)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.063)
    # algos['angled_%.3f' % (1.075)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.075)
    # algos['angled_%.3f' % (1.088)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.088)
    # algos['angled_%.3f' % (1.100)] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, np.pi/2.0*1.100)
    # algos['angled_nb_%.3f' % angle] = lambda X, n_rff, seed: kernels.angled_gaussian_neighbour_RFF(X, n_rff, seed, scale, angle)

    plt.figure(figsize = (6,4))
    results = {}
    # for algo_name, feature_handle in algos.items():#+[('test', None)]+[('test2', None)]:
    algo_keys_plot = ['iid', 'iid_fixed_norm', 'ort', 'ort_weighted', 'ort_fixed_norm', 'ort_fixed_norm_weighted']
    for algo_name, feature_handle in [(algo_key, algos[algo_key]) for algo_key in algo_keys_plot]:
        results[algo_name] = { }
        for n_rff in range(2, max_n_rff, 2):
            results[algo_name][n_rff] = np.zeros(n_seeds)
            for seed in range(n_seeds):
                # np.random.seed(n_rff * seed)
                vector1 = np.random.normal(size = (1, dim))
                vector2 = np.random.normal(size = (1, dim))
                true_K = kernels.gaussian_kernel(vector1, vector2, scale)
                
                if algo_name == 'test': # blocks of orthogonal then complete with iid
                    np.random.seed(n_rff * seed)
                    Omega = kernels.stacked_unif_ort((dim, (n_rff / dim) * dim), subsample_all = False)
                    Omega *= np.sqrt(np.random.chisquare(df = Omega.shape[0], size = Omega.shape[1]))
                    Omega = np.concatenate([Omega, np.random.normal(size = (dim, n_rff - Omega.shape[1]))], axis = 1)
                    est_K = np.mean(np.cos(np.dot(vector1 - vector2, Omega) / scale))
                elif algo_name == 'test2': # always max_n_rff features, n_rff of which are orthogonal, the rest iid
                    np.random.seed(n_rff * seed)
                    Omega = kernels.stacked_unif_ort((dim, n_rff), subsample_all = False)
                    Omega *= np.sqrt(np.random.chisquare(df = Omega.shape[0], size = Omega.shape[1]))
                    Omega = np.concatenate([Omega, np.random.normal(size = (dim, max_n_rff - Omega.shape[1]))], axis = 1)
                    est_K = np.mean(np.cos(np.dot(vector1 - vector2, Omega) / scale))
                else:
                    Phi = feature_handle(np.concatenate([vector1, vector2], axis = 0), n_rff, n_rff * seed)
                    est_K = np.real(np.dot(Phi, np.conj(Phi.T)))[0,1]

                results[algo_name][n_rff][seed] = float(est_K - true_K)
        x = np.array(sorted(results[algo_name].keys()))
        y = np.array([np.mean(np.square(results[algo_name][k])) for k in x])
        counts = np.array([np.sum(results[algo_name][k]>0) for k in x])
        print algo_name, np.mean(y[-dim:]), np.mean(counts[-dim:])#, results[algo_name][x[-1]][-10:]
        
        # plt.subplot(121)
        p = plt.plot(x, y, label = algo_name.replace('_', ' '), linewidth = 1)
        high_perc = np.array([np.percentile(np.square(results[algo_name][k]), 95) for k in x])
        # plt.fill_between(x, y, high_perc, color = p[0].get_color(), alpha = 0.05)
        
        # plt.subplot(122)
        # plt.plot(x, counts, label = algo_name.replace('_', ' '))
    # plt.subplot(122)
    # plt.legend()
    # plt.subplot(121)
    plt.yscale('log')
    plt.ylim(5e-6, 5e-2)
    plt.legend()
    # plt.title('Pointwise SE kernel approximation error')
    plt.xlabel('Number of random Fourier features')
    plt.xlim(0, max_n_rff - 1)
    # plt.ylabel('Mean pointwise error in SE kernel')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('pointwise_SE.eps', bbox_inches = 'tight')
    plt.show()

def polynomial_kernel():
    dim = 20
    degree = 2
    inhom_term = 0
    exact_feat_dim = comb(dim + degree, dim) + (inhom_term != 0)
    print 'Dimension of exact feature space = %d' % exact_feat_dim
    n_seeds = 1000
    algos = {}
    algos['iid'] =      lambda X, n_rff, seed:          kernels.iid_polynomial_sp_random_features(X, n_rff, seed, degree, inhom_term)
    algos['iid_unit'] = lambda X, n_rff, seed:     kernels.iid_polynomial_sp_random_unit_features(X, n_rff, seed, degree, inhom_term)
    algos['ort'] =      lambda X, n_rff, seed: kernels.ort_polynomial_sp_random_gaussian_features(X, n_rff, seed, degree, inhom_term)
    algos['ort_unit'] = lambda X, n_rff, seed:     kernels.ort_polynomial_sp_random_unit_features(X, n_rff, seed, degree, inhom_term)
    algos['HD_unit'] =  lambda X, n_rff, seed:      kernels.HD_polynomial_sp_random_unit_features(X, n_rff, seed, degree, inhom_term)
    plt.figure(figsize = (8,6))
    results = {}
    for algo_name, feature_handle in algos.items():
        results[algo_name] = {}
        for n_rff in range(1, 1 + 10 * dim, dim / 2):
            results[algo_name][n_rff] = []
            for seed in range(n_seeds):
                np.random.seed(seed * n_rff)
                x = np.random.normal(size = (1, dim))
                y = np.random.normal(size = (1, dim))
                x /= np.linalg.norm(x)
                y /= np.linalg.norm(y)

                true_K = kernels.polynomial_sp_kernel(x, y, degree, inhom_term)

                Phix = feature_handle(x, n_rff, seed)
                Phiy = feature_handle(y, n_rff, seed)
                est_K = np.dot(Phix, Phiy.T)
                
                results[algo_name][n_rff].append(float(np.abs(est_K - true_K)))

            # print n_rff, est_K, true_K, est_K - true_K
        x = np.array(sorted(results[algo_name].keys()))
        y = np.array([np.mean(results[algo_name][k]) for k in x])
        print algo_name, y[-1]
        # stds = np.array([np.std(results[algo_name][k]) for k in x])
        p = plt.plot(x, y, label = algo_name.replace('_', '\_'))
        # plt.fill_between(x, y - stds, y + stds, color = p[0].get_color(), alpha = 0.05)
    plt.yscale('log')
    plt.legend()
    plt.title('Pointwise polyn kernel approximation error')
    plt.xlabel('Number of random features')
    plt.xlim(0)
    plt.ylabel('Mean pointwise error polyn kernel')
    plt.plot([exact_feat_dim] * 2, plt.gca().get_ylim(),'-')
    plt.tight_layout()
    plt.savefig('pointwise_polyn.eps', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    # dim = 10
    # degree = 2
    # inhom_term = 3
    # x = np.random.normal(size = (1, dim))
    # y = np.random.normal(size = (1, dim))
    # print kernels.polynomial_sp_kernel(x,y,degree,inhom_term)
    # x = np.pad(x, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    # y = np.pad(y, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    # print kernels.polynomial_sp_kernel(x,y,degree,0)


    squared_exponential_kernel()
    # polynomial_kernel()
