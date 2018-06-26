import numpy as np
import matplotlib.pyplot as plt
import kernels
from scipy.special import comb

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def squared_exponential_kernel():
    dim = 30
    scale = np.sqrt(dim) * 2
    n_seeds = 100
    algos = {}
    algos['iid'] =           lambda X, n_rff, seed: kernels.iid_gaussian_RFF(X, n_rff, seed, scale)
    algos['iid_fix_norm'] =  lambda X, n_rff, seed: kernels.iid_fix_norm_RFF(X, n_rff, seed, scale)
    algos['iid_invnorm'] =   lambda X, n_rff, seed: kernels.iid_invnorm_gaussian_RFF(X, n_rff, seed, scale)
    algos['ort'] =           lambda X, n_rff, seed: kernels.ort_gaussian_RFF(X, n_rff, seed, scale)
    algos['ort_fix_norm'] =  lambda X, n_rff, seed: kernels.ort_fix_norm_RFF(X, n_rff, seed, scale)
    # algos['iid_anti'] =      lambda X, n_rff, seed: kernels.make_antithetic(kernels.iid_gaussian_RFF(X, n_rff / 2, seed, scale))
    # algos['ort_anti'] =      lambda X, n_rff, seed: kernels.make_antithetic(kernels.ort_gaussian_RFF(X, n_rff / 2, seed, scale))
    algos['HD_1'] =          lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 1)
    algos['HD_fix_norm_1'] = lambda X, n_rff, seed: kernels.HD_fix_norm_RFF(X, n_rff, seed, scale, 1)
    # algos['HD_2'] =        lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 2)
    # algos['HD_3'] =        lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 3)
    algos['fastfood'] =    lambda X, n_rff, seed: kernels.fastfood_RFF(X, n_rff, seed, scale)
    # for angle in [np.pi/2*0.6, np.pi/2*0.8, np.pi/2*1.2, np.pi/2*1.4]:
    #     algos['angled_%.3f' % angle] = lambda X, n_rff, seed: kernels.angled_gaussian_RFF(X, n_rff, seed, scale, angle)
    #     algos['angled_nb_%.3f' % angle] = lambda X, n_rff, seed: kernels.angled_gaussian_neighbour_RFF(X, n_rff, seed, scale, angle)

    plt.figure(figsize = (8,6))
    results = {}
    for algo_name, feature_handle in algos.items():
        results[algo_name] = {}
        for n_rff in range(2, 1 + 10 * dim, 2):
            results[algo_name][n_rff] = []
            for seed in range(n_seeds):
                np.random.seed(n_rff * seed)
                x = np.random.normal(size = (1, dim))
                y = np.random.normal(size = (1, dim))
                true_K = kernels.gaussian_kernel(x, y, scale)
                
                Phix = feature_handle(x, n_rff, seed * n_rff)
                Phiy = feature_handle(y, n_rff, seed * n_rff)
                est_K = np.real(np.dot(Phix, np.conj(Phiy.T)))

                results[algo_name][n_rff].append(float(est_K - true_K))
        x = np.array(sorted(results[algo_name].keys()))
        y = np.array([np.mean(np.abs(results[algo_name][k])) for k in x])
        print algo_name, np.mean(y[-10:])#, results[algo_name][x[-1]][-10:]
        # stds = np.array([np.std(results[algo_name][k]) for k in x])
        p = plt.plot(x, y, label = algo_name.replace('_', '\_'))
        # for xval in x:
        #     plt.scatter([xval] * len(results[algo_name][k]), results[algo_name][k])
        # plt.fill_between(x, y - stds, y + stds, color = p[0].get_color(), alpha = 0.05)
    # plt.yscale('log')
    plt.ylim(1e-16)
    plt.legend()
    plt.title('Pointwise SE kernel approximation error')
    plt.xlabel('Number of random features')
    plt.xlim(0)
    plt.ylabel('Mean pointwise error in SE kernel')
    plt.tight_layout()
    plt.savefig('pointwise.eps', bbox_inches = 'tight')
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