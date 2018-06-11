import numpy as np
import matplotlib.pyplot as plt
import kernels

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


if __name__ == '__main__':
    dim = 30
    scale = 0.5
    n_seeds = 50
    algos = {'iid':      lambda X, n_rff, seed: kernels.iid_gaussian_RFF(X, n_rff, seed, scale), \
            'iid_anti':  lambda X, n_rff, seed: kernels.make_antithetic(kernels.iid_gaussian_RFF(X, n_rff, seed, scale)), \
            'ort':       lambda X, n_rff, seed: kernels.ort_gaussian_RFF(X, n_rff, seed, scale), \
            'ort_anti':  lambda X, n_rff, seed: kernels.make_antithetic(kernels.ort_gaussian_RFF(X, n_rff, seed, scale))}#, \
            # 'HD_1':      lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 1), \
            # 'HD_2':      lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 2), \
            # 'HD_3':      lambda X, n_rff, seed: kernels.HD_gaussian_RFF(X, n_rff, seed, scale, 3)}
    plt.figure(figsize = (8,6))
    results = {}
    for algo_name, feature_handle in algos.items():
        results[algo_name] = {}
        for n_rff in range(1, 1 + 10 * dim):
            results[algo_name][n_rff] = []
            for seed in range(n_seeds):
                np.random.seed(seed)
                z = np.random.normal(size = (1, dim))
                true_K = kernels.gaussian_kernel(np.zeros((1, dim)), z, scale)
                
                Phiz = feature_handle(z, n_rff, seed)
                Phi0 = feature_handle(np.zeros(z.shape), n_rff, seed)
                est_K = np.dot(Phiz, np.conj(Phi0.T))
                
                results[algo_name][n_rff].append(float(np.abs(est_K - true_K)))
        x = np.array(sorted(results[algo_name].keys()))
        y = np.array([np.mean(results[algo_name][k]) for k in x])
        # stds = np.array([np.std(results[algo_name][k]) for k in x])
        p = plt.plot(x, y, label = algo_name.replace('_', '\_'))
        # plt.fill_between(x, y - stds, y + stds, color = p[0].get_color(), alpha = 0.05)
    plt.yscale('log')
    plt.legend()
    plt.title('Pointwise kernel approximation error')
    plt.xlabel('Number of random features')
    plt.xlim(0)
    plt.ylabel('Mean pointwise error')
    plt.tight_layout()
    plt.savefig('pointwise.eps', bbox_inches = 'tight')
    plt.show()

        

