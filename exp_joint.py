import numpy as np
import matplotlib.pyplot as plt
import kernels

def avg_cross_dist(X):
    # X.shape = (dim, n_samples)
    d = 0.0
    for i in range(0, X.shape[1] - 1):
        d += np.sum(np.linalg.norm(X[:, i+1:] - X[:, i, np.newaxis], axis = 0))
    d = d * 2.0 / (X.shape[1] * (X.shape[1] - 1.0))
    return d

if __name__ == '__main__':
    dim = 20
    n_samples = 40
    n_seeds = 100
    avg_dists = np.zeros((n_samples - 1, 2))
    for seed in range(n_seeds):
        np.random.seed(seed)
        # generate 100 samples
        X_iid = np.random.normal(size = (dim, n_samples))
        # X_iid = np.divide(X_iid, np.linalg.norm(X_iid, axis = 0))
        X_ort = kernels.stacked_unif_ort((dim, n_samples))
        X_ort = np.multiply(X_ort, np.linalg.norm(np.random.normal(size = (dim, n_samples)), axis = 0))
        # X_greedy = kernels.greedy_unif_directions(dim, n_samples)
        # measure the avg dis for 2, ..., 100 samples
        for i in range(2, n_samples):
            avg_dists[i-1] += np.array([avg_cross_dist(X_iid[:, :i]), avg_cross_dist(X_ort[:, :i])]) / n_seeds
    plt.plot(avg_dists[:,0], '.', label = 'iid')
    plt.plot(avg_dists[:,1], '.', label = 'stacked_ort')
    # plt.plot(avg_dists[:,2], '.', label = 'greedy')
    plt.title('Avg distance of %d unit samples in dimension %d' % (n_samples, dim))
    plt.legend()
    plt.show()
