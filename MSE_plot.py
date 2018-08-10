from math import factorial
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp_spec
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def MSE_iid(z, n_rff):
    return (1.0-np.exp(-z**2))**2 / 2.0 / n_rff

def MSE_ort(z, n_rff, dim):
    cov = 0.0
    increment = 1
    for j in range(1, 1000):
        cov += increment
        increment *= -1.0 / j * z ** 2 * (dim + j-1.0) / (dim + 2*j-2.0)
        # print increment, (-1.0)**j / factorial(j) * z**(2*j) * np.prod(np.divide(float(dim) + np.arange(j), dim+2*np.arange(j)))
    cov -= np.exp(-z**2)
    
    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff
    # estimator is made of J indep blocks of dim orthogonal directions followed by one block of I orthogonal directions
    return MSE_iid(z, n_rff) + (I * (I - 1) + J * dim * (dim - 1)) / np.square(n_rff) * cov

def MSE_ort_weighted(z, n_rff, dim):
    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff

    variance_full_blocks = np.zeros(J.shape)
    variance_full_blocks[J>0] = MSE_ort(z, J[J>0] * dim, dim)
    variance_partial_block = np.zeros(I.shape)
    variance_partial_block[I>0] = MSE_ort(z, I[I>0], dim)

    optimal_weight = variance_full_blocks / (variance_full_blocks + variance_partial_block) # estimator = (1-optimal_weight) * full_blocks + optimal_weight * partial_block
    optimal_weight[J==0] = 1
    optimal_weight[I==0] = 0

    variance = (1.0 - optimal_weight) ** 2 * variance_full_blocks + (optimal_weight) ** 2 * variance_partial_block
    # variance = variance_full_blocks * variance_partial_block / (variance_full_blocks + variance_partial_block)
    
    return variance

def mean_fixed_norm(z, dim, fixed_norm):
    z = z * fixed_norm
    mean = np.zeros(z.shape, dtype = np.float64)
    for j in range(100, -1, -1):
        mean = mean + (-1.0 / 2.0)**j * np.power(z, 2.0 * j) / float(factorial(j)) / np.prod(dim + 2.0 * np.arange(j))
    return mean

def MSE_iid_fixed_norm(z, n_rff, dim, fixed_norm):
    bias_cos = np.exp(-z**2 / 2) - mean_fixed_norm(z, dim, fixed_norm)
    var_cos = 0.5 + 0.5 * mean_fixed_norm(z, dim, 2 * fixed_norm) - mean_fixed_norm(z, dim, fixed_norm)**2

    return var_cos / n_rff + np.square(bias_cos)

def MSE_ort_fixed_norm(z, n_rff, dim, fixed_norm):
    bias_cos = np.exp(-z**2 / 2) - mean_fixed_norm(z, dim, fixed_norm)
    var_cos = 0.5 + 0.5 * mean_fixed_norm(z, dim, 2 * fixed_norm) - mean_fixed_norm(z, dim, fixed_norm)**2
    cov_cos = mean_fixed_norm(z, dim, np.sqrt(2) * fixed_norm) - np.square(mean_fixed_norm(z, dim, fixed_norm))

    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff

    # cov_cos = 0 # for quality testing purposes: ort should match iid spot on

    variance = (J * (dim * var_cos + dim * (dim - 1) * cov_cos) + I * var_cos + I * (I - 1) * cov_cos) / np.square(n_rff)
    
    return variance + np.square(bias_cos)

def MSE_ort_fixed_norm_weighted(z, n_rff, dim, fixed_norm):
    bias_cos = np.exp(-z**2 / 2) - mean_fixed_norm(z, dim, fixed_norm)
    var_cos = 0.5 + 0.5 * mean_fixed_norm(z, dim, 2 * fixed_norm) - mean_fixed_norm(z, dim, fixed_norm)**2
    cov_cos = mean_fixed_norm(z, dim, np.sqrt(2) * fixed_norm) - np.square(mean_fixed_norm(z, dim, fixed_norm))

    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff

    variance_full_blocks = np.zeros(J.shape)
    variance_full_blocks[J > 0] = (var_cos + (dim - 1) * cov_cos) / dim / J[J > 0]
    variance_partial_block = np.zeros(I.shape)
    variance_partial_block[I > 0] = (var_cos + (I[I > 0] - 1) * cov_cos) / I[I > 0]

    optimal_weight = variance_full_blocks / (variance_full_blocks + variance_partial_block) # estimator = (1-optimal_weight) * full_blocks + optimal_weight * partial_block
    optimal_weight[J==0] = 1
    optimal_weight[I==0] = 0

    variance = (1.0 - optimal_weight) ** 2 * variance_full_blocks + (optimal_weight) ** 2 * variance_partial_block

    return np.square(bias_cos) + variance

def plot_optimal_weights(z, dims, n_rff):
    dim = dims[0]
    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff

    variance_full_blocks = np.zeros(J.shape)
    variance_full_blocks[J>0] = MSE_ort(z, J[J>0] * dim, dim)
    variance_partial_block = np.zeros(I.shape)
    variance_partial_block[I>0] = MSE_ort(z, I[I>0], dim)

    optimal_weight = variance_full_blocks / (variance_full_blocks + variance_partial_block) # estimator = (1-optimal_weight) * full_blocks + optimal_weight * partial_block
    optimal_weight[J==0] = 1
    optimal_weight[I==0] = 0

    plt.figure(figsize = (6,4))
    index = (0, dim-1)
    plt.plot(n_rff[index[0]:index[1]], optimal_weight[index[0]:index[1]],               linewidth = 1, color = 'C0', label = r'optimal weighting $\frac{Var[E_1]}{Var[E_1]+Var[E_2]}$')
    plt.plot(n_rff[index[0]:index[1]], I[index[0]:index[1]] / n_rff[index[0]:index[1]], linewidth = 1, color = 'C1', label = r'naive equal weighting $\frac{I}{k}$')
    for i in range(1, int(max(n_rff) / dim)):
        index = (max(0, -1+i*dim), -1+(i+1)*dim)
        plt.plot(n_rff[index[0]:index[1]], optimal_weight[index[0]:index[1]],               linewidth = 1, color = 'C0')
        plt.plot(n_rff[index[0]:index[1]], I[index[0]:index[1]] / n_rff[index[0]:index[1]], linewidth = 1, color = 'C1')
    plt.xlim(0, max(n_rff))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.xlabel('Number of random Fourier features')
    plt.ylabel('Optimal weight of partial block')
    plt.tight_layout()
    plt.savefig('optimal_weight.eps', bbox_inches='tight')
    plt.show()

def iid_ort(z, dims, n_rffs):
    plt.figure(figsize = (4,3))
    for dim in dims:
        plt.plot(n_rffs, MSE_ort(z, n_rffs, dim), linewidth = 1, label = 'ort d = %d' % dim)
    plt.plot(n_rffs, MSE_iid(z, n_rffs), linewidth = 1, label = 'iid')
    
    plt.legend()
    plt.xlim(0, max(n_rffs))
    plt.yscale('log')
    plt.ylim(ymax = 0.01)
    plt.xlabel('Number of random Fourier features')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('MSE_ort_iid.eps', bbox_inches='tight')
    plt.show()

def iid_ort_weighted(z, dims, n_rffs, fixed_norm_handle = None):
    plt.figure(figsize = (6,4))
    dim = dims[0]
    fixed_norm = fixed_norm_handle(dim)
    plt.plot(n_rffs, MSE_iid(z, n_rffs), linewidth = 1, label = 'iid', color = 'C0')
    plt.plot(n_rffs, MSE_iid_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'iid fixed norm', color = 'C1')

    plt.plot(n_rffs, MSE_ort(z, n_rffs, dim), linewidth = 1, label = 'ort', color = 'C2')
    plt.plot(n_rffs, MSE_ort_weighted(z, n_rffs, dim), linewidth = 1, label = 'ort weighted', color = 'C3')
    
    plt.plot(n_rffs, MSE_ort_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'ort fixed norm', color = 'C4')
    plt.plot(n_rffs, MSE_ort_fixed_norm_weighted(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'ort fixed norm weighted', color = 'C5')
    
    plt.legend()
    plt.xlim(0, max(n_rffs))
    plt.yscale('log')
    plt.ylim(5e-6, 5e-2)
    plt.xlabel('Number of random Fourier features')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('MSE_iid_ort_weighted.eps', bbox_inches='tight')
    plt.show()

def iid_ort_fixed_norm(z, dims, n_rffs, fixed_norm_handle):
    plt.figure(figsize = (4,3))
    for dim in dims:
        fixed_norm = fixed_norm_handle(dim)
        plt.plot(n_rffs, MSE_ort_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'fixed norm ort d = %d' % dim)
    plt.plot(n_rffs, MSE_iid_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'fixed norm iid')
    
    plt.legend()
    plt.xlim(0, max(n_rffs))
    plt.yscale('log')
    plt.ylim(ymax = 0.1)
    plt.xlabel('Number of random Fourier features')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('MSE_iid_ort_fixed_norm.eps', bbox_inches='tight')
    plt.show()

def bias_fixed_norm(dims, fixed_norm_handle):
    zs = np.linspace(0.001, 3.0, 1000)
    
    plt.figure(figsize = (4,3))
    plt.plot(zs, np.exp(-np.square(zs) / 2), linewidth = 1, label = 'exact')
    for dim in dims:
        fixed_norm = fixed_norm_handle(dim)
        plt.plot(zs, mean_fixed_norm(zs, dim, fixed_norm), linewidth = 1, label = 'fixed norm d = %d' % dim)
    
    plt.legend()
    plt.xlim(0, max(zs))
    # plt.yscale('log')
    plt.xlabel(r'$||x-y||$')
    plt.ylabel('mean')
    plt.tight_layout()
    plt.savefig('bias_fixed_norm.eps', bbox_inches='tight')
    plt.show()

def iid_fixed_norm(dims, fixed_norm_handle): 
    n_rff = 1.0
    zs = np.linspace(0.001, 3.0, 1000)
    
    plt.figure(figsize = (4,3))
    plt.plot(zs, MSE_iid(zs, n_rff), linewidth = 1, label = 'MSE iid')
    for dim in dims:
        fixed_norm = fixed_norm_handle(dim)
        plt.plot(zs, MSE_iid_fixed_norm(zs, n_rff, dim, fixed_norm), linewidth = 1, label = 'iid fixed norm d = %d' % dim)
    
    plt.legend()
    plt.xlim(0, max(zs))
    plt.yscale('log')
    plt.ylim(0.05, 1.5)
    plt.xlabel(r'$||x-y||$')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig('MSE_iid_fixed_norm.eps', bbox_inches='tight')
    plt.show()

def iid_ort_indep_fixed_indep_norm(z, dims, n_rffs, fixed_norm_handle):
    plt.figure(figsize = (4,3))
    for dim in dims:
        fixed_norm = fixed_norm_handle(dim)
        plt.plot(n_rffs, MSE_ort_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'fixed norm ort d = %d' % dim)
        plt.plot(n_rffs, MSE_ort(z, n_rffs, dim), linewidth = 1, label = 'indep norm ort d = %d' % dim)
    plt.plot(n_rffs, MSE_iid_fixed_norm(z, n_rffs, dim, fixed_norm), linewidth = 1, label = 'fixed norm iid')
    plt.plot(n_rffs, MSE_iid(z, n_rffs), linewidth = 1, label = 'indep norm iid')
    
    # lower bound on MSE: squared bias
    bias_fixed = np.exp(-z**2 / 2) - mean_fixed_norm(z, dim, fixed_norm)
    plt.plot([np.min(n_rffs), np.max(n_rffs)], np.square([bias_fixed, bias_fixed]), linestyle = '--', linewidth = 1, label = 'Square bias fixed norm d = %d' % dim)
    
    plt.xlim(1e-10, max(n_rffs))
    plt.yscale('log')
    plt.ylim(ymax = 2e-3)
    plt.xlabel('Number of random Fourier features')
    plt.ylabel('MSE')
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.savefig('MSE_iid_ort_fixed_indep_norm.eps', bbox_inches='tight')
    plt.show()

def main():
    fixed_norm_sqrt = lambda d: np.sqrt(d)
    fixed_norm_mean_chi_sq = lambda d: np.sqrt(2) * sp_spec.gamma((d + 1.0) / 2.0) / sp_spec.gamma(d / 2.0)

    z = 1.0

    # dims = np.power(2, np.arange(2, 10, step = 2), dtype = np.float32)
    # dims = [1,2,3,4]
    # n_rffs = np.arange(1, 4*max(dims), dtype = np.float32)
    # iid_ort(z, dims, n_rffs)
    # iid_ort_weighted(np.sqrt(2 * 24) / (2.0 * np.sqrt(24)), [24], np.arange(1, 10 * 24 + 1, dtype = np.float32), fixed_norm_mean_chi_sq)
    # plot_optimal_weights(1, [24], np.arange(1, 10 * 24 + 1, dtype = np.float32))

    # dims = [8, 16, 32]
    # bias_fixed_norm(dims, fixed_norm_mean_chi_sq)
    # iid_fixed_norm(dims, fixed_norm_mean_chi_sq)
    # z = 1.25
    # n_rffs = np.arange(1, max(dims), dtype = np.float32)
    # iid_ort_fixed_norm(z, dims, n_rffs, fixed_norm_mean_chi_sq)

    dims = [8]
    n_rffs = np.arange(1, 1000 * max(dims), dtype = np.float32)
    iid_ort_indep_fixed_indep_norm(z, dims, n_rffs, fixed_norm_mean_chi_sq)

if __name__ == '__main__':
    main()

