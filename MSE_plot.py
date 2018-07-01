from math import factorial
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def MSE_iid(z, n_rff):
    return (1-np.exp(-z**2))**2 / 2 / n_rff

def MSE_ort(z, n_rff, dim):
    cov = 0.0
    for j in range(100, -1, -1):
        cov += (-1.0)**j / factorial(j) * z**(2 * j) * (np.prod(np.divide(dim + np.arange(j, dtype = np.float32), dim + 2.0 * np.arange(j, dtype = np.float32))) - 1.0)
    I = np.mod(n_rff, dim, dtype = np.float32)
    J = np.floor(n_rff / dim, dtype = np.float32) # J * dim + I = n_rff
    return MSE_iid(z, n_rff) + (I * (I - 1) + (J - 1) * dim * (dim - 1)) / np.square(n_rff) * cov

def MSE_HD(z, n_rff, dim):
    pass

def main():
    plt.figure(figsize = (4,3))

    zs = [0.75]
    dims = np.power(2, np.arange(10, step = 2), dtype = np.float32)
    n_rffs = np.arange(1, 4*dims[-1], dtype = np.float32)
    for idx, z in enumerate(zs):
        plt.subplot(1, len(zs), idx+1)
        for dim in dims:
            plt.plot(n_rffs, MSE_ort(z, n_rffs, dim), markersize = 1, linewidth = 1, label = 'MSE ort d = %d' % dim)
        plt.plot(n_rffs, MSE_iid(z, n_rffs), markersize = 3, linewidth = 1, label = 'MSE iid' % dim)
        
        plt.legend()
        plt.xlim(0, max(n_rffs))
        plt.yscale('log')
        plt.ylim(ymax = 0.01)
        plt.xlabel('Number of random Fourier features')
        plt.ylabel('MSE')
        plt.tight_layout()
    plt.savefig('MSE_ort_iid.eps', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

