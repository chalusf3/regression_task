import matplotlib.pyplot as plt
import numpy as np
from math import factorial
import scipy.special as sp_spec

def fact2(n):
    if n % 2 == 0:
        return factorial(n / 2) * 2 ** (n / 2)
    else:
        return factorial(n) / fact2(n-1)

def mean_fixed_norm_est(z, dim, fix_n):
    m = 100
    ret = np.zeros(z.shape)
    for k in range(m, -1, -1):
        ret += (-1)**k * fix_n**(2*k) / fact2(2 * k) * fact2(dim - 2) / fact2(dim + 2 * k - 2) * np.power(z, 2*k)
    return ret

def var_fixed_norm_est(z, dim, fix_n):
    m = 100
    return 0.5 * (1.0 + mean_fixed_norm_est(2 * z, dim, fix_n)) - mean_fixed_norm_est(z, dim, fix_n)**2

def main():
    dim = 4
    xvals = np.linspace(-10, 10, 1000)
    SE = np.exp(-np.power(xvals, 2) / 2)
    
    fixed_est_mean = mean_fixed_norm_est(xvals, dim, np.sqrt(2) * sp_spec.gamma((dim+1.0)/2.0) / sp_spec.gamma(dim/2.0))
    var_fixed_est_mean = var_fixed_norm_est(xvals, dim, np.sqrt(2) * sp_spec.gamma((dim+1.0)/2.0) / sp_spec.gamma(dim/2.0))
    fixed_est_sqrt_dim = mean_fixed_norm_est(xvals, dim, np.sqrt(dim))
    var_fixed_est_sqrt_dim = var_fixed_norm_est(xvals, dim, np.sqrt(dim))
    
    plt.subplot(121)
    p = plt.plot(xvals, fixed_est_mean, label = 'fixed norm estimation, mean')
    plt.fill_between(xvals, fixed_est_mean - var_fixed_est_mean, fixed_est_mean + var_fixed_est_mean, alpha = 0.1, color = p[0].get_color())
    p = plt.plot(xvals, fixed_est_sqrt_dim, label = 'fixed norm estimation, sqrt(dim)')
    plt.fill_between(xvals, fixed_est_sqrt_dim - var_fixed_est_sqrt_dim, fixed_est_sqrt_dim + var_fixed_est_sqrt_dim, alpha = 0.1, color = p[0].get_color())
    plt.plot(xvals, SE, label = 'true SE')
    plt.xlim(-5,5)
    plt.ylim(-1,1)
    plt.legend()

    plt.subplot(122)
    p = plt.plot(xvals, np.square(fixed_est_mean - SE), label = 'squared bias, mean')
    q = plt.plot(xvals, np.square(fixed_est_sqrt_dim - SE), label = 'squared bias, sqrt(dim)')
    plt.plot(xvals, var_fixed_est_mean, color = p[0].get_color(), label = 'var, mean')
    plt.plot(xvals, var_fixed_est_sqrt_dim, color = q[0].get_color(), label = 'var, sqrt(dim)')
    plt.plot(xvals, var_fixed_est_mean + np.square(fixed_est_sqrt_dim - SE), label = 'MSE, mean', color = p[0].get_color())
    plt.plot(xvals, var_fixed_est_sqrt_dim + np.square(fixed_est_sqrt_dim - SE), label = 'MSE, sqrt(dim)', color = q[0].get_color())
    plt.xlim(-5,5)
    plt.ylim(-1,1)
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
