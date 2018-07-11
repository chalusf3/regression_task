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

def plot_bias_var():
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

def plot_cov_cos():
    # plot cov(cos(<z,u1>),cos(<z,u2>)) for u1, u2 unit orthogonal and different lengths ||z||
    from MSE_plot import mean_fixed_norm
    zvals = np.linspace(0,1.1,1000)
    
    plt.subplot(121)
    for dim in [4,8,16,32,64,128]:
        cov_cos = mean_fixed_norm(zvals, dim, np.sqrt(2)) - np.square(mean_fixed_norm(zvals, dim, 1))
        plt.plot(zvals, cov_cos, linewidth = 1, label = 'cov cos dim = %d fixed norm' % dim)
    plt.legend()

    # plot cov(cos(<z,u1>),cos(<z,u2>)) for u1, u2 gaussian orthogonal and different lengths ||z||
    plt.subplot(122)
    for dim in [16,32,64]:
        dim = float(dim)
        mean_prod = 0.0
        increment = 1.0
        # increment = fact2(dim-2) / fact2(2*dim-3)
        print increment
        for j in range(25):
            # mean_prod += (-1.0)**j * float(factorial(dim + j - 1.0)) / float(fact2(dim + 2.0 * j - 2.0)) / float(factorial(j)) * np.power(zvals, 2.0 * j) 
        # mean_prod *= float(fact2(dim)) / float(factorial(dim))
            # mean_prod += increment
            # increment *= -1.0 / (1.0 + j) * np.square(zvals) * float(dim + j) / (dim + 2.0 * j)

            # mean_prod = mean_prod + increment
            # increment *= -np.square(zvals) / 2.0 / (j + 1.0) * (2.0 * (dim+j) - 1.0) / (2.0 * j + dim)
            # print increment[[200, -200]]
            pass
            
        cov_cos = mean_prod - np.exp(-np.square(zvals))
        plt.plot(zvals, cov_cos, linewidth = 1, label = 'cov cos dim = %d free norm' % dim)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_bias_var()
    plot_cov_cos()