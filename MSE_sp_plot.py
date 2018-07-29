import numpy as np
import matplotlib.pyplot as plt

def var_gaussian(x_dot_x, y_dot_y, x_dot_y, degree): 
    # variance of x^Tz1 y^Tz1 * ... * x^Tzdegree y^Tzdegree (degree times) where z are indep Gaussian
    return np.power(x_dot_x * y_dot_y + 2 * np.square(x_dot_y), degree) - np.power(x_dot_y, 2 * degree)

def var_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree):
    # variance of x^Tz1 y^Tz1 * ... * x^Tzdegree y^Tzdegree (degree times) where z are unit * sqrt(d)
    return np.power(dim / (dim + 2.0) * (x_dot_x * y_dot_y + 2 * np.square(x_dot_y)), degree) - np.power(x_dot_y, 2 * degree)

def cov_unit_ort(x_dot_x, y_dot_y, x_dot_y, dim, degree):
    # covariance of x^Tz1 y^Tz1 * ... * x^Tzdegree y^Tzdegree, x^Tw1 y^Tw1 * ... * x^Twdegree y^Twdegree (degree times) where z,w are unit * sqrt(d) conditionally orthogonal
    return np.power(np.divide(np.square(dim), (dim+2)*(dim-1)) * (np.square(x_dot_y) - x_dot_x * y_dot_y / dim), degree) - np.power(x_dot_y, 2 * degree)

def MSE_iid_gaussian(x_dot_x, y_dot_y, x_dot_y, degree, n_rf):
    return np.divide(var_gaussian(x_dot_x, y_dot_y, x_dot_y, degree), n_rf)

def MSE_iid_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf):
    return np.divide(var_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree), n_rf)

def MSE_ort_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf):
    variance = var_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree)
    covariance = cov_unit_ort(x_dot_x, x_dot_y, x_dot_y, dim, degree)
    print(variance, covariance)
    I = np.mod(n_rf, dim, dtype = np.float32)
    J = np.floor(n_rf / dim, dtype = np.float32) # J * dim + I = n_rf

    return (J * (dim * variance + dim * (dim - 1) * covariance) + I * variance + I * (I - 1) * covariance) / np.square(n_rf)

def MSE_ort_unit_weighted(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf):
    variance = var_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree)
    covariance = cov_unit_ort(x_dot_x, x_dot_y, x_dot_y, dim, degree)
    print(variance, covariance)

    I = np.mod(n_rf, dim, dtype = np.float32)
    J = np.floor(n_rf / dim, dtype = np.float32) # J * dim + I = n_rf

    variance_full_blocks = np.zeros(J.shape)
    variance_full_blocks[J > 0] = (variance + (dim - 1) * covariance) / dim / J[J > 0]
    variance_partial_block = np.zeros(I.shape)
    variance_partial_block[I > 0] = (variance + (I[I > 0] - 1) * covariance) / I[I > 0]

    optimal_weight = variance_full_blocks / (variance_full_blocks + variance_partial_block) # estimator = (1-optimal_weight) * full_blocks + optimal_weight * partial_block
    optimal_weight[J==0] = 1
    optimal_weight[I==0] = 0

    variance = (1.0 - optimal_weight) ** 2 * variance_full_blocks + (optimal_weight) ** 2 * variance_partial_block

    return variance

def plot_MSE_iid_gaussian_unit():
    dim = 24
    n_rf = np.arange(1, 1+5*dim)
    degree = 2
    x_dot_x = 1.0
    y_dot_y = 1.0
    for x_dot_y in np.arange(0.0, 1.0, 0.05):
        MSE_g = MSE_iid_gaussian(x_dot_x, y_dot_y, x_dot_y, degree, n_rf)
        MSE_u = MSE_iid_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf)
        MSE_ou = MSE_ort_unit(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf)
        MSE_ouw = MSE_ort_unit_weighted(x_dot_x, y_dot_y, x_dot_y, dim, degree, n_rf)
        plt.figure(figsize = (6,4))
        plt.plot(n_rf, MSE_g, label = 'iid Gaussian', linewidth = 1, color = 'C0')
        plt.plot(n_rf, MSE_u, label = 'iid unit', linewidth = 1, color = 'C1')
        plt.plot(n_rf, MSE_ou, label = 'ort unit', linewidth = 1, color = 'C2')
        plt.plot(n_rf, MSE_ouw, label = 'ort unit weighted', linewidth = 1, color = 'C3')
        plt.yscale('log')
        plt.xlim(0, max(n_rf)-1)
        plt.legend()
        plt.xlabel('Number of random features')
        plt.ylabel('MSE')
        plt.ylim(1e-2, 3e0)
        plt.tight_layout()
        plt.savefig('theo_MSE/MSE_polyn_iid_ort_gaussian_unit_%.3f.eps' % x_dot_y)
        plt.clf()
        # plt.show()

def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plot_MSE_iid_gaussian_unit()



if __name__ == '__main__':
    main()

