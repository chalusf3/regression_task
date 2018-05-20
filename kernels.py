import numpy as np
import scipy.stats as sp_stats

"""
deterministic Gaussian kernel
"""
def gaussian_kernel(X, loc, scale):
    # returns the gram matrix K_ij = exp(-||X_j - loc_i||^2/(2*scale^2))
    # X is given with rows as sample vectors (size n_samples x dimension_samples), same for loc
    K = np.tile(np.square(np.linalg.norm(X, axis = 1)), [loc.shape[0], 1]).T + np.tile(np.square(np.linalg.norm(loc, axis = 1)), [X.shape[0], 1]) - 2 * np.dot(X, loc.T)
    K = np.exp(- K / 2 / scale**2)
    return K

def gaussian_kernel_gram(X, scale):
    # returns the gram matrix K_ij = exp(-||X_i-X_j||^2/(2*scale^2))
    # X is given with rows as sample vectors (size n_samples x dimension_samples)
    return gaussian_kernel(X, X, scale)
    
"""
random Fourier features iid
"""
def iid_gaussian_RFF(X, n_rff, seed, scale):
    # returns the matrix (cos(<omega_{1}, X>), ..., cos(<omega_{n_rff}, X>)) / sqrt(n_rff)
    # where omega_{i} are i.i.d. N(0, 1/scale^2)
    np.random.seed(seed)
    omega = np.random.normal(loc = 0.0, scale = 1.0, size = (X.shape[1], n_rff))
    # omega = np.random.normal(loc = 0.0, scale = 1.0, size = (n_rff, X.shape[1])).T # fourier frequencies are columns! they must be identical when using the same seeds and X.shape[1] but different n_rff
    PhiX = np.cos(np.dot(X, omega) / scale) / np.sqrt(n_rff)
    return PhiX

"""
random Fourier features stack iid orthogonal
"""
def unif_ort(dim): 
    # generates a uniform orthonormal matrix using standard scipy method (slower)
    return sp_stats.ortho_group.rvs(dim = dim)

def unif_ort_QR(dim): 
    # generates a uniform orthonormal matrix using QR decomposition of a random gaussian matrix (faster)
    G_ort, R = np.linalg.qr(np.random.normal(size = (dim, dim)))
    G_ort = np.dot(G_ort, np.diag(np.sign(np.diag(R))))
    return G_ort

def unif_ort_gaussian(shape): 
    # generates a matrix with shape[1] orthogonal columns of lengths shape[0], each marginally gaussian
    G = [unif_ort_QR(shape[0]) for _ in range(int(np.ceil(float(shape[1]) / shape[0])))]
    G = np.concatenate(G, axis = 1) # TODO: create a 2nd version which couples the gaussians here? 
    G = G[:, :shape[1]]
    norms = np.sqrt(np.random.chisquare(df = shape[0], size = (1, shape[1])))
    G = np.multiply(norms, G)
    return G

def ort_gaussian_RFF(X, n_rff, seed, scale):
    np.random.seed(seed)
    omega = unif_ort_gaussian((X.shape[1], n_rff))
    PhiX = np.cos(np.dot(X, omega) / scale) / np.sqrt(n_rff)
    return PhiX

"""
Generate Fourier features with a certain angle between all vectors (if dimension allows it) 
"""
def T_matrix(C):
    # Let C be of size (k,k), this function computes one upper triangular matrix T of size (k,k) such that T.T * T = C
    # if X is a (d,k) matrix (d>k) and X.T * X = eye(k) 
    # (e.g. X is a matrix of k orthogonal d-dimensional columns) then if U = XT it holds that U.T * U = C, e.g. the scalar product between
    # the i-th and j-th columns of U is exactly C_ij.
    # Complexity: O(k^3), NOT vectorized. Considering the results when C = (1, c, c, ..., c \\ c, 1, c, c, ..., c \\ ... \\ c, c, ..., c, 1) it could be optimized for this particular case. 
    k = C.shape[0]
    T = np.zeros((k, k))
    T[0, 0] = np.sqrt(C[1, 1])
    for j in xrange(1, k):
        T[0, j] = C[0, j] / T[0, 0]
        for i in xrange(1, j):
            T[i, j] = (C[i, j] - np.dot(T[:(i), i].T, T[:(i), j])) / T[i, i]
        T[j, j] = np.sqrt(C[j,j] - np.dot(T[:, j], T[:, j].T))
    
    return T
C = -0.1 * np.ones((10, 10))
np.fill_diagonal(C, 1)
T = T_matrix(C)
print T
print np.dot(T.T, T)

"""
random Fourier features stacked with Hadamard-Rademacher products
"""
def hadamard_product(R): 
    # computes fast the product H_n * R where H_n is the 2^n x 2^n Hadamard matrix WITHOUT any normalization factor, output is a matrix of +- 1's
    lR = R.shape[0]
    half_lR = int(lR / 2)
    alt_idx = list(range(0, lR, 2)) + list(range(1, lR, 2))
    n = int(np.log(lR) / np.log(2))
    for _ in range(n):
        R[alt_idx] = np.concatenate([R[0:half_lR] + R[half_lR:], R[0:half_lR] - R[half_lR:]])
    return R

def hadamard_product_rec(R):
    # computes recursively the product H_n * R where H_n is the 2^n x 2^n Hadamard matrix WITHOUT any normalization factor, output is a matrix of +- 1's
    if R.shape[0] == 1:
        return R
    else:
        HR1 = hadamard_product_rec(R[0:len(R)/2])
        HR2 = hadamard_product_rec(R[len(R)/2:])
        return np.vstack([HR1 + HR2, HR1 - HR2])

def hadamard_rademacher_product(x, k): 
    # returns the product of  (HD)_k ... (HD)_1 x, x must be a matrix with feature vectors as x_shape0-dimensional columns. In other words it returns the product of x with an orthogonal approximately uniform matrix. 
    n = int(np.log(x.shape[0]) / np.log(2))
    D = -1.0 + 2 * np.random.binomial(1,0.5, size=(x.shape[0],k))
    for i in range(k):
        x = D[:,i][:, np.newaxis] * x
        x = hadamard_product(x) # the multiplication with D occurs row-wise (since the D matrices are applied on the left of x, they are elementary row operations)
    x = x / np.sqrt(2)**(n*k)
    return x

def hadamard_rademacher_product_scale_chi(X, n_rff, k):
    # Returns the product of x with orthogonal vectors, each having an approximately Gaussian marginal
    original_dimension = X.shape[1] # dimension of input feature vector
    HD_dim = 2 ** (int(np.ceil(np.log(X.shape[1]) / np.log(2)))) # the smallest power of 2 >= x.shape[1]. We embed X in (X.shape[0], R^{HD_dim}) by zero padding
    X = np.pad(X, ((0,0), (0,HD_dim-X.shape[1])), 'constant', constant_values = ((np.nan, np.nan), (np.nan, 0)))
    
    # The output of hadamard_rademacher_product is a (HD_dim, x.shape[1]) matrix. We stack ceil(n_rff/HD_dim) of those to get a (ceil(n_rff/HD_dim)*HD_dim, x.shape[1]) matrix
    X = np.concatenate([hadamard_rademacher_product(X.T, k) for _ in range(int(np.ceil(float(n_rff) / HD_dim)))]).T
    # X is now of shape (X.shape[0], ceil(n_rff / HD_dim) * HD_dim)
    
    # Then we discard some columns from the last block
    idx_last_block = int(np.floor(float(n_rff) / HD_dim)) * HD_dim
    idx = np.random.choice(HD_dim, size = n_rff - idx_last_block, replace = False) # those indices of the last block we'll keep
    X = np.concatenate([X[:, 0:idx_last_block], X[:, idx_last_block + idx]], axis = 1)

    # Scale all rows independently so that they're marginally Gaussian
    X *= np.sqrt(float(HD_dim) / original_dimension)
    norms = np.sqrt(np.random.chisquare(df = original_dimension, size = (1, n_rff)))
    X = norms * X
    
    return X

def HD_gaussian_RFF(X, n_rff, seed, scale, k):
    np.random.seed(seed)
    K = hadamard_rademacher_product_scale_chi(X, n_rff, k)
    PhiX = np.cos(K / scale) / np.sqrt(n_rff)
    return PhiX
