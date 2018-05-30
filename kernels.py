import warnings
import numpy as np
import scipy.stats as sp_stats

"""
deterministic Gaussian kernel
"""
def gaussian_kernel(X, loc, scale):
    # returns the gram matrix K_ij = exp(-||X_i - loc_j||^2/(2*scale^2))
    # X is given with rows as sample vectors (size n_samples x dimension_samples), same for loc
    K = np.tile(np.square(np.linalg.norm(X, axis = 1)), [loc.shape[0], 1]).T + np.tile(np.square(np.linalg.norm(loc, axis = 1)), [X.shape[0], 1]) - 2 * np.dot(X, loc.T)
    K = np.exp(- K / 2 / scale**2)
    return K
# X = np.random.normal(size = (5,6))
# loc = np.random.normal(size = (3, 6))
# print gaussian_kernel(X, loc, 0.4)
# print gaussian_kernel(X, loc, 0.4)[3,1], np.exp(- np.linalg.norm(loc[1] - X[3]) ** 2 / 2 / 0.4 ** 2)

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
    if n_rff % 2 != 0:
        raise Warning('One can only generate an even number of random fourier features (cos + sin)')
    omega = np.random.normal(loc = 0.0, scale = 1.0 / scale, size = (X.shape[1], n_rff / 2)) # random frequencies
    # PhiX = np.cos(np.dot(X, omega)) / np.sqrt(n_rff)
    PhiX = np.concatenate([np.cos(np.dot(X, omega)), np.sin(np.dot(X, omega))], axis = 1) / np.sqrt(n_rff)
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

def stacked_unif_ort(shape):
    # generates a matrix with shape[1] orthonormal columns of dimension shape[0]
    G = [unif_ort_QR(shape[0]) for _ in range(int(np.ceil(float(shape[1]) / shape[0])))]
    G = np.concatenate(G, axis = 1) # TODO: create a 2nd version which couples the gaussians here? 
    G = G[:, :shape[1]]
    return G

def unif_ort_gaussian(shape): 
    # generates a matrix with shape[1] orthogonal columns of lengths shape[0], each marginally gaussian
    G = stacked_unif_ort(shape)
    norms = np.sqrt(np.random.chisquare(df = shape[0], size = (1, shape[1])))
    G = np.multiply(norms, G)
    return G

def ort_gaussian_RFF(X, n_rff, seed, scale):
    # generates n_rff orthogonal frequencies of dimension X.shape[1] (e.g. omega is of shape (X.shape[1], n_rff))
    # and maps them to random fourier features vectors (stacked row by row, similar to the structure of X)
    np.random.seed(seed)
    if n_rff % 2 != 0:
        raise Warning('One can only generate an even number of random fourier features (cos + sin)')
    omega = unif_ort_gaussian((X.shape[1], n_rff / 2)) / scale
    PhiX = np.stack([np.cos(np.dot(X, omega)), np.sin(np.dot(X, omega))]) / np.sqrt(n_rff)
    return PhiX

"""
Generate Fourier features with a certain angle between all vectors (if dimension allows it) 
"""
def T_matrix(C):
    # Let C be of size (k,k). This function computes one upper triangular matrix T of size (k,k) such that T.T * T = C
    # if X is a (d,k) matrix (d>k) and X.T * X = eye(k) 
    # (e.g. X is a matrix of k orthonormal d-dimensional columns) then if U = XT it holds that U.T * U = C, e.g. the scalar product between
    # the i-th and j-th columns of U is exactly C_ij
    # Complexity: O(k^3), NOT vectorized. Considering the results when C = (1, c, c, ..., c \\ c, 1, c, c, ..., c \\ ... \\ c, c, ..., c, 1) it could be optimized for this particular case. 
    k = C.shape[0]
    T = np.zeros((k, k))
    T[0, 0] = np.sqrt(C[1, 1])
    for j in xrange(1, k):
        T[0, j] = C[0, j] / T[0, 0]
        for i in xrange(1, j):
            T[i, j] = (C[i, j] - np.dot(T[:(i), i].T, T[:(i), j])) / T[i, i]
        T[j, j] = np.sqrt(C[j,j] - np.dot(T[:, j], T[:, j].T))
        # if T[j, j] == np.nan:
        #     break
    
    return T
# C = 0.4 * np.ones((10, 10)) # 10 features
# np.fill_diagonal(C, 1)
# T = T_matrix(C)
# print np.dot(T.T, T)
# A = stacked_unif_ort_gaussian((7, 10)) # 10 7-dimensional orthogonal features
# print np.round(np.dot(A.T, A), decimals = 2)
# stacked_angled = np.dot(A, T)
# stacked_angled = stacked_angled * np.sqrt(np.random.chisquare(7, size = (10)))
# mat_norm = np.reshape(np.linalg.norm(stacked_angled, axis = 0), (10, 1))
# mat_norm = np.tile(mat_norm, [1, 10]) * np.tile(mat_norm.T, [10, 1])
# print np.round(mat_norm, decimals = 2)
# print np.round(np.divide(np.dot(stacked_angled.T, stacked_angled), mat_norm), decimals = 2)

# n_rff = 14
# angles = np.array([0, 0.2, 0.4, 0.5])
# C = np.eye(n_rff / 2)
# for j in range(1, n_rff / 4 + 1):
#     C += angles[j] * (np.diag(np.ones(n_rff / 2 - j), j) + np.diag(np.ones(n_rff / 2 - j), -j) + np.diag(np.ones(j), n_rff / 2 - j) + np.diag(np.ones(j), j - n_rff / 2))
# if n_rff % 4 == 0:
#     C -= np.diag(angles[n_rff / 4] * np.ones(n_rff / 4), n_rff / 4) + np.diag(angles[n_rff / 4] * np.ones(n_rff / 4), -n_rff / 4)
# print C

# C = (-1.0/3) * np.ones((4,4))
# np.fill_diagonal(C, 1)
# print T_matrix(C)

def angled_block(dim, scal_prod):
    # returns a dim x dim matrix of norm 1 columns having scalar products = scal_prod
    C = np.ones((dim, dim)) * scal_prod
    np.fill_diagonal(C, 1)
    angle_matrix = T_matrix(C)
    
    if np.isnan(angle_matrix[-1, -1]):
        max_d = np.min(np.argwhere(np.isnan(angle_matrix)).flatten()) # we can only draw that many vectors satisfying those angles
    else:
        max_d = angle_matrix.shape[0]
    max_d = min(max_d, dim)
    angle_matrix = angle_matrix[:, :max_d]

    ret = [np.dot(unif_ort_QR(dim), angle_matrix) for _ in range(int(np.ceil(float(dim) / max_d)))]
    ret = np.concatenate(ret, axis = 1)
    ret = ret[:, :dim]
    return ret

def angled_gaussian_RFF(X, n_rff, seed, scale, angle):
    # angle is between 0 and pi 
    # enforce that the random fourier frequencies satisfy: angle between X_i and X_{i+j} is equal to angle if possible. else stack such vectors
    np.random.seed(seed)
    
    omega = [angled_block(X.shape[1], np.cos(angle)) for _ in range(int(np.ceil(float(n_rff / 2) / X.shape[1])))]
    omega = np.concatenate(omega, axis = 1)
    omega = omega[:, :(n_rff / 2)]
    omega = np.multiply(np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff / 2))), omega / scale)

    PhiX = np.concatenate([np.cos(np.dot(X, omega)), np.sin(np.dot(X, omega))], axis = 1) / np.sqrt(n_rff)
    return PhiX

def sample_theta(m):
    # samples one theta with density = sin(theta)^(m-2) dtheta (e.g. the density of the angle between two random uniform vectors)
    # use acceptance-rejection sampling with a uniform as dominating density

    u = np.random.uniform()
    theta = np.pi * np.random.uniform()
    
    while u > np.sin(theta)**(m-2):
        u = np.random.uniform()
        theta = np.pi * np.random.uniform()
    
    return theta
# import matplotlib.pyplot as plt
# plt.hist([sample_theta(3) for _ in range(1000)], 30)
# plt.show()

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
    K = hadamard_rademacher_product_scale_chi(X, n_rff, k) / scale
    PhiX = np.stack([np.cos(K), np.sin(K)]) / np.sqrt(n_rff)
    return PhiX
