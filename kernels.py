import math
import numpy as np
import scipy.stats as sp_stats
import scipy.optimize as sp_opt
import scipy.special as sp_spec
from collections import defaultdict

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

def make_antithetic(feats):
    return np.concatenate([feats, np.conj(feats)], axis = 1) / np.sqrt(2)

"""
random Fourier features iid
"""
def iid_gaussian_RFF(X, n_rff, seed, scale):
    # returns the matrix (exp(j<omega_{1}, X>), ..., exp(j<omega_{n_rff}, X>) / sqrt(n_rff)
    # where omega_{i} are i.i.d. N(0, 1/scale^2)
    np.random.seed(seed)
    omega = np.random.normal(loc = 0.0, scale = 1.0 / scale, size = (X.shape[1], n_rff)) # random frequencies
    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
    return PhiX

"""
attempt at pairing frequencies so that we get a factor of e^{+- ipi} in the next feature
"""
def approx_antithetic_RFF(X, n_rff, seed, scale, main_axis):
    # main_axis given as a row (1, dim) vector
    np.random.seed(seed)
    omega = np.random.normal(loc = 0.0, scale = 1.0, size = (X.shape[1], n_rff / 2)) # each column is a random frequency
    omega = omega / np.linalg.norm(omega, axis = 0)
    omega2 = omega - 2 * np.dot(main_axis, omega) * main_axis[:, np.newaxis]
    
    norms = np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff / 2)))
    omega *= norms
    omega2 *= norms
    omega = np.concatenate([omega, omega2], axis = 1)

    PhiX = np.exp(1j * np.dot(X, omega) / scale) / np.sqrt(n_rff)
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
    # G = G[:, :shape[1]]
    idx = np.random.choice(G.shape[1], size = shape[1], replace = False)
    G = G[:, idx]
    return G
# A = stacked_unif_ort((3, 7))
# print np.round(np.dot(A.T, A), decimals = 2)

def unif_ort_gaussian(shape): 
    # generates a matrix with shape[1] orthogonal columns of dimension shape[0], each marginally gaussian (the columns are the sampled frequencies)
    G = stacked_unif_ort(shape)
    norms = np.sqrt(np.random.chisquare(df = shape[0], size = (1, shape[1])))
    # print G, norms, np.multiply(norms, G)
    G = np.multiply(norms, G)
    return G
# unif_ort_gaussian((3, 2))

def ort_gaussian_RFF(X, n_rff, seed, scale):
    # generates n_rff orthogonal frequencies of dimension X.shape[1] (e.g. omega is of shape (X.shape[1], n_rff))
    # and maps them to random fourier features vectors (stacked row by row, similar to the structure of X)
    np.random.seed(seed)
    omega = unif_ort_gaussian((X.shape[1], n_rff)) / scale
    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
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
        T[j, j] = np.sqrt(C[j,j] - np.dot(T[:, j], T[:, j].T)) # for normalization purposes
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
    # print 'can only enforce %d relations for scalar product %f' % (max_d, scal_prod)
    angle_matrix = angle_matrix[:, :max_d]

    ret = [np.dot(unif_ort_QR(dim), angle_matrix) for _ in range(int(np.ceil(float(dim) / max_d)))]
    ret = np.concatenate(ret, axis = 1)
    ret = ret[:, :dim]
    return ret

def angled_gaussian_RFF(X, n_rff, seed, scale, angle):
    # angle is between 0 and pi 
    # enforce that the random fourier frequencies satisfy: angle between X_i and X_{i+j} is equal to angle if possible. else stack such vectors
    np.random.seed(seed)
    
    omega = [angled_block(X.shape[1], np.cos(angle)) for _ in range(int(np.ceil(float(n_rff) / X.shape[1])))]
    omega = np.concatenate(omega, axis = 1)
    omega = omega[:, :n_rff]
    omega = np.multiply(np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff))), omega / scale)

    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
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

def spherical_coord(angles):
    v = np.ones(angles.shape[0]+1)
    for idx in range(angles.shape[0]):
        v[idx]    *= np.cos(angles[idx])
        v[idx+1:] *= np.sin(angles[idx])
    return v
# print spherical_coord(np.array([0,0,0]))
# print spherical_coord(np.array([np.pi / 2,0,0]))

def angled_neighbours(dim, n_samples, angle):
    ret = np.zeros((dim, n_samples))
    ret[:, 0] = np.random.normal(size = dim)
    ret[:, 0] /= np.linalg.norm(ret[:, 0])

    scalar_product = np.cos(angle)

    for idx in range(1, n_samples):
        ret[:, idx] = np.random.normal(size = dim)
        ret[:, idx] /= np.linalg.norm(ret[:, idx])
        b = np.sqrt((1.0 - scalar_product ** 2) / (1.0 - np.dot(ret[:, idx - 1].T, ret[:, idx]) ** 2))
        a = scalar_product - b * np.dot(ret[:, idx - 1].T, ret[:, idx])
        ret[:, idx] = a * ret[:, idx - 1] + b * ret[:, idx]

    return ret
# A = angled_neighbours(5, 10, np.pi / 2 * 1.3)
# print np.round(np.dot(A.T, A), decimals = 2)
# print np.mean(np.dot(A.T, A)), 1.0 / A.shape[1]
# import matplotlib.pyplot as plt
# plt.hist(np.reshape((np.dot(A.T, A)),  -1), 100)
# plt.show()

def angled_gaussian_neighbour_RFF(X, n_rff, seed, scale, angle):
    np.random.seed(seed)
    
    omega = angled_neighbours(X.shape[1], n_rff, angle)
    omega = np.multiply(np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff))), omega / scale)

    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
    return PhiX

"""
greedy approach to generate samples by taking directions as far from each other as possible
"""
# TODO: use singleton so that one only needs to do this once
def greedy_unif_directions(dim, n_rff):
    # samples fourier n_rff dim-dimensional frequencies which are as far from each other as possible
    def greedy_objective_fn(points, new_angle):
        return -np.mean(np.linalg.norm(points.T - spherical_coord(new_angle), axis = 1))

    ret = np.zeros((dim, n_rff))
    ret[0, 0] = 1
    for i in range(1, n_rff):
        optimal = sp_opt.minimize(fun = lambda x: greedy_objective_fn(ret[:, :i], x), 
                                  x0 = np.pi * 2 * np.random.rand(dim - 1), 
                                  bounds = [(0, 2 * np.pi)]*(dim-1), 
                                  tol = 1e-5)
        ret[:, i] = spherical_coord(optimal.x)

    return ret
# greedy_unif_directions(10, 64)
# A = greedy_unif_directions(5, 64)
# print np.linalg.norm(A, axis = 0)
# import matplotlib.pyplot as plt
# plt.hist(np.reshape((np.dot(A.T, A)),  -1), 100)
# plt.show()

def greedy_unif_gaussian_RFF(X, n_rff, seed, scale):
    np.random.seed(seed)
    
    omega = np.dot(unif_ort_QR(X.shape[1]), greedy_unif_directions(X.shape[1], n_rff))
    omega = np.multiply(np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff))), omega / scale)

    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
    return PhiX

"""
greedy approach to generate samples
"""
def greedy_directions(dim, n_rff):
    # samples fourier n_rff dim-dimensional frequencies which are as far from each other as possible
    def greedy_objective_fn_l(points, l, new_angle):
        return -np.mean(np.linalg.norm(points.T - l * spherical_coord(new_angle), axis = 1))

    ret = np.zeros((dim, n_rff))
    ret[0, 0] = np.sqrt(np.random.chisquare(df = dim))
    for i in range(1, n_rff):
        l = np.sqrt(np.random.chisquare(df = dim))
        optimal = sp_opt.minimize(fun = lambda x: greedy_objective_fn_l(ret[:, :i], l, x), 
                                  x0 = 2.0 * np.pi * np.random.rand(dim - 1), 
                                  bounds = [(0, 2 * np.pi)]*(dim-1), 
                                  tol = 1e-5)
        ret[:, i] = l * spherical_coord(optimal.x)
    return ret
# greedy_directions(10, 64)

def greedy_dir_gaussian_RFF(X, n_rff, seed, scale):
    np.random.seed(seed)
    
    omega = np.dot(unif_ort_QR(X.shape[1]), greedy_directions(X.shape[1], n_rff))
    omega = omega / scale

    PhiX = np.exp(1j * np.dot(X, omega)) / np.sqrt(n_rff)
    return PhiX

"""
Givens approach to generate samples with a certain angles
"""
def givens_angled_directions(dim, n_rff, seed, angle):
    ret = np.zeros((dim, n_rff))
    np.random.seed(seed)
    # ret[:, 0] = np.random.normal(size = dim)
    ret[:, 0] = ret[:, 0] / np.linalg.norm(ret[:, 0])
    ret[:, 0] = np.ones(dim) / np.sqrt(dim)
    print ret[:, 0].shape, n_rff
    ret = np.tile(ret[:, 0, np.newaxis], (1, n_rff))
    print ret.shape
    # ret = np.tile(np.eye(dim), (1, int(1.0 + float(n_rff) / dim)))[:, 0:n_rff]
    for i in range(1, n_rff):
        idx, jdx = np.random.choice(np.arange(dim), size = 2, replace = False)
        ret[:, i] = ret[:, i-1]
        R_mat = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ret[[idx, jdx], i] = np.dot(R_mat, ret[[idx, jdx], i])
    return ret
# A = givens_angled_directions(20, 30, 0, np.arccos(-0.4))
# A = np.divide(A, np.linalg.norm(A, axis = 0))
# print np.linalg.norm(A, axis = 0)
# print np.mean(np.dot(A.T, A))
# import matplotlib.pyplot as plt
# plt.hist(np.reshape((np.dot(A.T, A)),  -1), 100)
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
# A = hadamard_product(np.eye(16))
# A[0, :] *= -0.4
# A = A[:12, :]
# A = np.divide(A, np.linalg.norm(A, axis = 0))
# print np.linalg.norm(A, axis = 0)
# print np.mean(np.dot(A.T, A))
# print np.round(np.dot(A.T, A), 2)
# import matplotlib.pyplot as plt
# plt.hist(np.reshape((np.dot(A.T, A)),  -1), 100)
# plt.show()

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

def stack_power_hadamard_rademacher(x, n_blocks):
    # returns (I*x, HD*x, HDHD*x, ..., HDHD...HD*x)^T 
    ret = np.zeros((x.shape[0] * n_blocks, x.shape[1]))
    ret[0:x.shape[0], :] = x
    for r in range(1, n_blocks):
        ret[(r * x.shape[0]):((r+1) * x.shape[0]), :] = hadamard_rademacher_product(ret[((r-1) * x.shape[0]):(r * x.shape[0]), :], 1)
    return ret
# print stack_power_hadamard_rademacher(np.eye(8), 10)

def embed_in_power_of_two(X): # embeds X in (X.shape[0], 2^l) where 2^l is the next highest power of two. Embeds the rows!
    original_dimension = X.shape[1]
    HD_dim = 2 ** (int(np.ceil(np.log(original_dimension) / np.log(2)))) # the smallest power of 2 >= x.shape[1]. We embed X in (X.shape[0], R^{HD_dim}) by zero padding
    
    # newX = np.zeros((X.shape[0], HD_dim))
    # indices = np.random.choice(HD_dim, size = (original_dimension), replace = False)
    # newX[:, indices] = X

    newX = np.pad(X, ((0, 0), (0, HD_dim - X.shape[1])), 'constant', constant_values = ((np.nan, np.nan), (np.nan, 0)))

    return newX

def stacked_hadamard_rademacher(X, n_rff, k):
    # returns the product of X.T with orthogonal vectors, approximately of unit length. 
    original_dimension = X.shape[1] # dimension of input feature vector
    # HD_dim = 2 ** (int(np.ceil(np.log(original_dimension) / np.log(2)))) # the smallest power of 2 >= x.shape[1]. We embed X in (X.shape[0], R^{HD_dim}) by zero padding
    # X = np.pad(X, ((0,0), (0,HD_dim-original_dimension)), 'constant', constant_values = ((np.nan, np.nan), (np.nan, 0)))
    # newX = np.zeros((X.shape[0], HD_dim))
    # indices = np.random.choice(HD_dim, size = (original_dimension), replace = False)
    # newX[:, indices] = X
    X = embed_in_power_of_two(X)
    HD_dim = X.shape[1]

    # The output of hadamard_rademacher_product is a (HD_dim, x.shape[1]) matrix. We stack ceil(n_rff/HD_dim) of those to get a (ceil(n_rff/HD_dim)*HD_dim, x.shape[1]) matrix
    K = np.concatenate([hadamard_rademacher_product(X.T, k) for _ in range(int(np.ceil(float(n_rff) / HD_dim)))]).T
    # K is now of shape (K.shape[0], ceil(n_rff / HD_dim) * HD_dim)
    del X
    # Then we discard some columns from the last block
    # idx_last_block = int(np.floor(float(n_rff) / HD_dim)) * HD_dim
    # idx = np.random.choice(HD_dim, size = n_rff - idx_last_block, replace = False) # those indices of the last block we'll keep
    # K = np.concatenate([K[:, 0:idx_last_block], K[:, idx_last_block + idx]], axis = 1)

    idx = np.random.choice(K.shape[1], size = n_rff, replace = False)
    K = K[:, idx]

    # Scale all rows independently so that they're approximately unit length
    K *= np.sqrt(float(HD_dim) / original_dimension)
    return K 
# print stacked_hadamard_rademacher(np.eye(7), 3, 1)

def hadamard_rademacher_product_scale_chi(X, n_rff, k):
    # Returns the product of x with orthogonal vectors, each having an approximately Gaussian marginal

    K = stacked_hadamard_rademacher(X, n_rff, k)

    norms = np.sqrt(np.random.chisquare(df = X.shape[1], size = (1, n_rff)))
    K = norms * K
    
    return K

def HD_gaussian_RFF(X, n_rff, seed, scale, k):
    np.random.seed(seed)
    K = hadamard_rademacher_product_scale_chi(X, n_rff, k) / scale
    PhiX = np.exp(1j * K) / np.sqrt(n_rff)
    return PhiX

"""
fastfood
"""
def fastfood_prod(x):
    """ 
    x must have shape (dimension, n_vec) (e.g. samples are columns) 
    dimension must be a power of 2
    returns SHGPiHBx / sqrt(d)
    """
    d = x.shape[0]
    B = -1.0 + 2.0 * np.random.binomial(1, 0.5, size = (d, 1))
    P = np.arange(d)
    np.random.shuffle(P)
    G = np.random.normal(size = (d, 1))
    S = np.sqrt(np.random.chisquare(df = d, size = (d, 1))) / np.linalg.norm(G)
    
    K = np.multiply(B, x) # B * X
    K = hadamard_product(K) # H * B*X
    K = K[P, :] # P * H*B*K
    K = np.multiply(G, K) # G * P*H*B*K
    K = hadamard_product(K) # H * G*P*H*B*K
    K = np.multiply(S, K) # S * H*G*P*H*B*K
    K = K / np.sqrt(d) # TODO: check that divide by sqrt(HD_dim) and not sqrt(original_dimension)?
    return K

def fastfood_RFF(X, n_rff, seed, scale):
    """ X must have shape (n_vec, dimension) (e.g. samples are ROWS) """
    np.random.seed(seed)

    # Embed rows of X in dimension 2^k = HD_dim
    original_dimension = X.shape[1] # dimension of input feature vector
    X = embed_in_power_of_two(X)
    HD_dim = X.shape[1]
    # HD_dim = 2 ** (int(np.ceil(np.log(X.shape[1]) / np.log(2)))) # the smallest power of 2 >= x.shape[1]. We embed X in (X.shape[0], R^{HD_dim}) by zero padding
    # X = np.pad(X, ((0, 0), (0, HD_dim - X.shape[1])), 'constant', constant_values = ((np.nan, np.nan), (np.nan, 0)))

    K = np.concatenate([fastfood_prod(X.T) for _ in range(int(np.ceil(float(n_rff) / HD_dim)))], axis = 0).T
    
    # Then we discard some columns from the last block
    idx_last_block = int(np.floor(float(n_rff) / HD_dim)) * HD_dim
    idx = np.random.choice(HD_dim, size = n_rff - idx_last_block, replace = False) # those indices of the last block we'll keep
    
    K = np.concatenate([K[:, 0:idx_last_block], K[:, idx_last_block + idx]], axis = 1)
    
    K = K / scale * np.sqrt(HD_dim) / np.sqrt(original_dimension)

    return np.exp(1j * K) / np.sqrt(n_rff)


#TODO: abstract all those in 1. a function mapping matrix np.dot(X, freq) to full PhiX
"""
polynomial kernel using unit length random projections
"""
def iid_polynomial_sp_random_features(X, n_features, seed, degree, inhom_term = 0):
    """
    generates features for feature matrix X and kernel (<x,y>+inhom_term)^degree
    """

    if inhom_term != 0:
        X = np.pad(X, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))

    np.random.seed(seed)
    iid_freq = np.random.normal(size = (X.shape[1], n_features * degree))
    PhiX = np.dot(X, iid_freq)
    PhiX = [np.matrix(np.prod(PhiX[:, l::n_features], axis = 1)).T for l in range(n_features)]
    PhiX = np.concatenate(PhiX, axis = 1) / np.sqrt(n_features)
    
    return PhiX
# Phi = iid_polynomial_sp_random_features(np.eye(4), 10000, 0, 1, inhom_term = 0)
# print np.dot(Phi, Phi.T)


def iid_polynomial_sp_random_unit_features(X, n_features, seed, degree, inhom_term = 0):
    """
    generates features for feature matrix X and kernel (<x,y>+inhom_term)^degree
    """
    if inhom_term != 0:
        X = np.pad(X, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    
    dim = X.shape[1]

    np.random.seed(seed)
    iid_freq = np.random.normal(size = (dim, n_features * degree))
    iid_freq /= np.linalg.norm(iid_freq, axis = 0)[np.newaxis, :]
    iid_freq *= np.sqrt(dim)
    PhiX = np.dot(X, iid_freq)
    PhiX = [np.matrix(np.prod(PhiX[:, l::n_features], axis = 1)).T for l in range(n_features)]
    PhiX = np.concatenate(PhiX, axis = 1) / np.sqrt(n_features)

    return PhiX

def ort_polynomial_sp_random_unit_features(X, n_features, seed, degree, inhom_term = 0):
    """
    generates features for feature matrix X and kernel (<x,y>+inhom_term)^degree using random orthonormal projections
    """
    if inhom_term != 0:
        X = np.pad(X, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    
    dim = X.shape[1]

    np.random.seed(seed)
    ort_freq = np.concatenate([stacked_unif_ort((dim, n_features)) for _ in range(degree)], axis = 1)
    PhiX = np.dot(X, ort_freq) * np.sqrt(dim)
    PhiX = [np.matrix(np.prod(PhiX[:, l::n_features], axis = 1)).T for l in range(n_features)]
    PhiX = np.concatenate(PhiX, axis = 1) / np.sqrt(n_features)

    return PhiX

def ort_polynomial_sp_random_gaussian_features(X, n_features, seed, degree, inhom_term = 0):
    """
    generates features for feature matrix X and kernel (<x,y>+inhom_term)^degree using random orthonormal projections
    """
    if inhom_term != 0:
        X = np.pad(X, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    
    dim = X.shape[1]

    np.random.seed(seed)
    ort_freq = np.concatenate([stacked_unif_ort((dim, n_features)) for _ in range(degree)], axis = 1)
    ort_freq *= np.sqrt(np.random.chisquare(df = dim, size = (1, degree * n_features)))
    PhiX = np.dot(X, ort_freq)# * np.sqrt(dim)
    PhiX = [np.matrix(np.prod(PhiX[:, l::n_features], axis = 1)).T for l in range(n_features)]
    PhiX = np.concatenate(PhiX, axis = 1) / np.sqrt(n_features)

    return PhiX


def HD_polynomial_sp_random_unit_features(X, n_features, seed, degree, inhom_term = 0):
    """
    generates features for feature matrix X and kernel (<x,y>+inhom_term)^degree using random HD projections
    """
    if inhom_term != 0:
        X = np.pad(X, (((0,0), (1,0))), 'constant', constant_values = ((np.nan, np.nan), (np.sqrt(inhom_term), np.nan)))
    
    dim = X.shape[1]

    np.random.seed(seed)
    HD_freq = np.concatenate([stacked_hadamard_rademacher(np.eye(dim), n_features, 1) for _ in range(degree)], axis = 1)
    PhiX = np.dot(X, HD_freq) * np.sqrt(dim)
    PhiX = [np.matrix(np.prod(PhiX[:, l::n_features], axis = 1)).T for l in range(n_features)]
    PhiX = np.concatenate(PhiX, axis = 1) / np.sqrt(n_features)
    
    return PhiX

# """
# scalar product based kernels
# """
# def iid_scalar_prod_random_features(X, n_features, mclaurin_coeff):
#     """
#     mclaurin_coeff must be function handle of an integer n>=0
#     """
#     p = 0.5
#     random_features = np.zeros((X.shape[0], n_features))
#     N_mclaurin = np.random.geometric(p = p, size = n_features) - 1 # point until which we compute the mclaurin expansion
#     for idx in range(n_features):
#         if N_mclaurin[idx] > 0:
#             Omega = -1 + 2 * np.random.binomial(1, 0.5, size = (X.shape[1], N_mclaurin[idx]))
#             random_features[:, idx] = np.squeeze(np.prod(np.dot(X, Omega), axis = 1))
#         else:
#             random_features[:, idx] = np.ones(shape = X.shape[0])
#         
#         random_features[:, idx] *= np.sqrt(mclaurin_coeff(N_mclaurin[idx]) * p**(N_mclaurin[idx] + 1)) 
#     return random_features / np.sqrt(n_features)
# 
# def HD_scalar_prod_random_features(X, n_features, mclaurin_coeff):
#     p = 0.5
#     random_features = np.zeros((X.shape[0], n_features))
#     N_mclaurin = np.random.geometric(p = p, size = n_features) - 1# point until which we compute the mclaurin expansion
#     for idx in range(n_features):
#         if N_mclaurin[idx] > 0:
#             random_features[:, idx] = np.prod(X.shape[1] * stacked_hadamard_rademacher(X, N_mclaurin[idx], 1), axis = 1)
#         else:
#             random_features[:, idx] = np.ones(shape = (X.shape[0], 1))
#         random_features[:, idx] = np.sqrt(mclaurin_coeff(N_mclaurin[idx]) * p**(N_mclaurin[idx] + 1)) * random_features[:, idx]
#     return random_features / np.sqrt(n_features)

"""
iid polynomial kernel
"""
def polynomial_sp_kernel(X, loc, degree, inhom_term):
    return np.power(np.dot(X, loc.T) + inhom_term, degree)

# def mclaurin_coeff_generic(k, degree, inhom_term):
#     if k > degree: 
#         return 0
#     else:
#         return inhom_term ** (degree - k) * sp_spec.comb(degree, k, exact = False)

# def iid_polynomial_sp_random_features(X, n_features, seed, degree, inhom_term):    
#     np.random.seed(seed)
#     return iid_scalar_prod_random_features(X, n_features, lambda k: mclaurin_coeff_generic(k, degree, inhom_term))

# def HD_polynomial_sp_random_features(X, n_features, seed, degree, inhom_term):
#     np.random.seed(seed)
#     return HD_scalar_prod_random_features(X, n_features, lambda k: mclaurin_coeff_generic(k, degree, inhom_term))

"""
exponential kernel
"""
def exponential_sp_kernel(X, loc, scale):
    return np.exp(np.dot(X, loc.T) / scale ** 2)

# def iid_exponential_sp_random_features(X, n_rff, seed, scale):
#     def mclaurin_coeff(k):
#         return 1.0 / math.factorial(k) / scale ** (2 * k)
#     np.random.seed(seed)
#     return iid_scalar_prod_random_features(X, n_rff, mclaurin_coeff)