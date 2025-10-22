import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eig


def ravel_3D_matrix(img: np.ndarray) -> np.ndarray:
    return img.T.reshape((-1, img.shape[0] * img.shape[1]))


def get_noise_matrix(Y):
    """n_(b,p) = 1 / 2 (y_(b,p) - y_(b,p+1))"""
    # TODO: use better algorithm
    N = np.zeros_like(Y)
    N[:, 1:] = 1 / 2 * (Y[:, 0:-1] - Y[:, 1:])
    S = Y - N
    return S, N


def get_centered(Y):
    """remove mean in each row (channel)"""
    return Y - Y.mean(axis=1)[:, None]


def get_cov(X):
    return X @ X.T


def get_transformation_matrix(S, N):
    """
    need to solve Sigma_N a = lam Sigma a

    this is the generalized EV problem
        Ax = lam Bx where A =
    where
        A <==> Sigma_N
        B <===> Sigma
    """
    Sigma = get_cov(S)
    Sigma_N = get_cov(N)

    # noinspection PyTupleAssignmentBalance
    eig_vals, eig_vecs = eig(a=Sigma_N, b=Sigma)

    # eig_vecs is A_T, but need to ensure proper ordering by eigenvalues
    # so rows correspond to normalized eigenvectors
    o = np.argsort(np.abs(eig_vals))[::-1]
    A_T = eig_vecs[o, :]

    return A_T.T, eig_vals[o]


def reduce_dimensions(Z, A, r: int):
    n_bands = A.shape[0]

    # set r last bands to zero
    R = np.identity(n_bands)
    for i in range(r):
        i_p = n_bands - i - 1
        R[i_p, i_p] = 0

    Z_star = (np.linalg.inv(A).T @ R @ A.T).real @ Z
    return Z_star


def plot_eigenvals(w, ax: plt.Axes = None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.abs(w))
    ax.set_ylabel(r'$|\lambda_i|$')
    ax.set_xlabel('i')
    return ax


def run(img):
    print('transforming image to 2D ...')
    Y = ravel_3D_matrix(img)
    print('centering values ...')
    Z = get_centered(Y)
    print('estimating noise matrix ...')
    S, N = get_noise_matrix(Z)
    print('calculating transformation matrix ...')
    A, w = get_transformation_matrix(S, N)

    plt.figure()
    plot_eigenvals(w)
    plt.savefig('temp.pdf')
    plt.close()

    r_keep = int(input("number of channels to keep (check temp image):"))
    r = img.shape[2] - r_keep

    # keep only the first channel
    print('reducing dimensions ...')
    Z_star = reduce_dimensions(Z, A, r=r).real

    # reconstruct 2D image
    channel = Z_star[:, :].sum(axis=0).reshape(img.shape[:2]).T

    print('creating final plot')
    plt.imshow(channel)
    plt.savefig('temp.pdf')
    plt.close()
    print('done!')

    return A


if __name__ == '__main__':
    # import skimage

    # 3d test image
    # test = skimage.data.astronaut()

    # open calibrated data
    print('starting script ...')
    import h5py
    from test_iceland.paths import path_file_hdf5
    print('creating connection to hdf5 file')
    f = h5py.File(path_file_hdf5, 'r')
    print("loading calibrated data ...")
    img = f['roi_calibrated'][:, :, ::10]
    print("done loading data, starting MNR ...")

    print('transforming image to 2D ...')
    Y = ravel_3D_matrix(img)
    print('centering values ...')
    Z = get_centered(Y)
    print('estimating noise matrix ...')
    S, N = get_noise_matrix(Z)
    print('calculating transformation matrix ...')
    A, w = get_transformation_matrix(S, N)

    plt.figure()
    plot_eigenvals(w)
    plt.savefig('temp.pdf')
    plt.close()

    r_keep = int(input("number of channels to keep (check temp image):"))
    r = img.shape[2] - r_keep

    # keep only the first channel
    print('reducing dimensions ...')
    Z_star = reduce_dimensions(Z, A, r=r).real

    # reconstruct 2D image
    channel = Z_star[0, :].reshape(img.shape[:2]).T

    print('creating final plot')
    plt.imshow(channel)
    plt.savefig('temp.pdf')
    plt.close()
    print('done!')

    # run(test)
