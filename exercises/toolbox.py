import numpy as np
import scipy.optimize


def to_homogeneous_coordinates(matrix):
    hmatrix = np.ones(shape=(matrix.shape[0] + 1, matrix.shape[1]))
    hmatrix[:-1, :] = matrix

    return hmatrix


def normalization(xy, xyz):
    n = xy.shape[-1]

    xy_centroid = np.mean(xy[:-1, :], axis=-1)
    xyz_centroid = np.mean(xyz[:-1, :], axis=-1)

    tmp = np.ones_like(xy)
    tmp[0, :] = tmp[0, :] * xy_centroid[0]
    tmp[1, :] = tmp[1, :] * xy_centroid[1]
    tmp[-1, :] = 0
    xy_n = xy - tmp

    tmp = np.ones_like(xyz)
    tmp[0, :] = tmp[0, :] * xyz_centroid[0]
    tmp[1, :] = tmp[1, :] * xyz_centroid[1]
    tmp[2, :] = tmp[2, :] * xyz_centroid[2]
    tmp[-1, :] = 0
    xyz_n = xyz - tmp

    xy_scale = np.sum(np.sqrt(np.sum(np.square(xy_n[:-1, :]), axis=0)))
    xy_scale = n * np.sqrt(2) / xy_scale

    xyz_scale = np.sum(np.sqrt(np.sum(np.square(xyz_n[:-1, :]), axis=0)))
    xyz_scale = n * np.sqrt(3) / xyz_scale

    T = np.array([[1, 0, -xy_centroid[0]], [0, 1, -xy_centroid[1]], [0, 0, 1]])
    T[:-1, :] = xy_scale * T[:-1, :]

    U = np.array(
        [
            [1, 0, 0, -xyz_centroid[0]],
            [0, 1, 0, -xyz_centroid[1]],
            [0, 0, 1, -xyz_centroid[2]],
            [0, 0, 0, 1],
        ]
    )
    U[:-1, :] = xyz_scale * U[:-1, :]

    xy_n = T @ xy
    xyz_n = U @ xyz

    return xy_n, xyz_n, T, U


def project_xyz(P, xyz):
    xy = P @ xyz

    xy = xy / xy[-1, :]
    return xy


def mean_reprojection_error(xy, xyz, P):
    xyp = project_xyz(P, xyz)

    error_vec = np.linalg.norm(xyp - xy, axis=0)

    error = np.mean(error_vec)
    return error


def dlt(xy, xyz):
    # Hardcoded! Only works for 6 point correspondences.
    M = np.zeros((18, 18))

    for i in range(6):
        X = xyz[:, i]
        x = xy[:, i]

        for j in range(3):
            M[i * 3 + j, (j * 4): ((j + 1) * 4)] = X
            M[i * 3 + j, 12 + i] = x[j]

    _, _, V = np.linalg.svd(M)
    V = np.conjugate(V.T)

    p = V[:12, -1]
    P = np.reshape(p, newshape=(3, 4))

    return P


def decompose(P):
    M = P[:, :-1]
    Qt, Rt = np.linalg.qr(np.linalg.inv(M))

    K = np.linalg.inv(Rt)
    R = np.linalg.inv(Qt)
    t = np.linalg.solve(K, P[:, -1])

    K = K / K[-1, -1]

    return K, R, t


def run_dlt(xy, xyz):
    xy_n, xyz_n, T, U = normalization(xy, xyz)

    Pn = dlt(xy_n, xyz_n)

    P = np.linalg.inv(T) @ Pn @ U
    P = P / P[-1, -1]

    K, R, t = decompose(P)

    error = mean_reprojection_error(xy, xyz, P)

    _, _, V = np.linalg.svd(P)
    V = np.conjugate(V.T)
    C = V[:, -1]
    C = C / C[-1]

    return P, K, R, t, C, error


def construct_P_matrix(K, R, t):
    P = np.zeros((3, 4))
    P[:3, :3] = R
    P[:, -1] = t
    P = K @ P

    return P


def make_grid(x_max, y_max, z_max):
    xyz_grid = np.zeros((3, (z_max + 1) * (x_max + y_max + 1) + (x_max * y_max)))

    i = 0
    for z in range(z_max + 1):
        for x in range(x_max + 1):
            xyz_grid[:, i] = np.array([x, 0, z])
            i = i + 1

        for y in range(1, y_max + 1):
            xyz_grid[:, i] = np.array([0, y, z])
            i = i + 1

    for x in range(1, x_max + 1):
        for y in range(1, y_max + 1):
            xyz_grid[:, i] = np.array([x, y, 0])
            i = i + 1

    return to_homogeneous_coordinates(xyz_grid)


def fmin_gold_standard(p, xy, xyz):
    P = np.reshape(p, (3, 4))
    return mean_reprojection_error(xy, xyz, P)


def run_gold_standard(xy, xyz, max_iter=500):
    xy_n, xyz_n, T, U = normalization(xy, xyz)

    Pn = dlt(xy_n, xyz_n)

    pn = Pn.flatten()
    result = scipy.optimize.fmin(
        func=fmin_gold_standard,
        args=(xy_n, xyz_n),
        x0=pn,
        maxiter=max_iter,
        maxfun=100000,
        disp=True,
    )

    pn = result
    Pn = np.reshape(pn, (3, 4))

    P = np.linalg.inv(T) @ Pn @ U
    P = P / P[-1, -1]

    K, R, t = decompose(P)

    error = mean_reprojection_error(xy, xyz, P)

    _, _, V = np.linalg.svd(P)
    V = np.conjugate(V.T)
    C = V[:, -1]
    C = C / C[-1]

    return P, K, R, t, C, error
