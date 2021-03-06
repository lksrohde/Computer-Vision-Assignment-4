import matplotlib.pyplot as plt
import numpy as np


def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2, 2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """
    u, s, v_trans = np.linalg.svd(A)
    s[-1] = 0
    A_hat = u @ np.diag(s) @ v_trans
    return A_hat


def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    n = np.shape(p1)[0]

    A = np.zeros((n, 9))
    for i, row in enumerate(p1):
        x_p1 = row[0]
        y_p1 = row[1]

        x_p2 = p2[i][0]
        y_p2 = p2[i][1]
        A[i] = ([x_p1 * x_p2, y_p1 * x_p2, x_p2, x_p1 * y_p2, y_p1 * y_p2, y_p2, x_p1, y_p1, 1])

    s, v, d = np.linalg.svd(A)
    D = np.transpose(d[-1:, ])
    return enforce_rank2(D.reshape((3, 3)))


def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """
    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)

    fund = compute_fundamental(ps1, ps2)

    F = np.transpose(T2) @ fund @ T1
    return F

def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """
    X1, X2, Y1, Y2 = [], [], [], []

    for i in range(p1.shape[0]):
        l = F @ np.append(p1[i],1)

        x1 = 0
        y1 = -l[2]/l[1]

        x2 = img.shape[1]
        y2 = -(l[2] + l[0]*x2)/l[1]

        X1.append(x1)
        X2.append(x2)
        Y1.append(y1)
        Y2.append(y2)

    return np.array(X1), np.array(X2), np.array(Y1), np.array(Y2)




def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    max_residual = 0
    avg_residual = 0

    for i in range(p1.shape[0]):
        abs_residual = np.linalg.norm(p1[i].T @ F @ p2[i])
        max_residual = max(max_residual, abs_residual)
        avg_residual += abs_residual
    
    avg_residual /= p1.shape[0]

    return max_residual, avg_residual


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    _,_, v = np.linalg.svd(F)
    e1 = v[:, 2]
    e2 = v[:, -1]

    return e1, e2