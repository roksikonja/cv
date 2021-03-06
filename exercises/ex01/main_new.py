import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from toolbox.calibration import (
    to_homogeneous_coordinates,
    run_dlt,
    project_xyz,
    construct_P_matrix,
    make_grid,
    mean_reprojection_error,
    run_gold_standard,
)
from toolbox.visualizer import draw_markers, print_matrix

data_dir = "./data"
results_dir = "./results"

image_name = "image_new.jpg"

"""
    Load data.
    Convert to homogeneous coordinates.
"""
img = cv.imread(os.path.join(data_dir, image_name))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = np.rot90(img, 2)

m, n, _ = img.shape
data = scipy.io.loadmat(os.path.join(data_dir, "points_new.mat"))
xy = data["xy"]
XYZ = data["XYZ"]
xy_gs = xy[:, 6:]
XYZ_gs = XYZ[:, 6:]

print(f"img {str(img.shape)} {img.dtype} {img.min()}-{img.max()}")
print(f"xy {str(xy.shape)} {xy.dtype} XYZ {str(XYZ.shape)} {XYZ.dtype}")

xy = to_homogeneous_coordinates(xy)
XYZ = to_homogeneous_coordinates(XYZ)
print(f"xy {str(xy.shape)} {xy.dtype} XYZ {str(XYZ.shape)} {XYZ.dtype}")

xy_gs = to_homogeneous_coordinates(xy_gs)
XYZ_gs = to_homogeneous_coordinates(XYZ_gs)
print(f"xy_gs {str(xy_gs.shape)} {xy_gs.dtype} XYZ {str(XYZ_gs.shape)} {XYZ_gs.dtype}")

print_matrix(xy, "xy")
print_matrix(XYZ, "XYZ")

img_m = draw_markers(img, xy[:-1, :])

"""
    Direct Linear Transform.
"""
P, K, R, t, C, error = run_dlt(xy, XYZ)

print(f"reprojection error {error}")
print_matrix(C, "C")
print_matrix(P @ C, "P * C")

xy_p = project_xyz(construct_P_matrix(K, R, t), XYZ)
img_mp = draw_markers(
    img_m, xy_p[:-1, :], color=(255, 0, 0), marker_type=cv.MARKER_TILTED_CROSS
)

# Grid
X_max = 10
Y_max = 10
Z_max = 10

XYZ_grid = make_grid(X_max, Y_max, Z_max)
xy_grid = project_xyz(construct_P_matrix(K, R, t), XYZ_grid)

img_grid = draw_markers(
    img, xy_grid[:-1, :], color=(0, 255, 0), marker_type=cv.MARKER_CROSS
)

"""
    Gold Standard Algorithm.
"""
img_gs = draw_markers(
    img, xy_gs[:-1, :], color=(0, 255, 0), marker_type=cv.MARKER_CROSS
)

error = mean_reprojection_error(xy, XYZ, P)
print(f"reprojection error {error}")

P_gs, K_gs, R_gs, t_gs, C_gs, error_gs = run_gold_standard(xy, XYZ)
print(f"reprojection error {error_gs}")

print_matrix(P_gs, "P_gs")
print_matrix(K_gs, "K_gs")
print_matrix(R_gs, "R_gs")
print(f"det(R_gs) {np.linalg.det(R_gs)}")
print_matrix(-np.linalg.inv(R) @ t_gs, "t_gs")
print_matrix(C_gs, "C_gs")

xy_grid_gs = project_xyz(construct_P_matrix(K_gs, R_gs, t_gs), XYZ_grid)
img_grid_gs = draw_markers(
    img, xy_grid_gs[:-1, :], color=(0, 255, 0), marker_type=cv.MARKER_CROSS
)

""""
    Camera intrinsic parameters.
"""
c_x = K[0, -1]
c_y = K[1, -1]

f_px = K[0, 0]
f_py = K[1, 1]

px_py = f_py / f_px

alpha = np.arctan(K[0, 1] / f_py) * 180 / (2 * np.pi)

print(f"c_x {c_x} c_y {c_y} f_px {f_px} f_py {f_py} alpha {alpha} px_py {px_py}")

""""
    Plot.
"""
plt.figure()
plt.imshow(np.rot90(img, 3))
plt.show()

plt.figure()
plt.imshow(np.rot90(img_m, 3))
plt.show()

plt.figure()
plt.imshow(np.rot90(img_mp, 3))
plt.show()

plt.figure()
plt.imshow(np.rot90(img_grid, 3))
plt.show()

plt.figure()
plt.imshow(np.rot90(img_grid_gs, 3))
plt.show()
