import os

import numpy as np
import cv2 as cv
import scipy.io

import matplotlib.pyplot as plt

from exercises.visualizer import print_matrix

data_dir = "../exercises/ex01/data"
results_dir = "./results"

"""
    Following a tutorial at https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html.
    OpenCV calibration toolbox 
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a.
"""

"""
    Load data.
    Convert to homogeneous coordinates.
"""
img = cv.imread(os.path.join(data_dir, "calibration_image.jpg"))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
m, n, _ = img.shape

data = scipy.io.loadmat(os.path.join(data_dir, "points_acc.mat"))
xy = data["xy"]
XYZ = data["XYZ"]
data = scipy.io.loadmat(os.path.join(data_dir, "points_ag.mat"))
xy_gs = data["xy_ag"]
XYZ_gs = data["XYZ_ag"]

xy = np.hstack((xy, xy_gs)).astype(np.float32).T
xy = xy[:, np.newaxis, :]  # (n, 1, 2)
XYZ = np.hstack((XYZ, XYZ_gs)).astype(np.float32).T  # (n, 3)

print(f"img {str(img.shape)} {img.dtype} {img.min()}-{img.max()}")
print(f"gray {str(gray.shape)} {gray.dtype} {gray.min()}-{gray.max()}")
print(f"xy {str(xy.shape)} {xy.dtype} XYZ {str(XYZ.shape)} {XYZ.dtype}")

# print_matrix(xy, "xy")
# print_matrix(XYZ, "XYZ")

plt.figure()
plt.imshow(img)
plt.show()


# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
#     objectPoints=[XYZ],
#     imagePoints=[xy],
#     imageSize=gray.shape[::-1],
#     cameraMatrix=None,
#     distCoeffs=None,
#     rvecs=None,
#     tvecs=None,
#     flags=cv.CALIB_USE_INTRINSIC_GUESS
# )
