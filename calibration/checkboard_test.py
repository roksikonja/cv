import glob
import os

import cv2 as cv
import numpy as np

from toolbox.visualizer import print_matrix, draw_markers
from toolbox.calibration import (
    to_homogeneous_coordinates,
    project_xyz,
    construct_P_matrix,
)


data_dir = "./data/checkerboard"
results_dir = "./results"

"""
    Following a tutorial at https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html.
    OpenCV calibration toolbox 
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a.
"""
cv.namedWindow("preview")
vc = cv.VideoCapture(0, cv.CAP_DSHOW)

img = None
if vc.isOpened():  # try to get the first frame
    rval, img = vc.read()
else:
    rval = False

grid_dims = (6, 6)
resize_fraction = 0.2

grid = np.zeros((grid_dims[0] * grid_dims[1], 3), np.float32)
grid[:, :2] = np.mgrid[0 : grid_dims[0], 0 : grid_dims[1]].T.reshape(-1, 2)

XYZ = []
xy = []

for image_path in glob.glob(os.path.join(data_dir, "*.jpg")):
    img = cv.imread(image_path)

    img = cv.resize(
        img,
        (int(img.shape[1] * resize_fraction), int(img.shape[0] * resize_fraction)),
        interpolation=cv.INTER_AREA,
    )
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    found, corners = cv.findChessboardCorners(gray, grid_dims, None)

    if found:
        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        XYZ.append(grid)
        xy.append(corners2)

        ret, K, dist, Rs, ts = cv.calibrateCamera(XYZ, xy, gray.shape[::-1], None, None)
        print(type(Rs), len(Rs))
        R, _ = cv.Rodrigues(Rs[0])
        t = ts[0]

        print(grid.shape)
        print_matrix(K, "K")
        print_matrix(dist, "dist")
        print_matrix(R, "R")
        print_matrix(np.linalg.inv(R) @ t, "t")

        # Camera undistortion
        height, width = img.shape[:2]
        K_, roi = cv.getOptimalNewCameraMatrix(
            K, dist, (width, height), 1, (width, height)
        )
        img_u = cv.undistort(img, K, dist, None, K_)

        P = construct_P_matrix(K, R, t.flatten())
        xy_grid = project_xyz(P, to_homogeneous_coordinates(grid.T))
        img_grid = draw_markers(
            img,
            xy_grid[:-1, :],
            color=(0, 255, 0),
            marker_type=cv.MARKER_CROSS,
            marker_size=10,
        )

        # Visualise
        cv.drawChessboardCorners(img, grid_dims, corners2, found)
        cv.imshow("preview", img)
        # cv.imshow("preview", img_grid)
    else:
        cv.imshow("preview", img)

    key = cv.waitKey(20)
    if key == 27 or key == ord("q"):  # exit on ESC
        break


cv.destroyAllWindows()
