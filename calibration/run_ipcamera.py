from urllib.request import urlopen
import time

import cv2 as cv
import numpy as np

from exercises.toolbox import to_homogeneous_coordinates, project_xyz, construct_P_matrix
from exercises.visualizer import draw_markers

data_dir = "./data/checkerboard"
results_dir = "./results"

"""
    Following a tutorial at https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html.
    OpenCV calibration toolbox 
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a.
"""
cv.namedWindow("ipcamera")
url = "http://192.168.2.187:8080/shot.jpg"
fps = 120

grid_dims = (6, 6)
resize_fraction = 0.4

grid = np.zeros((grid_dims[0] * grid_dims[1], 3), np.float32)
grid[:, :2] = np.mgrid[0: grid_dims[0], 0: grid_dims[1]].T.reshape(-1, 2)

while True:
    frame_start = time.time()
    response = urlopen(url)
    img_array: np.ndarray = np.array(bytearray(response.read()), dtype=np.uint8)
    img: np.ndarray = cv.imdecode(img_array, -1)
    img = np.rot90(img, 3)

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

        ret, K, dist, Rs, ts = cv.calibrateCamera([grid], [corners], gray.shape[::-1], None, None)

        # Visualise
        cv.drawChessboardCorners(img, grid_dims, corners2, found)
        cv.imshow("ipcamera", img)
    else:
        cv.imshow("ipcamera", img)

    frame_end = time.time()
    # key = cv.waitKey(int(1000 / fps))
    key = cv.waitKey(1)
    if key == 27 or key == ord("q"):  # exit on ESC
        break

cv.destroyAllWindows()
