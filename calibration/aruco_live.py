import cv2 as cv
import numpy as np

from toolbox.calibration import (
    run_gold_standard,
    to_homogeneous_coordinates,
    project_xyz,
)
from toolbox.input import IPCamera
from toolbox.markers import ArucoMarkers
from toolbox.visualizer import print_matrix, draw_markers

reader = IPCamera(url="http://192.168.2.187:8080/shot.jpg", fps=20)
aruco = ArucoMarkers(grid_cm=20, marker_cm=2)

while True:
    frame = reader.get_frame(fraction=0.4, rot=3)
    # frame = resize(cv.imread("./data/aruco.jpg"), 0.3)

    corners, ids = aruco.detect_markers(frame, verbose=False)

    xy, XYZ = aruco.get_positions_by_id(corners, ids)

    print(xy.shape)

    if xy.shape[1] > 6:
        P, _, R, t, _, error = run_gold_standard(
            to_homogeneous_coordinates(xy), to_homogeneous_coordinates(XYZ), 2000
        )

        frame = draw_markers(
            frame,
            project_xyz(P, to_homogeneous_coordinates(XYZ))[:-1, :],
            color=(0, 0, 255),
            marker_type=cv.MARKER_TILTED_CROSS,
            marker_size=10,
        )
        print_matrix(-np.linalg.inv(R) @ t, "t")

    frame = aruco.draw_markers(frame, corners, ids)

    key = reader.show_frame(frame)
    if key == 27 or key == ord("q"):
        break

cv.destroyAllWindows()
