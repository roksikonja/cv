import cv2 as cv
import numpy as np

from toolbox.visualizer import print_matrix, draw_markers


class ArucoMarkers(object):
    def __init__(
        self,
        grid_cm,
        marker_cm,
        dictionary=cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250),
        parameters=cv.aruco.DetectorParameters_create(),
    ):
        self.markers = dict()

        self.marker_dictionary = dictionary
        self.detect_parameters = parameters

        # (X, Y, Z) position of marker's upper-left corner

        # xy-plane
        self.markers["xy"] = [
            {"id": 0, "X_cm": marker_cm, "Y_cm": marker_cm, "Z_cm": 0},
            {"id": 1, "X_cm": grid_cm - 2 * marker_cm, "Y_cm": marker_cm, "Z_cm": 0},
            {"id": 2, "X_cm": marker_cm, "Y_cm": grid_cm - 2 * marker_cm, "Z_cm": 0},
            {
                "id": 3,
                "X_cm": grid_cm - 2 * marker_cm,
                "Y_cm": grid_cm - 2 * marker_cm,
                "Z_cm": 0,
            },
        ]

        # yz-plane
        self.markers["yz"] = [
            {"id": 4, "X_cm": 0, "Y_cm": marker_cm, "Z_cm": 2 * marker_cm},
            {
                "id": 5,
                "X_cm": 0,
                "Y_cm": grid_cm - 2 * marker_cm,
                "Z_cm": 2 * marker_cm,
            },
            {"id": 6, "X_cm": 0, "Y_cm": marker_cm, "Z_cm": grid_cm - marker_cm},
            {
                "id": 7,
                "X_cm": 0,
                "Y_cm": grid_cm - 2 * marker_cm,
                "Z_cm": grid_cm - marker_cm,
            },
        ]

        # zx-plane
        self.markers["zx"] = [
            {"id": 8, "X_cm": 2 * marker_cm, "Y_cm": 0, "Z_cm": 2 * marker_cm},
            {"id": 9, "X_cm": grid_cm - marker_cm, "Y_cm": 0, "Z_cm": 2 * marker_cm},
            {"id": 10, "X_cm": 2 * marker_cm, "Y_cm": 0, "Z_cm": grid_cm - marker_cm},
            {
                "id": 11,
                "X_cm": grid_cm - marker_cm,
                "Y_cm": 0,
                "Z_cm": grid_cm - marker_cm,
            },
        ]

    def create_marker(
        self, marker_id, h_cm, v_cm, marker_pi, dpcm, marker_border=1, verbose=True,
    ):
        h_pi, v_pi = int(h_cm * dpcm), int(v_cm * dpcm)

        marker = np.zeros((marker_pi, marker_pi), dtype=np.uint8)
        marker = cv.aruco.drawMarker(
            self.marker_dictionary, marker_id, marker_pi, marker, marker_border
        )

        if verbose:
            print(marker.shape, f"{v_pi}:{v_pi + marker_pi}, {h_pi}:{h_pi + marker_pi}")

        return marker, h_pi, v_pi

    def detect_markers(self, frame, verbose=True):
        corners, ids, _ = cv.aruco.detectMarkers(
            frame, self.marker_dictionary, parameters=self.detect_parameters
        )

        if verbose:
            if ids.any():
                print_matrix(ids, "ids")

        return corners, ids

    @staticmethod
    def draw_markers(
        frame,
        corners,
        ids,
        marker_color=(0, 255, 0),
        marker_size=10,
        marker_type=cv.MARKER_CROSS,
        font_color=(0, 255, 0),
        font_scale=0.8,
    ):
        if ids.any():
            for i, c in enumerate(corners):
                c = np.squeeze(c)
                frame = draw_markers(
                    frame,
                    c.T,
                    color=marker_color,
                    marker_size=marker_size,
                    marker_type=marker_type,
                )

                frame = cv.putText(
                    frame,
                    str(int(ids[i])),
                    tuple(c.mean(axis=0)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_color,
                    thickness=2,
                )
        return frame
