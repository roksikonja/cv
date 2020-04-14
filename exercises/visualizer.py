import cv2
import numpy as np


def matrix_to_tuple_list(matrix):
    tuple_list = []
    for i in range(matrix.shape[1]):
        vector = matrix[:, i]
        tuple_list.append(tuple(vector))

    return tuple_list


def print_matrix(matrix, name=None):
    if name:
        print(name, "=")
    lines = []
    matrix = np.atleast_2d(matrix)

    for row in matrix:
        line = ""
        for cell in row:
            if cell == 0:
                line = line + "{:>10}".format(int(cell))
            else:
                line = line + "{:>10.4f}".format(cell)

        lines.append(line)
    print("\n".join(lines))
    print("\n")


def draw_markers(
    img,
    xy,
    color=(0, 0, 255),
    marker_type=cv2.MARKER_CROSS,
    marker_size=50,
    marker_thickness=4,
):
    img_markers = img.copy()

    for marker in matrix_to_tuple_list(xy.astype(int)):
        img_markers = cv2.drawMarker(
            img_markers,
            marker,
            color=color,
            markerType=marker_type,
            markerSize=marker_size,
            thickness=marker_thickness,
        )

    return img_markers
