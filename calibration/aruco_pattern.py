import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from toolbox.markers import ArucoMarkers

# Load the predefined dictionary
marker_dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

"""
    Parameters.
"""
patterns_dir = "./patterns"

dpi = 300
inch_to_cm = 2.54
dpcm = dpi / inch_to_cm

marker_cm = 2
grid_cm = 20

marker_border = 1

grid_pi = int(grid_cm * dpcm)
marker_pi = int(marker_cm * dpcm)

"""
    Grid construction.
"""
grids = dict()
grids["xy"] = 255 * np.ones((grid_pi, grid_pi), dtype=np.uint8)
grids["yz"] = 255 * np.ones((grid_pi, grid_pi), dtype=np.uint8)
grids["zx"] = 255 * np.ones((grid_pi, grid_pi), dtype=np.uint8)
print(grids["xy"].shape)

"""
    Marker construction.
"""
aruco_grid = ArucoMarkers(grid_cm, marker_cm)

# xy-plane
for params in aruco_grid.markers["xy"]:
    marker, h_pi, v_pi = ArucoMarkers.create_marker(
        marker_dictionary,
        params["id"],
        params["Y_cm"],
        params["X_cm"],
        marker_pi,
        dpcm,
        marker_border,
    )
    grids["xy"][v_pi : v_pi + marker_pi, h_pi : h_pi + marker_pi] = marker

for params in aruco_grid.markers["yz"]:
    marker, h_pi, v_pi = ArucoMarkers.create_marker(
        marker_dictionary,
        params["id"],
        params["Y_cm"],
        grid_cm - params["Z_cm"],
        marker_pi,
        dpcm,
        marker_border,
    )
    grids["yz"][v_pi : v_pi + marker_pi, h_pi : h_pi + marker_pi] = marker

for params in aruco_grid.markers["zx"]:
    marker, h_pi, v_pi = ArucoMarkers.create_marker(
        marker_dictionary,
        params["id"],
        grid_cm - params["X_cm"],
        grid_cm - params["Z_cm"],
        marker_pi,
        dpcm,
        marker_border,
    )
    grids["zx"][v_pi : v_pi + marker_pi, h_pi : h_pi + marker_pi] = marker


plt.figure()
plt.imshow(grids["xy"], cmap="gray", vmin=0, vmax=255)
plt.show()

plt.figure()
plt.imshow(grids["yz"], cmap="gray", vmin=0, vmax=255)
plt.show()

plt.figure()
plt.imshow(grids["zx"], cmap="gray", vmin=0, vmax=255)
plt.show()

cv.imwrite(
    os.path.join(patterns_dir, "checkerboard_xy.png"),
    grids["xy"],
    [cv.IMWRITE_PNG_COMPRESSION, 0],
)
cv.imwrite(
    os.path.join(patterns_dir, "checkerboard_yz.png"),
    grids["yz"],
    [cv.IMWRITE_PNG_COMPRESSION, 0],
)
cv.imwrite(
    os.path.join(patterns_dir, "checkerboard_zx.png"),
    grids["zx"],
    [cv.IMWRITE_PNG_COMPRESSION, 0],
)
