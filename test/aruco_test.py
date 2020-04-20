import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from toolbox.visualizer import draw_markers


# Load the predefined dictionary
marker_dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
marker_id = 33
marker_dim = 200
border_dim = 1

img_m = np.zeros((marker_dim, marker_dim), dtype=np.uint8)
img_m = cv.aruco.drawMarker(marker_dictionary, marker_id, marker_dim, img_m, border_dim)

frame = 255 * np.ones((1000, 1000), dtype=np.uint8)
x, y = 500, 300
frame[y : y + marker_dim, x : x + marker_dim] = img_m

parameters = cv.aruco.DetectorParameters_create()
corners, ids, _ = cv.aruco.detectMarkers(
    frame, marker_dictionary, parameters=parameters
)

corners = np.squeeze(corners[0]).T
ids = ids[0]

frame_c = draw_markers(frame, corners, color=(0, 0, 0), marker_type=cv.MARKER_CROSS)

print(corners)
print(ids)

plt.figure()
plt.imshow(img_m, cmap="gray")
plt.show()

plt.figure()
plt.imshow(frame, cmap="gray")
plt.show()

plt.figure()
plt.imshow(frame_c, cmap="gray")
plt.show()
