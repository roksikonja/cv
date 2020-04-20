import cv2 as cv

from toolbox.input import IPCamera
from toolbox.markers import ArucoMarkers

reader = IPCamera(url="http://192.168.2.187:8080/shot.jpg", fps=20)
aruco = ArucoMarkers(grid_cm=20, marker_cm=2)

while True:
    frame = reader.get_frame(fraction=0.4, rot=3)

    corners, ids = aruco.detect_markers(frame, verbose=True)
    frame = aruco.draw_markers(frame, corners, ids)

    key = reader.show_frame(frame)
    if key == 27 or key == ord("q"):
        break

cv.destroyAllWindows()
