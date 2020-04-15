from urllib.request import urlopen

import cv2 as cv
import numpy as np

cv.namedWindow("ipcamera")

# Get IP from IP Camera App
url = "http://192.168.2.187:8080/shot.jpg"
fps = 120

while True:
    response = urlopen(url)
    img_array: np.ndarray = np.array(bytearray(response.read()), dtype=np.uint8)
    img: np.ndarray = cv.imdecode(img_array, -1)

    cv.imshow("ipcamera", img)

    key = cv.waitKey(int(1000.0 / fps))
    if key == ord("q"):
        break

cv.destroyWindow("preview")
cv.destroyAllWindows()
