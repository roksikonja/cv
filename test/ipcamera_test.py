from urllib.request import urlopen

import cv2
import numpy as np

cv2.namedWindow("ipcamera")

# Get IP from IP Camera App
url = "http://192.168.2.187:8080/shot.jpg"
fps = 120

while True:
    response = urlopen(url)
    img_array: np.ndarray = np.array(bytearray(response.read()), dtype=np.uint8)
    img: np.ndarray = cv2.imdecode(img_array, -1)

    cv2.imshow("ipcamera", img)

    key = cv2.waitKey(int(1000.0 / fps))
    if key == ord("q"):
        break

cv2.destroyWindow("preview")
cv2.destroyAllWindows()
