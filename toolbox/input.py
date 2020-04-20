from urllib.request import urlopen

import cv2 as cv
import numpy as np

from toolbox.image import resize


class IPCamera(object):
    def __init__(self, url, window_name="ipcamera", fps=50):
        self.url = url
        self.window_name = window_name
        self.fps = fps

        cv.namedWindow(self.window_name)

    def get_frame(self, fraction=1.0, rot=3):
        response = urlopen(self.url)
        frame_array = np.array(bytearray(response.read()), dtype=np.uint8)
        frame = cv.imdecode(frame_array, -1)

        if rot > 0:
            frame = np.rot90(frame, rot)

        if fraction != 1.0:
            frame = resize(frame, fraction)

        return frame

    def show_frame(self, frame):
        cv.imshow(self.window_name, frame)

        key = cv.waitKey(int(1000 / self.fps))

        return key
