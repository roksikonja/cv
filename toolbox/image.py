import cv2 as cv


def resize(img, fraction):
    m, n = img.shape[:2]

    img = cv.resize(
        img, (int(n * fraction), int(m * fraction)), interpolation=cv.INTER_AREA,
    )

    return img
