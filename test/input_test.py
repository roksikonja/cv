import cv2 as cv

cv.namedWindow("preview")
vc = cv.VideoCapture(0, cv.CAP_DSHOW)

frame = None
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv.imshow("preview", frame)
    rval, frame = vc.read()

    key = cv.waitKey(20)
    if key == 27 or key == ord("q"):  # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")
cv.destroyAllWindows()
