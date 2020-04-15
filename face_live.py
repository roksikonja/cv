import cv2 as cv

cv.namedWindow("preview")
vc = cv.VideoCapture(0, cv.CAP_DSHOW)

face_cascade = cv.CascadeClassifier(
    "./venv/Lib/site-packages/cv/data/haarcascade_frontalface_default.xml"
)
eye_cascade = cv.CascadeClassifier(
    "./venv/Lib/site-packages/cv/data/haarcascade_eye.xml"
)

frame = None
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv.imshow("preview", frame)
    rval, frame = vc.read()

    key = cv.waitKey(20)
    if key == 27 or key == ord("q"):  # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")
cv.destroyAllWindows()
