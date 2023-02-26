import cv2

while True:
    _, frame = cv2.VideoCapture(0).read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break