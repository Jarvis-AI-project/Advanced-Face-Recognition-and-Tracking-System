import cv2
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    # print(frame.height, frame.width)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break