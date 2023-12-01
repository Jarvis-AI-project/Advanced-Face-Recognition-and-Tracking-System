import tensorflow as tf
import numpy as np
import cv2

# Loading the model
model = tf.keras.models.load_model(r'C:\Shared\face_recognition_vggface.h5')

# Extraction of face from image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    faces_cord = face_cascade.detectMultiScale(img, 1.3, 5)
    return faces_cord

# Prediction on webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_cord = detect_face(frame)
    for face_cord in faces_cord:
        x, y, w, h = face_cord
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        if label == 0:
            print('Devasheesh')
        elif label == 1:
            print('Bundiwall')
        elif label == 2:
            print('Unknown')
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
        # cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('LIVE', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
https://late-fog-a0a7.fisij712323030.workers.dev/b8ad52d44c93566306e669b2ac04888171ebf89c59b75c802cb33789550bcc4828c9592baf8f81689835622a8d0cd485::acf77901c3650de140f8bb98592099ca/1395897669/Koogle%20Kuttappa%20(2022)%201080p%20UnCut%20HDRip%20ORG%20Dual%20x264%20ESub%20-%20Vegamovies.to.mkv