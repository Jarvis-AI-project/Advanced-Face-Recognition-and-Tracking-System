import cv2
import numpy as np
# function to extract the face from the image
# using the Haar Cascade Classifier
def extract_frontal_face_harr(image_ndarray, grayscale=True, size=(150, 150)):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')
    # Convert into grayscale
    if grayscale:
        image_gray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_ndarray
    # Detect faces
    faces_cord = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # Return the face or None if not found
    if len(faces_cord) == 0:
        return None, None
    # Extract the face
    (x, y, w, h) = faces_cord[0]
    # Resize the image to 150x150
    image_gray_resized = cv2.resize(image_gray[y:y+w, x:x+h], size)
    # Return only the face part of the image
    return image_gray_resized, faces_cord

# using DNN


def extract_frontal_face(image_ndarray, grayscale=False, size=(150, 150)):
    modelFile = "dnn/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "dnn/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # resize the image to 300x300 for the DNN model
    h, w = 300, 300
    image_ndarray = cv2.resize(image_ndarray, (h, w))
    # Convert into blob
    blob = cv2.dnn.blobFromImage(
        image_ndarray, 1.0, (h, w), (104.0, 177.0, 123.0))
    # Convert into grayscale
    if grayscale:
        image_ndarray = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Extract the face ROI (region of interest) from the image
            face = image_ndarray[y:y2, x:x2]
            return cv2.resize(face, size), (x, y, x2, y2)
        else:
            return None, None
