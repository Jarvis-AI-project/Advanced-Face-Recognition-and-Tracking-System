# Test the model - Self independent test

# imports
import cv2
import tensorflow as tf
import numpy as np

# load the model
model = tf.keras.models.load_model(
    r"trained_models\face_recognition_model_v2_ep15_bs16.h5")
# using DNN
modelFile = "dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def extract_frontal_face(image_ndarray, size, grayscale=False):
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


def predict_video(model, labels_list):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        frame = cv2.resize(frame, (300, 300))
        try:
            # extract the face
            face, (x, y, x2, y2) = extract_frontal_face(
                frame, size=(300, 300))
            if face is not None:
                # predict the face
                pred = model.predict(face.reshape(1, 300, 300, 3))
                # get the label
                label = labels_list[np.argmax(pred)]
                # draw the rectangle and put the text
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # show the frame
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Quitting...')
                    break
            else:
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('Quitting...')
                    break
        except Exception as e:
            print(e)
            pass


# call the predict function
predict_video(model, labels_list=['Unknown', 'Devasheesh', 'Swarnim'])
