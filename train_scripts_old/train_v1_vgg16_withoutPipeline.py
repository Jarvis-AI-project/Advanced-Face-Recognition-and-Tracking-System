import tensorflow as tf
from keras import layers, models
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
# from time import sleep
import cv2
# import pandas as pd
# import logging
tf.config.set_visible_devices([], 'GPU')

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use the first GPU and enable memory growth
#         tf.config.set_visible_devices(gpus[0], 'GPU')
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         # print(len(gpus), "Physical GPUs:", len(logical_gpus), "Logical GPU")
#         print(f'Physical GPUs: {gpus} Logical GPUs: {logical_gpus}')
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print('Error: ', e)
        
def list_images(path):
    image_files = []
    count = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                print(f'Image found: {count}', end='\r')
                count += 1
    
    return image_files.sort()

neg_faces = list_images('/coding-drive/DATASETS/negative-faces/VGG-Face2/data/test-batch/')
deva_faces = list_images('/coding-drive/DATASETS/positive_face_datase/Devasheesh/')
swar_faces = list_images('/coding-drive/DATASETS/positive_face_datase/Swarnim/')

# using DNN
modelFile = "dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "dnn/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def extract_frontal_face(image_ndarray, grayscale=False, size=(224, 224)):
    # resize the image to 300x300 for the DNN model
    h, w = 300, 300
    image_ndarray = cv2.resize(image_ndarray, (h, w))
    # Convert into blob
    blob = cv2.dnn.blobFromImage(image_ndarray, 1.0, (h, w), (104.0, 177.0, 123.0))
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
# function to make the image size uniform and make train and test object with labels
class dataframe():
    def __init__(self):
        self.deva_faces = deva_faces
        self.swar_faces = swar_faces
        self.neg_faces = neg_faces
        self.images, self.labels = list(), list()

        self.total_images = len(deva_faces)+len(swar_faces)+len(neg_faces)
        self.labels_list = ['Unknown', 'Devasheesh', 'Swarnim']

        print('Total images: ', self.total_images)

        self.datagen = ImageDataGenerator(
            rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
            shear_range=0.2,    # set range for random shear
            zoom_range=0.2,    # set range for random zoom
            horizontal_flip=True,   # randomly flip images
            fill_mode='nearest' # set mode for filling points outside the input boundaries
        )

    def append_dataframe(self, images_list, label, augment=False, grayscale=True):
        count = 0
        for image_adr in images_list:
            try:
                # read the image
                image = cv2.imread(image_adr)
                # extract the face
                face, _ = extract_frontal_face(image, grayscale=grayscale, size=(224, 224))

                print('Count: ', count, end='\r')
                if face is not None:
                    # print('Face type: ', type(face))
                    # print('Image shape: ', image.shape)
                    # print('Face shape: ', face.shape)
                    # print('Face: ', plt.imshow(face))

                    # append the face image to the list
                    if augment:
                        # using the image data generator to augment the images
                        i = 0
                        for batch in self.datagen.flow(face.reshape(1, 224, 224, 1), batch_size=1):
                            # append the image to the list
                            self.images.append(batch[0])
                            self.labels.append(label)
                            i += 1
                            if i > 10:
                                break
                        count += 1
                
                    else:
                        self.images.append(face)
                        self.labels.append(label)
                        count += 1
                else:
                    print(f'Face not found in {image_adr}')
                    continue
            except Exception as e:
                print(f'Error: {e} in {image_adr}')
                # logging
                # logging.error(f'Error: {e} in {image_adr}')
                continue
             
    def make_dataframe(self):
        self.append_dataframe(self.neg_faces, 0, augment=False)
        self.append_dataframe(self.deva_faces, 1, augment=True)
        self.append_dataframe(self.swar_faces, 2, augment=True)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print('Images shape: ', self.images.shape)
        print('Labels shape: ', self.labels.shape)
        return self.images/255.0, self.labels, self.labels_list

x = dataframe()
images, labels, labels_list = x.make_dataframe()
        
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=42)

# Get the current physical devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Print the current device
    print("Currently using GPU:", gpus[0])
else:
    print("No GPU available.")

# model v1
model_v1 = models.Sequential()
model_v1.add(layers.Conv2D(
    64, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model_v1.add(layers.MaxPooling2D((2, 2)))
model_v1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_v1.add(layers.MaxPooling2D((2, 2)))
model_v1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_v1.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model_v1.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
model_v1.add(layers.Dense(1024, activation='relu'))
# model_v1.add(layers.Dense(512, activation='relu'))
model_v1.add(layers.Dense(256, activation='relu'))
# model_v1.add(layers.Dense(128, activation='relu'))
model_v1.add(layers.Dense(64, activation='relu'))
# model_v1.add(layers.Dense(32, activation='relu'))
model_v1.add(layers.Dense(len(labels_list), activation='softmax'))

print(model_v1.summary())
model_v1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model_v1.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels),
                    batch_size=16)
test_loss, test_acc = model_v1.evaluate(test_images,  test_labels, verbose=2)
print('Accuracy:',test_acc)
print('Loss',test_loss)
model_v1.save(
    './trained_models/face_recognition_model_v1_default_ep5_bs16_gray_temp.h5')

# model - vgg16
def VGG_FACE_16_layers(input_shape, labels):
    model_vgg16 = tf.keras.models.Sequential()

    # block 1
    model_vgg16.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
              activation='relu', name='conv1_1', input_shape=input_shape))
    model_vgg16.add(tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', activation='relu', name='conv1_2'))
    model_vgg16.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), name='pool1'))

    # block 2
    model_vgg16.add(tf.keras.layers.Conv2D(
        128, (3, 3), padding='same', activation='relu', name='conv2_1'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        128, (3, 3), padding='same', activation='relu', name='conv2_2'))
    model_vgg16.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), name='pool2'))

    # block 3
    model_vgg16.add(tf.keras.layers.Conv2D(
        256, (3, 3), padding='same', activation='relu', name='conv3_1'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        256, (3, 3), padding='same', activation='relu', name='conv3_2'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        256, (3, 3), padding='same', activation='relu', name='conv3_3'))
    model_vgg16.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), name='pool3'))

    # block 4
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv4_1'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv4_2'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv4_3'))
    model_vgg16.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), name='pool4'))

    # block 5
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv5_1'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv5_2'))
    model_vgg16.add(tf.keras.layers.Conv2D(
        512, (3, 3), padding='same', activation='relu', name='conv5_3'))
    model_vgg16.add(tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), name='pool5'))

    # flatten and fully connected layers
    model_vgg16.add(tf.keras.layers.Flatten(name='flatten'))
    model_vgg16.add(tf.keras.layers.Dense(4096, activation='relu', name='fc6'))
    model_vgg16.add(tf.keras.layers.Dense(4096, activation='relu', name='fc7'))
    model_vgg16.add(tf.keras.layers.Dense(2622, activation='relu', name='fc8'))
    model_vgg16.add(tf.keras.layers.Dense(labels, activation='softmax'))

    return model_vgg16


model_vgg16 = VGG_FACE_16_layers((224, 224, 1), len(labels_list))
model_vgg16.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model_vgg16.summary())

history = model_vgg16.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels),
                    batch_size=16)
test_loss, test_acc = model_vgg16.evaluate(
    test_images,  test_labels, verbose=2)
print('Accuracy:',test_acc)
print('Loss',test_loss)
model_vgg16.save('./trained_models/face_recognition_model_vgg16_ep5_bs16_gray_temp.h5')