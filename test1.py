import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
num_classes = 10

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# decrease the size of the images

train_images = train_images[:1000]
train_labels = train_labels[:1000]


import cv2
import numpy as np

train_images = np.array([cv2.resize(img, (128, 128)) for img in train_images])
test_images = np.array([cv2.resize(img, (128, 128)) for img in test_images])
print(train_labels.shape)
# Define the CNN model
model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
  MaxPooling2D((2, 2)),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Flatten(),
  Dense(512, activation='relu'),
  Dense(1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Save the trained model
model.save('face_recognition_model.h5')

# Load the trained model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Use the model to predict the identity of a new face image