from keras import layers, models

def model_v1(labels_list, input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(len(labels_list), activation='softmax'))
    return model
