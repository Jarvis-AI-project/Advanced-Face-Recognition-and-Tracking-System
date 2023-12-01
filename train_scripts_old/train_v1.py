import datetime
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Input
import tensorflow as tf

model_name = 'v1_final'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU and enable memory growth
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs:", len(logical_gpus), "Logical GPU")
        print(f'Physical GPUs: {gpus} Logical GPUs: {logical_gpus}')
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print('Error: ', e)
else:
    print('No GPUs found')


# Estimated time: 1.5 min
train = tf.keras.utils.image_dataset_from_directory(
    '/coding-drive/DATASETS/face-recognition-data/',
    batch_size=4,
    image_size=(224, 224),
    color_mode='grayscale',
    shuffle=True
)


test = tf.keras.utils.image_dataset_from_directory(
    '/coding-drive/DATASETS/face-recognition-tensorflow-test-data/',
    batch_size=4,
    image_size=(224, 224),
    color_mode='grayscale',
    shuffle=True
)


train = train.map(lambda x, y: (x/255, y))
test = test.map(lambda x, y: (x/255, y))

model = Sequential()
model.add(Input(shape=(224, 224, 1), name='input'))

# Convolutional layer - 1 (32 filters, 3x3 kernel, relu activation)
model.add(Conv2D(filters=32, kernel_size=(3, 3),
          padding='same', name='conv2d_1', activation='relu'))
model.add(BatchNormalization(name='batch_norm_1'))

model.add(Conv2D(filters=32, kernel_size=(3, 3),
          padding='same', name='conv2d_2', activation='relu'))
model.add(BatchNormalization(name='batch_norm_2'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer - 2 (64 filters, 3x3 kernel, relu activation)
model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding='same', name='conv2d_3', activation='relu'))
model.add(BatchNormalization(name='batch_norm_3'))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
          padding='same', name='conv2d_4', activation='relu'))
model.add(BatchNormalization(name='batch_norm_4'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer - 3 (128 filters, 3x3 kernel, relu activation)
model.add(Conv2D(filters=128, kernel_size=(3, 3),
          padding='same', name='conv2d_5', activation='relu'))
model.add(BatchNormalization(name='batch_norm_5'))

model.add(Conv2D(filters=128, kernel_size=(3, 3),
          padding='same', name='conv2d_6', activation='relu'))
model.add(BatchNormalization(name='batch_norm_6'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer - 4 (256 filters, 3x3 kernel, relu activation)
model.add(Conv2D(filters=256, kernel_size=(3, 3),
          padding='same', name='conv2d_7', activation='relu'))
model.add(BatchNormalization(name='batch_norm_7'))

model.add(Conv2D(filters=256, kernel_size=(3, 3),
          padding='same', name='conv2d_8', activation='relu'))
model.add(BatchNormalization(name='batch_norm_8'))

# Add an attention layer
# model.add(Attention(name='attention'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Dense layer - 1 (512 neurons, relu activation)
model.add(Dense(512, activation='relu', name='dense_1'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', name='dense_2'))
model.add(Dropout(0.5))

# Dense layer - output (3 neurons, softmax activation)
model.add(Dense(3, activation='softmax', name='output'))


model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)


history = model.fit(
    train,
    epochs=10,
    validation_data=test,
    callbacks=[
        tf.keras.callbacks.TensorBoard(
            log_dir=f"training_output/{model_name}/tensorboard/fit/" +
            datetime.datetime.now().strftime("%m-%d_%H-%M-%S"),
            histogram_freq=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'training_output/{model_name}/checkpoints/' +
            'epoch_{epoch: 02d}_val_loss_{val_loss: .2f}',
            save_freq='epoch',
            save_best_only=False,
            save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            'val_loss', patience=2, verbose=1),
        tf.keras.callbacks.CSVLogger(
            f'training_output/{model_name}/training_log.csv'),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-3 * 10**(epoch / 20)),
    ]
)

model.save(f'training_output/{model_name}/final_model.h5')

print('Final_Performance: ', history.history)
