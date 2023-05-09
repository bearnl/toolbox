import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential

INPUT_DATASET = "/mnt/d/Downloads/datasets/mensa_extracted"
LOG_PATH = "/mnt/d/Downloads/datasets/log"


def load_data(input_dataset, data_type):
    x = []
    y = []
    dtype = None
    channels = None
    for file in os.listdir(input_dataset):
        if file.endswith(f"{data_type}.npy"):
            participant_id = int(file.split('_')[0])
            data = np.load(os.path.join(input_dataset, file))
            print('Loaded', file, 'with shape', data.shape, 'and dtype', data.dtype, 'from participant', participant_id, '.')
            x.extend(data)
            y.extend([participant_id] * len(data))
            if dtype is None:
                dtype = data.dtype
            if channels is None:
                channels = data.shape[-1]
    x = np.array(x, dtype=dtype)
    y = np.array(y)
    return x, y, dtype, channels


def create_alexnet(input_shape):
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(np.max(y) + 1, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(x, y, input_shape, log_path):
    early_stopping = EarlyStopping(patience=3)
    tensorboard = TensorBoard(log_dir=log_path)

    k = 5
    kfold = KFold(n_splits=k)
    overall_cm = None
    fold = 1

    for train_indices, test_indices in kfold.split(x, y):
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model = create_alexnet(input_shape)

        model.fit(x_train, y_train, epochs=100, batch_size=2, validation_data=(x_test, y_test),
                  callbacks=[early_stopping, tensorboard])

        y_pred = np.argmax(model.predict(x_test), axis=-1)
        cm = confusion_matrix(y_test, y_pred)
        if overall_cm is None:
            overall_cm = cm
        else:
            overall_cm += cm
        print(f"Confusion Matrix for fold {fold}:\n{cm}")
        fold += 1

    print(f"Overall Confusion Matrix:\n{overall_cm}")


if __name__ == '__main__':
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    DATA_TYPE = "rgb"  # or "depth"
    x, y, dtype, num_channels = load_data(INPUT_DATASET, DATA_TYPE)

    # Get the input shape of the data, accounting for the number of channels:
    input_shape = x.shape[1:]

    train_model(x, y, input_shape, LOG_PATH)
