import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, BatchNormalization,
                                     Activation)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras import mixed_precision

INPUT_DATASET = "/home/chenzz/projects/def-czarnuch/chenzz/datasets/mensa/"
LOG_PATH = "./logs"


def load_data(input_dataset, data_type):
    x = []
    y = []
    dtype = None
    channels = None
    max_datasets = 100
    for file in os.listdir(input_dataset):
        if file.endswith(f"{data_type}.npy") and max_datasets > 0:
            max_datasets -= 1
            participant_id = int(file.split('_')[0])
            data = np.load(os.path.join(input_dataset, file))
            if len(data.shape) == 3:  # Add channel dimension if it's missing
                data = np.expand_dims(data, axis=-1)
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


def create_alexnet(input_shape, num_classes):
    print(f'Creating model for {input_shape}, # class={num_classes}')
    model = Sequential([
        # Resizing(256, 256, 3),

        Conv2D(96, (11, 11), strides=4, activation='relu',
        input_shape=input_shape
        ),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        Conv2D(256, (5, 5), padding='same', activation='relu'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        Conv2D(384, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(384, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),

        Flatten(),

        Dense(4096, activation='relu'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(4096, activation='relu'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape)
    return model

def create_old_model(inputsize, nclass):
	if len(inputsize) == 2:
		newsize = inputsize + (1,)
	else:
		newsize = inputsize
	print('new size', newsize)

	# CNN
	model = models.Sequential()
	model.add(layers.Reshape(newsize, input_shape=inputsize))
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=newsize, data_format='channels_last'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	# dense layers
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	# model.add(layers.Dropout(0.5))
	model.add(layers.Dense(nclass, activation='linear'))

	model.build(input_shape=inputsize)
	model.summary()

	model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])
	return model

def train_model(x, y, input_shape, log_path):
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    early_stopping = EarlyStopping(patience=3)
    tensorboard = TensorBoard(log_dir=log_path)

    k = 2
    kfold = KFold(n_splits=k)
    overall_cm = None
    fold = 1

    for train_indices, test_indices in kfold.split(x, y):
        # manually clean up GPU memory
        tf.keras.backend.clear_session()

        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        num_classes = len(np.unique(y_train))
        print(f'We have the following classes for fold={fold}',
            np.unique(y_train))
        model = create_alexnet(input_shape, num_classes)
        # model = create_old_model(input_shape, num_classes)
        model.summary()

        model.fit(x_train, y_train_encoded, epochs=100, batch_size=8,
            validation_data=(x_test, y_test_encoded),
            callbacks=[early_stopping, tensorboard])

        y_pred = np.argmax(model.predict(x_test), axis=-1)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        cm = confusion_matrix(y_test, y_pred_decoded)
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
    print('Input shape', input_shape)

    train_model(x, y, input_shape, LOG_PATH)
