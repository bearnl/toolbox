import os
import sys
import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, BatchNormalization,
                                     Activation)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers, mixed_precision

#INPUT_DATASET = "/home/chenzz/projects/def-czarnuch/chenzz/datasets/mensa/"
INPUT_DATASET = sys.argv[1]
dataset_name = os.path.basename(os.path.normpath(INPUT_DATASET))
DATA_TYPE = sys.argv[2]  # "rgb" or "depth"
MODEL_NAME = len(sys.argv) >= 4 and sys.argv[3] or 'alexnet'
BASE_LOG_PATH = "/home/chenzz/scratch/logs"
LOG_PATH_SUFFIX = f"{dataset_name}-{DATA_TYPE}/{MODEL_NAME}"
TASK_ID = os.getenv('SLURM_ARRAY_TASK_ID')
JOB_ID = os.getenv('SLURM_ARRAY_JOB_ID')
if JOB_ID and TASK_ID:
    LOG_PATH = f"{BASE_LOG_PATH}/{JOB_ID}-{TASK_ID}/{LOG_PATH_SUFFIX}"
else:
    d = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_PATH = f"{BASE_LOG_PATH}/{d}/{LOG_PATH_SUFFIX}"
print('LOG_PATH', LOG_PATH)
BATCH_SIZE = 4

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def parse_tfrecord_fn(example, dtype):
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)

    image = tf.io.parse_tensor(example['image_raw'], out_type=dtype)
    image = tf.reshape(image, [example['height'], example['width'], example['channels']])
    label = example['label']

    return image, label

def load_data(input_dataset, data_type):
    dataset = tf.data.TFRecordDataset(f'{input_dataset}-{data_type}.tfrecords')

    if data_type == 'depth':
        parser = lambda x: parse_tfrecord_fn(x, tf.uint16)
    else:
        parser = lambda x: parse_tfrecord_fn(x, tf.uint8)
    dataset = dataset.map(parser)
    return dataset

# def load_data(input_dataset, data_type):
#     x = []
#     y = []
#     dtype = None
#     channels = None
#     max_datasets = 100
#     for file in os.listdir(input_dataset):
#         if file.endswith(f"{data_type}.npy") and max_datasets > 0:
#             max_datasets -= 1
#             participant_id = file.split('_')[0]
#             data = np.load(os.path.join(input_dataset, file))
#             if len(data.shape) == 3:  # Add channel dimension if it's missing
#                 data = np.expand_dims(data, axis=-1)
#             print('Loaded', file, 'with shape', data.shape, 'and dtype', data.dtype, 'from participant', participant_id, '.')
#             x.extend(data)
#             y.extend([participant_id] * len(data))
#             if dtype is None:
#                 dtype = data.dtype
#             if channels is None:
#                 channels = data.shape[-1]
#     x = np.array(x, dtype=dtype)
#     y = np.array(y)
#     return x, y, dtype, channels

def create_alexnet(input_shape, num_classes):
    print(f'Creating model for {input_shape}, # class={num_classes}')
    model = Sequential([
        Resizing(256, 256, input_shape=input_shape),

        Conv2D(96, (11, 11), strides=4, activation='relu',
        # input_shape=input_shape
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

def train_model(dataset, log_path):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    k = 5
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    class_names = sorted(set(item.numpy() for item in dataset.map(lambda image, label: label)))
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    fold = 1
    for train_indices, test_indices in kfold.split(dataset):
        # manually clean up GPU memory
        tf.keras.backend.clear_session()

        train_dataset = dataset.take(len(train_indices)).map(lambda image, label: (image, label_encoder(label))).batch(BATCH_SIZE)
        test_dataset = dataset.skip(len(train_indices)).map(lambda image, label: (image, label_encoder(label))).batch(BATCH_SIZE)

        num_classes = len(np.unique(class_names))
        print(f'We have the following classes for fold={fold}', class_names)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        early_stopping = EarlyStopping(patience=10)
        log_dir = f'{log_path}/fold{fold}'
        tensorboard = TensorBoard(log_dir=log_dir)

        with mirrored_strategy.scope():
            if MODEL_NAME == 'alexnet':
                model = create_alexnet(input_shape, num_classes)
            elif MODEL_NAME == 'old':
                model = create_old_model(input_shape, num_classes)
        model.summary()

        performance_metrics = PerformanceMetrics(validation_data=test_dataset,
            log_dir=log_dir, label_encoder=label_encoder)
        model.fit(train_dataset, epochs=100,
            validation_data=test_dataset,
            callbacks=[early_stopping, tensorboard, performance_metrics])

        fold += 1


class PerformanceMetrics(Callback):
    def __init__(self, validation_data, log_dir, label_encoder):
        super(PerformanceMetrics, self).__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.label_encoder = label_encoder
        self.writer = tf.summary.create_file_writer(f'{self.log_dir}/validation')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_predict = []
        val_targ = []
        for x, y in self.validation_data:
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=-1)
            val_predict.extend(y_pred)
            val_targ.extend(y.numpy())

        cm = confusion_matrix(val_targ, val_predict)
        accuracy = accuracy_score(val_targ, val_predict)
        precision = precision_score(val_targ, val_predict, average='macro')
        recall = recall_score(val_targ, val_predict, average='macro')
        f1 = f1_score(val_targ, val_predict, average='macro')

        with self.writer.as_default():
            tf.summary.scalar('val_accuracy', accuracy, step=epoch)
            tf.summary.scalar('val_precision', precision, step=epoch)
            tf.summary.scalar('val_recall', recall, step=epoch)
            tf.summary.scalar('val_f1', f1, step=epoch)
            tf.summary.text('val_cm', cm, step=epoch)
            tf.summary.text('label_encoding', list(self.label_encoder.classes_))

        logs['val_accuracy'] = accuracy
        logs['val_precision'] = precision
        logs['val_recall'] = recall
        logs['val_f1'] = f1

        print(f"— val_accuracy: {accuracy} — val_precision: {precision} — val_recall {recall} — val_f1: {f1}")

        return


if __name__ == '__main__':
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    dataset = load_data(INPUT_DATASET, DATA_TYPE)

    train_model(dataset, LOG_PATH)
