import os
import sys
import datetime
import itertools
import io
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, BatchNormalization,
                                     Activation, Reshape)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers, mixed_precision
from keras_cv.models.legacy.vit import ViTTiny16

#INPUT_DATASET = "/home/chenzz/projects/def-czarnuch/chenzz/datasets/mensa/"
INPUT_DATASET = sys.argv[1]
dataset_name = os.path.basename(os.path.normpath(INPUT_DATASET))
DATA_TYPE = sys.argv[2]  # "rgb" or "depth"
MODEL_NAME = len(sys.argv) >= 4 and sys.argv[3] or 'alexnet'
LEARNING_RATE = 0.0001
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
BATCH_SIZE = 32

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

def remap_to_depth8(img):
    min_val, max_val = np.min(img), np.max(img)
    img = ((img - min_val) / (max_val - min_val)) * 255
    img = img.astype(np.uint8)

def load_biwi(base_dir, depth8=False):
    def load_and_preprocess_image(path, label, is_depth=False):
        def read_depth(path):
            fname = path.numpy().decode('utf-8')
            img = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

            if depth8:
                min_val, max_val = np.min(img), np.max(img)
                img = ((img - min_val) / (max_val - min_val)) * 255
                img = img.astype(np.uint8)

            return np.expand_dims(img, axis=-1)
        try:
            if is_depth:
                image = tf.py_function(func=read_depth,
                    inp=[path], Tout=tf.uint16)
                image.set_shape([None, None, 1])
            else:
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, [256, 256])
            return image, label
        except tf.errors.InvalidArgumentError:
            print('Invalid image file, skipping:', path)
            return None, label

    def create_dataset(image_dir, is_depth=False):
        # Get the class names
        class_names = sorted(os.listdir(image_dir))

        # List of all file paths and corresponding labels
        all_image_paths = []
        all_image_labels = []
        for class_name in class_names:
            class_dir = os.path.join(image_dir, class_name)
            for fname in os.listdir(class_dir):
                if is_depth and fname.lower().endswith('_depth.pgm') or not is_depth and fname.lower().endswith('_rgb.jpg'):
                    all_image_paths.append(os.path.join(class_dir, fname))
                    all_image_labels.append(class_name)

        # Create a dataset of paths and labels
        path_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

        # Map the loading and preprocessing function over the dataset of paths and labels
        image_label_ds = path_label_ds.map(lambda path, label: load_and_preprocess_image(path, label, is_depth), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_label_ds = image_label_ds.ignore_errors(log_warning=True)

        # Filter out any corrupted images
        image_label_ds = image_label_ds.filter(lambda image, label: image is not None)

        return image_label_ds, list(set(all_image_labels)), len(all_image_paths)


    # Create validation datasets
    rgb_dirs = [os.path.join(base_dir, 'Testing', sub_dir) for sub_dir in ['Still', 'Walking']]
    rgb_dirs.append(os.path.join(base_dir, 'Training'))
    depth_dirs = [os.path.join(base_dir, 'Testing', sub_dir) for sub_dir in ['Still', 'Walking']]
    depth_dirs.append(os.path.join(base_dir, 'Training'))

    labels = []
    rgb_datasets = []
    size = 0
    for dir in rgb_dirs:
        ds, lb, s = create_dataset(dir) 
        labels.extend(lb)
        rgb_datasets.append(ds)
        size += s
    labels = list(set(labels))
    depth_datasets = [create_dataset(dir, is_depth=True)[0] for dir in depth_dirs]

    # Combine the validation datasets
    combined_rgb_dataset = rgb_datasets[0]
    combined_depth_dataset = depth_datasets[0]

    for i in range(1, len(rgb_datasets)):
        combined_rgb_dataset = combined_rgb_dataset.concatenate(rgb_datasets[i])
        combined_depth_dataset = combined_depth_dataset.concatenate(depth_datasets[i])

    combined_rgb_dataset = combined_rgb_dataset.shuffle(buffer_size=size, reshuffle_each_iteration=False)
    combined_depth_dataset = combined_depth_dataset.shuffle(buffer_size=size, reshuffle_each_iteration=False)

    VALIDATION_SIZE = 400
    return (
        combined_rgb_dataset.skip(VALIDATION_SIZE),
        combined_rgb_dataset.take(VALIDATION_SIZE)
        ), (
        combined_depth_dataset.skip(VALIDATION_SIZE),
        combined_depth_dataset.take(VALIDATION_SIZE)
        ), labels

def load_mensa(directory, depth8=False):

    def load_and_preprocess_from_path_label(data_path):
        # Read image file
        image = tf.io.read_file(data_path)

        # Get label (folder name) from file path
        label = tf.strings.split(data_path, os.path.sep)[-2]
        
        # Check if image is depth or rgb
        is_depth = tf.strings.regex_full_match(data_path, ".*_depth\.png")
        
        # Decode image: 1 channel if depth, 3 if RGB
        image = tf.cond(is_depth,
            lambda: tf.image.decode_png(image, channels=1),
            lambda: tf.image.decode_png(image, channels=3))

        # Remap to depth8 if depth image and depth8 is True
        if depth8 and is_depth:
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            image = ((image - min_val) / (max_val - min_val)) * 255
            image = tf.cast(image, tf.uint8)

        # Resize image to the desired size
        image = tf.image.resize(image, [256, 256])

        return image, label


    file_list = tf.data.Dataset.list_files(directory + '/*/*')
    print('counting files...')
    files = [1 for _ in file_list]
    size = len(files)
    print('file count', size)
    file_list = tf.data.Dataset.list_files(directory + '/*/*')
    file_list = file_list.shuffle(buffer_size=size)

    depth_filelist = file_list.filter(lambda x: tf.strings.regex_full_match(x, ".*_depth\.png"))
    rgb_filelist = file_list.filter(lambda x: tf.strings.regex_full_match(x, ".*_rgb\.png"))

    depth_data = depth_filelist.map(load_and_preprocess_from_path_label)
    rgb_data = rgb_filelist.map(load_and_preprocess_from_path_label)

    labels = ['01', '03', '05', '07', '09', '12', '14', '16', '18', '20', '22', '24', '28', '30', '32', '34', '02', '04', '06', '08', '10', '13', '15', '17', '19', '21', '23', '27', '29', '31', '33']

    VALIDATION_SIZE = 400

    return (
        rgb_data.skip(VALIDATION_SIZE),
        rgb_data.take(VALIDATION_SIZE)
        ), (
        depth_data.skip(VALIDATION_SIZE),
        depth_data.take(VALIDATION_SIZE)
        ), labels

def load_inhouse(directory, depth8=False):

    def load_and_preprocess_from_path_label(data_path):
        # Read image file
        image = tf.io.read_file(data_path)

        # Get label (folder name) from file path
        label = tf.strings.split(data_path, '_')[-2]
        
        # Check if image is depth or rgb
        is_depth = tf.strings.regex_full_match(data_path, ".*_depth\.png")
        
        # Decode image: 1 channel if depth, 3 if RGB
        image = tf.cond(is_depth,
            lambda: tf.image.decode_png(image, channels=1),
            lambda: tf.image.decode_png(image, channels=3))

        # Remap to depth8 if depth image and depth8 is True
        if depth8 and is_depth:
            min_val = tf.reduce_min(image)
            max_val = tf.reduce_max(image)
            image = ((image - min_val) / (max_val - min_val)) * 255
            image = tf.cast(image, tf.uint8)

        # Resize image to the desired size
        image = tf.image.resize(image, [256, 256])

        return image, label


    file_list = tf.data.Dataset.list_files(directory + '/*/*')
    print('counting files...')
    files = [1 for _ in file_list]
    size = len(files)
    print('file count', size)
    file_list = tf.data.Dataset.list_files(directory + '/*/*')
    # file_list = file_list.shuffle(buffer_size=size)

    depth_filelist = file_list.filter(lambda x: tf.strings.regex_full_match(x, ".*_depth\.png"))
    rgb_filelist = file_list.filter(lambda x: tf.strings.regex_full_match(x, ".*_rgb\.png"))

    depth_data = depth_filelist.map(load_and_preprocess_from_path_label).shuffle(buffer_size=size/2)
    rgb_data = rgb_filelist.map(load_and_preprocess_from_path_label).shuffle(buffer_size=size/2)

    labels = ['leo', 'bear', 'ranyi', 'xiaoyu', 'lirunze', 'lady1', 'lady2', 'man1', 'man2', 'stephen']

    VALIDATION_SIZE = 1500

    return (
        rgb_data.skip(VALIDATION_SIZE),
        rgb_data.take(VALIDATION_SIZE)
        ), (
        depth_data.skip(VALIDATION_SIZE),
        depth_data.take(VALIDATION_SIZE)
        ), labels

def create_alexnet(input_shape, num_classes):
    print(f'Creating model for {input_shape}, # class={num_classes}')
    model = Sequential([
        # Resizing(256, 256, input_shape=input_shape),
        # Reshape(input_shape+(1,), input_shape=input_shape),

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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
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

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])
	return model

def create_vit(inputsize, nclass):
    model = ViTTiny16(
        include_rescaling=True,
        include_top=True,
        num_classes=nclass,
        input_shape=inputsize
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.build(input_shape)

    return model

def train_model(dataset, test_dataset, labels, input_shape, log_path):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    tf.keras.backend.clear_session()

    class_names = labels

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    # Create a mapping of labels to encoded values
    label_mapping = {label: index for index, label in enumerate(class_names)}
    # Convert the dictionary to a TensorFlow lookup table
    label_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(list(label_mapping.keys()), list(label_mapping.values())), 0)

    num_classes = len(np.unique(class_names))
    print(f'We have the following classes', label_mapping)

    def map_label(image, label):
        return image, label_table.lookup(label)

    dataset = dataset.map(map_label).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(map_label).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = dataset
    test_dataset = test_dataset

    mirrored_strategy = tf.distribute.MirroredStrategy()
    early_stopping = EarlyStopping(patience=10)
    log_dir = f'{log_path}'
    tensorboard = TensorBoard(
        log_dir=log_dir,
        write_images=True,
        write_steps_per_second=True,
        )
    # tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

    with mirrored_strategy.scope():
        if MODEL_NAME == 'alexnet':
            model = create_alexnet(input_shape, num_classes)
        elif MODEL_NAME == 'old':
            model = create_old_model(input_shape, num_classes)
        elif MODEL_NAME == 'vit':
            model = create_vit(input_shape, num_classes)
    model.summary()

    performance_metrics = PerformanceMetrics(validation_data=test_dataset,
        log_dir=log_dir, label_encoder=label_encoder)
    model.fit(train_dataset, epochs=50,
        validation_data=test_dataset,
        callbacks=[
            early_stopping,
            tensorboard,
            performance_metrics
            ])


class PerformanceMetrics(Callback):
    def __init__(self, validation_data, log_dir, label_encoder):
        super(PerformanceMetrics, self).__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.label_encoder = label_encoder
        self.writer = tf.summary.create_file_writer(f'{self.log_dir}/validation')
        self.image_writer = tf.summary.create_file_writer(f'{self.log_dir}/validation')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_pred_all = []
        val_targ_all = []

        # Loop over validation data
        for image, label in self.validation_data:
            val_targ = label.numpy()

            # Make predictions for batch
            y_pred = self.model.predict(image)
            y_pred = np.argmax(y_pred, axis=-1)

            y_pred_all.append(y_pred)
            val_targ_all.append(val_targ)

        y_pred_all = np.concatenate(y_pred_all, axis=0)
        val_targ_all = np.concatenate(val_targ_all, axis=0)

        # Compute metrics
        cm = confusion_matrix(val_targ_all, y_pred_all)
        accuracy = accuracy_score(val_targ_all, y_pred_all)
        precision = precision_score(val_targ_all, y_pred_all, average='macro')
        recall = recall_score(val_targ_all, y_pred_all, average='macro')
        f1 = f1_score(val_targ_all, y_pred_all, average='macro')

        # Add results to logs
        logs.update({
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_confusion_matrix': cm
        })

        with self.writer.as_default():
            tf.summary.scalar('val_accuracy', accuracy, step=epoch)
            tf.summary.scalar('val_precision', precision, step=epoch)
            tf.summary.scalar('val_recall', recall, step=epoch)
            tf.summary.scalar('val_f1', f1, step=epoch)
        
        with self.image_writer.as_default():
            figure = self.plot_confusion_matrix(cm,
                class_names=self.label_encoder.classes_)
            cm_image = self.plot_to_image(figure)
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


        print(f"— val_accuracy: {accuracy} — val_precision: {precision} — val_recall {recall} — val_f1: {f1}")
        print(f'— cm:\n{cm}')

        return

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        
        figure = plt.figure(figsize=(20, 20))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
    
    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        
        buf = io.BytesIO()
        
        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')
        
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        
        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
        
        return image

if __name__ == '__main__':
    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    if INPUT_DATASET == 'biwi':
        rgb, depth, labels = load_biwi(
            '/home/chenzz/projects/def-czarnuch/chenzz/rawdata/biwi',
            DATA_TYPE=='depth8'
            )
    elif INPUT_DATASET == 'mensa':
        rgb, depth, labels = load_mensa(
            '/home/chenzz/projects/def-czarnuch/chenzz/datasets/mensa_extracted',
            DATA_TYPE=='depth8'
            )
    elif INPUT_DATASET == 'inhouse':
        rgb, depth, labels = load_inhouse(
            '/home/chenzz/projects/def-czarnuch/chenzz/datasets/inhouse',
            DATA_TYPE=='depth8')
    else:
        raise ValueError('Unknown dataset', INPUT_DATASET)

    if DATA_TYPE == 'rgb':
        training, validation = rgb
        input_shape = (256, 256, 3)
    elif DATA_TYPE == 'depth' or DATA_TYPE == 'depth8':
        training, validation = depth
        input_shape = (256, 256, 1)
    else:
        raise ValueError('Unknown data type', DATA_TYPE)

    print('Training...')

    train_model(training, validation, labels, input_shape, LOG_PATH)
