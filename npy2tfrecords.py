import numpy as np
import tensorflow as tf
import os
import argparse


def load_npy_file(file_path):
    return np.load(file_path)

def chunks(np_array, chunk_size=100):
    """Yield successive chunk_size chunks from np_array."""
    for i in range(0, len(np_array), chunk_size):
        yield np_array[i:i + chunk_size]

def load_and_convert_to_tf_dataset(directory, rgb_filename, depth_filename):
    rgb_writer = tf.io.TFRecordWriter(rgb_filename)
    depth_writer = tf.io.TFRecordWriter(depth_filename)
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            full_path = os.path.join(directory, file)
            print('Loading', full_path)
            np_array = load_npy_file(full_path)
            participant_name = file.split("_")[0]  # Assuming the name is before the ".npy"
            print('Loaded. Label set to', participant_name)

            for frame in np_array:  # iterating over each frame
                image = tf.convert_to_tensor(frame)
                tf_example = image_example(image, participant_name)

                if file.endswith('rgb.npy'):
                    rgb_writer.write(tf_example.SerializeToString())
                elif file.endswith('depth.npy'):
                    depth_writer.write(tf_example.SerializeToString())

    rgb_writer.close()
    depth_writer.close()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    """Returns a bytes_list from a string."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image, label):
    feature = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'channels': _int64_feature(image.shape[2] if len(image.shape) > 2 else 1),  # handle grayscale images
        'label': _string_feature(label),
        'image_raw': _bytes_feature(tf.io.serialize_tensor(image).numpy()),  # store raw image data
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def inspect_tfrecord(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    count = 0

    for parsed_record in parsed_dataset:
        count += 1
        height = parsed_record['height'].numpy()
        width = parsed_record['width'].numpy()
        channels = parsed_record['channels'].numpy()

        dtype = tf.uint8
        if channels == 1:
            # depth image
            dtype = tf.uint16
        img_tensor = tf.io.parse_tensor(parsed_record['image_raw'].numpy(), out_type=dtype)

        print(f"Record {count} - Height: {height}, Width: {width}, Channels: {channels}")
        print(f"Image Tensor Shape: {img_tensor.shape}")

    print(f"Total Records: {count}")


def main(args):
    # Sanity checks
    if not os.path.isdir(args.directory):
        raise ValueError(f"{args.directory} is not a valid directory")

    if not args.output:
        raise ValueError("Output filename cannot be empty")

    rgb_filename = f'{args.output}-rgb.tfrecords'
    depth_filename = f'{args.output}-depth.tfrecords'

    if args.inspect:
        print("Inspecting RGB tfrecords")
        inspect_tfrecord(rgb_filename)
        print("Inspecting Depth tfrecords")
        inspect_tfrecord(depth_filename)
    else:
        load_and_convert_to_tf_dataset(args.directory, rgb_filename, depth_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert npy files to tfrecord files')
    parser.add_argument('directory', type=str, help='Directory containing npy files')
    parser.add_argument('output', type=str, help='Output tfrecord filename without extension')
    parser.add_argument('--inspect', action='store_true', help='Inspect tfrecord files')

    args = parser.parse_args()
    main(args)
