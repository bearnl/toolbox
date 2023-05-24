import argparse
import tensorflow as tf

def combine_tfrecords(input_files, output_file):
    # Create a writer to the output TFRecord file
    writer = tf.io.TFRecordWriter(output_file)

    # Iterate over the input files
    for file in input_files:
        # Open the TFRecord file
        reader = tf.data.TFRecordDataset(file)

        # Iterate over the records in the input file
        for record in reader:
            # Write the record to the output file
            writer.write(record.numpy())

    # Close the writer
    writer.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Combine multiple TensorFlow TFRecord files.')
    parser.add_argument('input_files', nargs='+', help='Input TFRecord files')
    parser.add_argument('output_file', help='Output combined TFRecord file')
    args = parser.parse_args()

    # Combine TFRecord files
    combine_tfrecords(args.input_files, args.output_file)

if __name__ == '__main__':
    main()
