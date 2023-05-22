import glob
import os
import site

import numpy as np

try:
	import cv2
except ImportError:
	site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

DATASET_PATH = '/home/chenzz/datasets/biwi/Training'
OUTPUT_PATH = '/home/chenzz/datasets/biwi'

# Iterate through all sub-directories in DATASET_PATH
for subdir in os.listdir(DATASET_PATH):
    subdir_path = os.path.join(DATASET_PATH, subdir)

    # Check if the path is a directory
    if os.path.isdir(subdir_path):
        print('Processing sub-directory:', subdir_path, '...')
        # Create dictionaries to store image data
        rgb_data = {}
        depth_data = {}

        # Iterate through all image files in the sub-directory
        for file in glob.glob(os.path.join(subdir_path, '*_rgb.jpg')):
            # Read the image file using OpenCV
            img = cv2.imread(file)
            # Get the participant number from the file name
            participant_num = os.path.basename(file).split('_')[0][-3:]

            # Store the image data in the appropriate dictionary
            if participant_num not in rgb_data:
                rgb_data[participant_num] = []
            rgb_data[participant_num].append(img)

        for file in glob.glob(os.path.join(subdir_path, '*_depth.pgm')):
            # Read the image file using OpenCV
            img = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
            # Get the participant number from the file name
            participant_num = os.path.basename(file).split('_')[0][-3:]

            # Store the image data in the appropriate dictionary
            if participant_num not in depth_data:
                depth_data[participant_num] = []
            depth_data[participant_num].append(img)

        # Save the image data as .npy files
        for participant_num, img_array in rgb_data.items():
            np.save(os.path.join(OUTPUT_PATH, f"{participant_num}_rgb.npy"), img_array)
        for participant_num, img_array in depth_data.items():
            np.save(os.path.join(OUTPUT_PATH, f"{participant_num}_depth.npy"), img_array)

        print("All images have been converted to .npy files.")

