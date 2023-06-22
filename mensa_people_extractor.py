# This tool is to extract people from mensa dataset
# with their annotations
#
# Ref: RGB-D People Dataset
# L. Spinello, M. Luber, K. Arras
# Univ. of Freiburg, 2011

import os
import re
import site

import numpy as np

try:
	import cv2
except ImportError:
	site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

DATASET_PATH = './mensa_seq0_1.1'
OUTPUT_PATH = './mensa_extracted'

def clip_coordinates(x, y, w, h, img_width, img_height):
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    return x, y, w, h

def load_depth_image(file_path):
    depth_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    # if np.max(depth_image) > 10**4:
    #     depth_image = np.right_shift(np.bitwise_and(depth_image, 0xFF00), 8) | np.left_shift(np.bitwise_and(depth_image, 0x00FF), 8)
    return depth_image

def main():
    annotations_path = os.path.join(DATASET_PATH, "track_annotations")
    rgb_path = os.path.join(DATASET_PATH, "rgb")
    depth_path = os.path.join(DATASET_PATH, "depth")

    for track_file in os.listdir(annotations_path):
        participant_num = re.findall(r'\d+', track_file)[0]

        participant_dir = os.path.join(OUTPUT_PATH, f"{participant_num}")
        os.makedirs(participant_dir, exist_ok=True)

        with open(os.path.join(annotations_path, track_file), "r") as f:
            for line in f:
                if not line.startswith("#"):
                    data = line.strip().split(" ")
                    image_name, ts, x_d, y_d, w_d, h_d, x_rgb, y_rgb, w_rgb, h_rgb, vsb = data
                    if vsb == "0":
                        print(f'{track_file}: {image_name} is not visible, skipping...')
                    frame_num = image_name.split('_')[1]
                    
                    rgb_image_file = os.path.join(rgb_path, image_name + ".ppm")
                    depth_image_file = os.path.join(depth_path, image_name + ".pgm")
                    
                    rgb_image = cv2.imread(rgb_image_file)
                    depth_image = load_depth_image(depth_image_file)

                    x_d, y_d, w_d, h_d = map(int, (x_d, y_d, w_d, h_d))
                    x_rgb, y_rgb, w_rgb, h_rgb = map(int, (x_rgb, y_rgb, w_rgb, h_rgb))

                    depth_img_height, depth_img_width = depth_image.shape
                    rgb_img_height, rgb_img_width, _ = rgb_image.shape

                    x_d, y_d, w_d, h_d = clip_coordinates(x_d, y_d, w_d, h_d, depth_img_width, depth_img_height)
                    x_rgb, y_rgb, w_rgb, h_rgb = clip_coordinates(x_rgb, y_rgb, w_rgb, h_rgb, rgb_img_width, rgb_img_height)

                    rgb_crop = rgb_image[y_rgb:y_rgb + h_rgb, x_rgb:x_rgb + w_rgb]
                    depth_crop = depth_image[y_d:y_d + h_d, x_d:x_d + w_d]

                    output_rgb_file = os.path.join(participant_dir, f"{frame_num}_rgb.png")
                    output_depth_file = os.path.join(participant_dir, f"{frame_num}_depth.png")

                    cv2.imwrite(output_rgb_file, rgb_crop)
                    cv2.imwrite(output_depth_file, depth_crop)

if __name__ == "__main__":
    main()
