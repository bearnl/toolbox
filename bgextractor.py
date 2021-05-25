import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import os
os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
import argparse
from pathlib import Path
import segment_k4a
import cv2
import numpy as np


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=Path, nargs=1)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_path: Path = args.input[0]
	if input_path.is_dir():
		parser.error('input can only be a recording file')

	output_path: Path = args.output[0]
	if output_path.is_dir():
		parser.error('output can only be a .npy file')

	ctx = segment_k4a.K4a()
	playback = ctx.playback_open(str(input_path))
	transformer = playback.get_calibration().transformation_create()

	output_dataset = []
	for cap in playback.get_capture_iter():
		print('processing new capture...')
		colour = cap.get_colour_image()
		depth = cap.get_depth_image()

		if colour is None or depth is None:
			print('colour or depth image is missing. skip this capture')
			continue

		colour_trans = transformer.color_image_to_depth_camera(colour, depth)
		depth_norm = cv2.cvtColor(
			cv2.normalize(depth, None, 0, 0xff, cv2.NORM_MINMAX, cv2.CV_8UC1),
			cv2.COLOR_GRAY2BGRA
			)
		img_preview = np.hstack((
			depth_norm,
			colour_trans
		))

		cv2.imshow('preview', img_preview)
		key = cv2.waitKey()
		if key == ord('q'):		# 'Q' key
			break
		elif key == ord('s'):
			print('select as background')
			output_dataset.append(depth)

	print('saving...')
	out: np.ndarray = np.asarray(output_dataset)
	print('output array size', out.shape)
	np.save(output_path, out)
	print('done')