import site
import os
try:
	import cv2
except ImportError:
	site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

try:
	os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
	import segment_k4a
except FileNotFoundError:
	print('Azure Kinect SDK DLL not found')
except ImportError:
	print('Unable to import segment_k4a')
import argparse
from pathlib import Path
import numpy as np
from common import get_images_from_recording, get_images_from_datasets


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=Path, nargs=1)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_path: Path = args.input[0]
	if input_path.is_dir():
		parser.error('input can only be a recording file or numpy file')

	output_path: Path = args.output[0]
	if output_path.is_dir():
		parser.error('output can only be a .npy file')

	if input_path.suffix == '.mkv':
		ctx = segment_k4a.K4a()
		playback = ctx.playback_open(str(input_path))
		img_iter = get_images_from_recording(playback)
	if input_path.suffix == '.npy':
		img_iter = get_images_from_datasets(input_path)
	if not img_iter:
		parser.error('input file type is not supported')

	output_dataset = []
	for depth, colour, colour_trans in img_iter:
		print('processing new capture...')

		depth_norm = cv2.cvtColor(
			cv2.normalize(depth, None, 0, 0xff, cv2.NORM_MINMAX, cv2.CV_8UC1),
			cv2.COLOR_GRAY2BGRA
			)
		if colour_trans:
			img_preview = np.hstack((
				depth_norm,
				colour_trans
			))
		else:
			img_preview = depth_norm

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
