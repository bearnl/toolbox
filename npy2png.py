import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import os
os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
import argparse
from pathlib import Path
import cv2
import numpy as np


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('input', type=Path, nargs=1)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_path: Path = args.input[0]
	output_path: Path = args.output[0]
	if not input_path.is_dir() and output_path.is_dir():
		print('converting npy to png...')

		dataset = np.load(input_path)
		for idx, row in enumerate(dataset):
			filename = '%s/%s.png' % (output_path, str(idx))
			print('writing', filename)
			cv2.imwrite(filename, row)

	if input_path.is_dir() and not output_path.is_dir():
		print('converting png to npy...')
		output_dataset = []
		for file in input_path.glob('*.png'):
			filename = file.as_posix()
			print('reading', filename)
			img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
			output_dataset.append(img)
		out: np.ndarray = np.asarray(output_dataset)
		print('output array size', out.shape)
		np.save(output_path, out)

	print('done')
