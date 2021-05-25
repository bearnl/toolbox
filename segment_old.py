import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import os
os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
from typing import List
import cv2
import numpy as np
import argparse
from pathlib import Path


def bg_subtract(input_img: np.ndarray, bg_img: np.ndarray,
	bg_threshold: int=100,
	bg_skip_zeros: bool=False,
	diff_threshold: int=100) -> np.ndarray:
	bg_img[bg_img<diff_threshold] = 0
	mask_diff = np.uint8(cv2.threshold(
		cv2.absdiff(bg_img, input_img),
		bg_threshold,
		255,
		cv2.THRESH_BINARY
		)[1])
	mask = mask_diff
	if bg_skip_zeros:
		mask_non0 = (input_img != 0) & (bg_idx != 0)
		mask = np.where(mask_non0, mask_diff, 0)
	return cv2.bitwise_and(input_img, input_img, mask=mask)


def bg_slice(img: np.ndarray, range_low: int, range_high: int) -> np.ndarray:
	mask = cv2.inRange(img, range_low, range_high)
	return cv2.bitwise_and(img, img, mask=mask)


def clustering(img: np.ndarray, idx: List[int], size_min: int, size_max: int) -> np.ndarray:
	binimg = np.uint8(cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1])
	_, labels, _, centroids = cv2.connectedComponentsWithStats(binimg)
	stats = np.unique(labels, return_counts=True)[1]
	indices = np.delete(np.argsort(stats)[::-1], 0)

	cnt = 0
	mask = np.zeros(img.shape)
	for i in indices:
		if size_max is not None and stats[i] > size_max:
			print('cluster', i, 'larger than', size_max, ', skipped')
			continue
		elif size_min is not None and stats[i] < size_min:
			print('clusters is smaller than', size_min, '. searching completed with nothing found')
			break
		elif not idx or cnt in idx:
			mask = np.logical_or(np.uint8((labels==i).astype(bool)), mask)
			cnt += 1
	mask = mask.astype(np.uint8)
	mask[mask > 0] = 255
	img = cv2.bitwise_and(img, img, mask=mask)
	print('cluster size', np.count_nonzero(img))
	return img


class SplitArgs(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, values.split(','))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--range-low', type=int)
	parser.add_argument('--range-high', type=int)
	parser.add_argument('--bg-idx', type=str)
	parser.add_argument('--bg-threshold', type=int, default=100)
	parser.add_argument('--bg-skip-zeros', action='store_true', default=False)
	parser.add_argument('--cluster-size-min', type=int)
	parser.add_argument('--cluster-size-max', type=int)
	parser.add_argument('--cluster-idx', type=str)
	parser.add_argument('--preview', action='store_true')
	parser.add_argument('input', type=Path, nargs=1)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_path: Path = args.input[0]
	if input_path.is_dir():
		parser.error('input can only be a .npy file')

	output_path: Path = args.output[0]
	if output_path.is_dir():
		parser.error('output can only be a .npy file')

	print('arguments:', vars(args))
	preview: bool = args.preview
	if preview:
		cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

	input_dataset = np.load(input_path)

	output_dataset = []
	for i in range(0, len(input_dataset)):
		print('processing row', i)

		img = input_dataset[i]
		if preview:
			img_preview = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

		bg_idx: List[int] = []
		if args.bg_idx is not None:
			bg_idx = [int(x) for x in args.bg_idx.strip().split(',')]
		if not bg_idx:
			print('no background subtraction')
		elif i in bg_idx:
			print('this image is the bg img, skipped')
			continue
		else:
			bg_threshold: int = args.bg_threshold
			bg_skip_zeros: bool = args.bg_skip_zeros
			img = bg_subtract(img,
				np.mean([input_dataset[i] for i in bg_idx], axis=0).astype(np.uint16),
				bg_threshold,
				bg_skip_zeros=bg_skip_zeros)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.normalize(
						img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
						), axis=1)

		range_low: int = args.range_low
		range_high: int = args.range_high
		if bool(range_low) != bool(range_high):
			parser.error('range low and high must be both or neither presented')
		if not range_low or not range_high:
			print('no slicing')
		else:
			img = bg_slice(img, range_low, range_high)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.normalize(
						img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
						), axis=1)

		cluster_size_min: int = args.cluster_size_min
		cluster_size_max: int = args.cluster_size_max
		cluster_idx: List[int] = []
		if args.cluster_idx is not None:
			cluster_idx = [int(x) for x in args.cluster_idx.strip().split(',')]
		if not cluster_idx and cluster_size_min is None and cluster_size_max is None:
			print('no clustering')
		else:
			img = clustering(img, cluster_idx, cluster_size_min, cluster_size_max)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.normalize(
						img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
						), axis=1)

		if preview:
			cv2.imshow('preview', img_preview)
			if cv2.waitKey() == 113:	# 'Q' key
				break

		if not np.any(img):
			print('img is empty, skipped')
			continue
		output_dataset.append(img)

print('saving...')
out:np.ndarray = np.asarray(output_dataset)
print('output array size', out.shape)
np.save(output_path, out)
print('done')
