import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import os
os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
import segment_k4a
from typing import List
import cv2
import numpy as np
import argparse
from pathlib import Path

# python .\segment.py --preview --bg-dataset .\mar-22-bear-dk-1-bg.npy .\rawdata\mycapture\mar-22-bear-dk-1.mkv mar-22-bear-dk-1-sliced.npy --cluster-idx 0 --cluster-size-min 30000 --range-low 100 --range-high 2500

def bg_subtract(input_img: np.ndarray, bg_img: np.ndarray,
	bg_threshold: int = 100,
	diff_threshold: int = 100) -> np.ndarray:
	bg_img[bg_img < diff_threshold] = 0

	mask_diff = np.uint8(cv2.threshold(
		cv2.absdiff(bg_img, input_img),
		bg_threshold,
		255,
		cv2.THRESH_BINARY
		)[1])
	mask = mask_diff
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
			print('found cluster')
			mask = np.logical_or(np.uint8((labels==i).astype(bool)), mask)
			cnt += 1
	mask = mask.astype(np.uint8)
	mask[mask > 0] = 255
	img = cv2.bitwise_and(img, img, mask=mask)
	print('cluster size', np.count_nonzero(img))
	return img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--range-low', type=int)
	parser.add_argument('--range-high', type=int)
	parser.add_argument('--bg-dataset', type=Path)
	parser.add_argument('--bg-threshold', type=int, default=100)
	parser.add_argument('--cluster-size-min', type=int)
	parser.add_argument('--cluster-size-max', type=int)
	parser.add_argument('--cluster-idx', type=str)
	parser.add_argument('--preview', action='store_true')
	parser.add_argument('--dump-original', type=str)
	parser.add_argument('input', type=Path, nargs=1)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_path: Path = args.input[0]
	if input_path.is_dir():
		parser.error('input can only be a recording file')

	output_path: Path = args.output[0]
	if output_path.is_dir():
		parser.error('output can only be a .npy file')

	bg_dataset: Path = args.bg_dataset
	if bg_dataset is not None:
		if bg_dataset.is_dir():
			parser.error('bg dataset can only be a .npy file')
		else:
			print('loading bg dataset')
			bg_img = np.mean(np.load(str(bg_dataset)), axis=0).astype(np.uint16)
			print('bg dataset loaded')

	print('arguments:', vars(args))
	preview: bool = args.preview
	if preview:
		cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

	# input_dataset = np.load(input_path)
	ctx = segment_k4a.K4a()
	playback = ctx.playback_open(input_path)
	track_count = playback.get_track_count()
	print('the recording includes', track_count, 'tracks')
	for i in range(track_count):
		print('track', i, ':', playback.get_track_name(i))
	transformer = playback.get_calibration().transformation_create()

	output_dataset = []
	dump_original: str = args.dump_original
	if dump_original:
		output_original = []
	for cap in playback.get_capture_iter():
		print('processing new capture...')
		colour = cap.get_colour_image()
		depth = cap.get_depth_image()
		if colour is None or depth is None:
			print('colour or depth image is missing. skip this capture')
			continue

		colour_trans = transformer.color_image_to_depth_camera(colour, depth)

		# img = input_dataset[i]
		# img = depth
		if preview:
			img_preview = cv2.cvtColor(
				cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
				cv2.COLOR_GRAY2BGRA
				)

		if args.bg_dataset is not None:
			bg_threshold: int = args.bg_threshold
			depth = bg_subtract(depth, bg_img, bg_threshold)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA)), axis=1)

		range_low: int = args.range_low
		range_high: int = args.range_high
		if bool(range_low) != bool(range_high):
			parser.error('range low and high must be both or neither presented')
		if not range_low or not range_high:
			print('no slicing')
		else:
			depth = bg_slice(depth, range_low, range_high)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA)), axis=1)

		cluster_size_min: int = args.cluster_size_min
		cluster_size_max: int = args.cluster_size_max
		cluster_idx: List[int] = []
		if args.cluster_idx is not None:
			cluster_idx = [int(x) for x in args.cluster_idx.strip().split(',')]
		if not cluster_idx and cluster_size_min is None and cluster_size_max is None:
			print('no clustering')
		else:
			depth = clustering(depth, cluster_idx, cluster_size_min, cluster_size_max)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA)), axis=1)

		if not np.any(depth):
			print('img is empty, skipped')
			continue
		mask = depth
		mask[mask > 0] = 0xff
		final = cv2.bitwise_and(colour_trans, colour_trans, mask=mask.astype(np.uint8))
		if preview:
			img_preview = np.concatenate(
					(img_preview, final), axis=1)

		if preview:
			cv2.imshow('preview', img_preview)
			if cv2.waitKey() == ord('q'):
				break
		if dump_original:
			output_original.append(colour_trans)
		output_dataset.append(final)

print('saving...')
out: np.ndarray = np.asarray(output_dataset)
print('output array size', out.shape)
np.save(output_path, out)
origial: np.ndarray = np.asarray(output_original)
np.save(dump_original, origial)
print('done')
