import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import os
os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
import segment_k4a
from typing import List, Tuple
import cv2
import numpy as np
import argparse
from pathlib import Path

# python .\segment.py --preview --bg-dataset .\mar-22-bear-dk-1-bg.npy .\rawdata\mycapture\mar-22-bear-dk-1.mkv mar-22-bear-dk-1-sliced.npy --cluster-idx 0 --cluster-size-min 30000 --range-low 100 --range-high 2500

def bg_subtract(input_img: np.ndarray, bg_img: np.ndarray,
	bg_threshold: int = 100,
	diff_threshold: int = 0) -> np.ndarray:
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


def clustering(img: np.ndarray, idx: List[int],
	size_min: int, size_max: int,
	centroid_exclude: Tuple[int, int, int, int]=(0, 0, 0, 0)) -> np.ndarray:
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
			print('cluster is smaller than', size_min, '. searching completed')
			break
		elif centroid_exclude[0] < centroids[i][0] < centroid_exclude[2] \
			and centroid_exclude[1] < centroids[i][1] < centroid_exclude[3]:
			print('cluster centroid is excluded', centroids[i])
			continue
		elif not idx or cnt in idx:
			print('found cluster', i, stats[i], centroids[i])
			mask = np.logical_or(np.uint8((labels==i).astype(bool)), mask)
			cnt += 1
	mask = mask.astype(np.uint8)
	mask[mask > 0] = 255
	img = cv2.bitwise_and(img, img, mask=mask)
	print('cluster size', np.count_nonzero(img))
	return img


def get_images_from_recording(playback):
	transformer = playback.get_calibration().transformation_create()
	for cap in playback.get_capture_iter():
		print('processing new capture...')
		colour = cap.get_colour_image()
		depth = cap.get_depth_image()
		if colour is None or depth is None:
			print('colour or depth image is missing. skip this capture')
			continue

		colour_trans = transformer.color_image_to_depth_camera(colour, depth)
		yield depth, colour, colour_trans

def get_images_from_datasets(depth: Path, colour: Path=None):
		depth_dataset = np.load(depth)
		colour_dataset = None
		if colour:
			colour_dataset = np.load(colour)
		for i in range(len(depth_dataset)):
			if colour:
				yield depth_dataset[i], colour_dataset[i], None
			else:
				yield depth_dataset[i], None, None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--range-low', type=int)
	parser.add_argument('--range-high', type=int)
	parser.add_argument('--bg-dataset', type=Path)
	parser.add_argument('--bg-threshold', type=int, default=100)
	parser.add_argument('--bg-median', action='store_true', help='Use median filter')
	parser.add_argument('--cluster-size-min', type=int)
	parser.add_argument('--cluster-size-max', type=int)
	parser.add_argument('--cluster-idx', type=str)
	parser.add_argument('--cluster-exclude-centroid', type=str, help='x1,y1,x2,y2', default='0,0,0,0')
	parser.add_argument('--preview', action='store_true')
	parser.add_argument('--preview-timeout', type=int, default=0)
	parser.add_argument('--dump-original', type=str)
	parser.add_argument('--dump-raw-depth', type=str)
	parser.add_argument('--no-skip-empty-image', action='store_true')
	group = parser.add_argument_group()
	group_numpy = group.add_argument_group()
	group_numpy.add_argument('--colour-dataset', type=Path)
	group_numpy.add_argument('--depth-dataset', type=Path)
	group.add_argument('--recording', type=Path)
	parser.add_argument('output', type=Path, nargs=1)
	args = parser.parse_args()

	input_recording: Path = args.recording
	input_colour: Path = args.colour_dataset
	input_depth: Path = args.depth_dataset
	if args.recording:
		if input_recording.is_dir():
			parser.error('input can only be a recording')
	elif args.depth_dataset:
		if input_colour.is_dir() or input_depth.is_dir():
			parser.error('input can only be .npy files')
	else:
		parser.error('either recording or colour and depth dataset must be provided')

	output_path: Path = args.output[0]
	if output_path.is_dir():
		parser.error('output can only be a .npy file')

	bg_dataset: Path = args.bg_dataset
	if bg_dataset is not None:
		if bg_dataset.is_dir():
			parser.error('bg dataset can only be a .npy file')
		else:
			print('loading bg dataset')
			if not args.bg_median:
				bg_img = np.mean(np.load(str(bg_dataset)), axis=0).astype(np.uint16)
			else:
				bg_img = np.median(np.load(str(bg_dataset)), axis=0).astype(np.uint16)
			print('bg dataset loaded')

	print('arguments:', vars(args))
	preview: bool = args.preview
	if preview:
		cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

	if input_recording and input_recording.exists():
		print('loading recording file')
		ctx = segment_k4a.K4a()
		playback = ctx.playback_open(input_recording)
		track_count = playback.get_track_count()
		print('the recording includes', track_count, 'tracks')
		for i in range(track_count):
			print('track', i, ':', playback.get_track_name(i))
		img_iter = get_images_from_recording(playback)
	elif input_depth and input_depth.exists():
		print('loading numpy dataset')
		img_iter = get_images_from_datasets(input_depth, input_colour)

	output_dataset = []
	dump_original: str = args.dump_original
	if dump_original:
		output_original = []
	dump_raw_depth: str = args.dump_raw_depth
	if dump_raw_depth:
		output_raw_depth = []

	for depth, colour, colour_trans in img_iter:
		if colour is not None and colour.shape[2] == 3:
			print('colour image has only 3 channels. Adding alpha channel...')
			colour = cv2.cvtColor(colour, cv2.COLOR_BGR2BGRA)
		if colour_trans is None:
			print('no colour_trans image found, use colour image instead')
			colour_trans = colour

		raw_depth = depth
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
		cluster_exclude_centroid: Tuple[int, int, int, int] = [
			int(x) for x in args.cluster_exclude_centroid.strip().split(',')]
		if args.cluster_idx is not None:
			cluster_idx = [int(x) for x in args.cluster_idx.strip().split(',')]
		if not cluster_idx and cluster_size_min is None and cluster_size_max is None:
			print('no clustering')
		else:
			depth = clustering(depth, cluster_idx, cluster_size_min, cluster_size_max, cluster_exclude_centroid)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA)), axis=1)

		mask = depth
		mask[mask > 0] = 0xff
		final = cv2.bitwise_and(colour_trans, colour_trans, mask=mask.astype(np.uint8))
		if preview:
			img_preview = np.concatenate(
					(img_preview, final), axis=1)

		empty_img = not np.any(depth)
		if preview:
			cv2.imshow('preview', img_preview)
			if not args.no_skip_empty_image and empty_img:
				k_timeout = 1
			else:
				k_timeout = args.preview_timeout
			if cv2.waitKey(k_timeout) == ord('q'):
				break

		if empty_img:
			print('img is empty, skipped')
			continue

		if dump_original:
			output_original.append(colour_trans)
		if dump_raw_depth:
			output_raw_depth.append(raw_depth)
		output_dataset.append(final)

print('saving...')
out: np.ndarray = np.asarray(output_dataset)
print('output array size', out.shape)
np.save(output_path, out)
if dump_original:
	origial: np.ndarray = np.asarray(output_original)
	np.save(dump_original, origial)
if dump_raw_depth:
	raw_depth: np.ndarray = np.asarray(output_raw_depth)
	np.save(dump_raw_depth, raw_depth)
print('done')
