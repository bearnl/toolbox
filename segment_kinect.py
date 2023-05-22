import pathlib
import site
import os
from turtle import down
from typing import List, Optional, Tuple
import argparse
from pathlib import Path
import csv
import numpy as np
from common import get_images_from_recording, get_images_from_datasets
from connectivity import clustering as clustering_multi_foreground

try:
	os.add_dll_directory('C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin')
	import segment_k4a
except FileNotFoundError:
	print('Azure Kinect SDK DLL not found')
except ImportError:
	print('Unable to import segment_k4a')
except AttributeError:
	print('This is Linux and we cannot call add_dll_directory(). Please set LD_LIBRARY_PATH if the module cannot be imported')
	import segment_k4a

try:
	import cv2
except ImportError:
	site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

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
	ret = cv2.bitwise_and(img, img, mask=mask)
	print('bg slice', 'max value of input image after slicing', np.max(ret))
	return ret


def clustering(img: np.ndarray,
	size_min: int, size_max: int,
	centroid_exclude: Tuple[int, int, int, int]=(0, 0, 0, 0),
	bounding_boxes: List[Tuple[int, int, int, int]]=[],
	tolerance: Optional[int]=None,
	downsample: int=0,
	) -> List[np.ndarray]:
	print('downsample', downsample)
	downsampled = img
	for _ in range(downsample):
		downsampled = cv2.pyrDown(downsampled)
	if tolerance is None:
		print('tolerance is None, using OpenCV clustering')
		labels, centroids = clustering_traditional(downsampled)
	else:
		print('tolerance is', tolerance, 'using multi-foreground clustering')
		labels, centroids = clustering_multi_foreground(downsampled, tolerance)

	labels = labels.repeat(2**downsample, axis=0).repeat(2**downsample, axis=1)
	centroids = [(x * 2**downsample, y * 2**downsample) for (x, y) in centroids]

	unique_labels, unique_stats = np.unique(labels, return_counts=True)
	indices = np.flip(np.argsort(unique_stats))

	if not bounding_boxes:
		bounding_boxes = [{
			'bounding_box': (0, 0, img.shape[1], img.shape[0]),
			'label': 'default',
		}]
	print('bounding boxes', bounding_boxes)
	for item in bounding_boxes:
		bounding_box = item['bounding_box']

		# iterate through clusters
		for i in indices:
			mask = np.zeros(img.shape)
			label = unique_labels[i]
			if label == 0:
				print('skipping background')
				continue
			if size_max is not None and unique_stats[i] > size_max:
				print('cluster', i, 'larger than', size_max, ', skipped')
				continue
			elif size_min is not None and unique_stats[i] < size_min:
				print('cluster is smaller than', size_min, '. searching completed')
				break		# clusters are sorted by size, so we can break here
			if centroid_exclude[0] < centroids[i][0] < centroid_exclude[2] \
				and centroid_exclude[1] < centroids[i][1] < centroid_exclude[3]:
				print('cluster centroid is excluded', centroids[i])
				continue
			if not (bounding_box[0] < centroids[i][0] < bounding_box[2] \
				and bounding_box[1] < centroids[i][1] < bounding_box[3]):
				print('cluster centroid is not in bounding box', centroids[i])
				continue

			print('found cluster', i, unique_stats[i], centroids[i])
			mask = np.uint8((labels==label).astype(bool))

			mask = mask.astype(np.uint8)
			mask[mask > 0] = 255
			img_out = cv2.bitwise_and(img, img, mask=mask)
			img_out_size = np.count_nonzero(img_out)
			print('img cluster size', img_out_size)

			if size_max is not None and img_out_size > size_max:
				print('img cluster', i, 'larger than', size_max, ', skipped')
				continue
			elif size_min is not None and img_out_size < size_min:
				print('img cluster is smaller than', size_min, ', skipped')
				continue

			yield mask, item['label']
			break


def clustering_traditional(img: np.ndarray):
	binimg = np.uint8(cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1])
	# binimg = cv2.adaptiveThreshold(np.uint8(img), 255,
	# 	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 	cv2.THRESH_BINARY, 11, 2)
	_, labels, _, centroids = cv2.connectedComponentsWithStats(binimg)

	return labels, centroids


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--range-low', type=int)
	parser.add_argument('--range-high', type=int)
	parser.add_argument('--bg-dataset', type=Path)
	parser.add_argument('--bg-threshold', type=int, default=100)
	parser.add_argument('--bg-median', action='store_true', help='Use median filter')
	parser.add_argument('--cluster-size-min', type=int)
	parser.add_argument('--cluster-size-max', type=int)
	parser.add_argument('--cluster-idx', type=int, help='Select n-th largest cluster', default=None)
	parser.add_argument('--cluster-exclude-centroid', type=str, help='x1,y1,x2,y2', default='0,0,0,0')
	parser.add_argument('--cluster-tolerance', type=int, help='tolerance for multi-foreground clustering', default=None)
	parser.add_argument('--downsample', type=int, help='Downsample image by pyramid', default=0)
	parser.add_argument('--preview', action='store_true')
	parser.add_argument('--preview-timeout', type=int, default=0)
	parser.add_argument('--dump-original', type=str)
	parser.add_argument('--dump-raw-depth', type=str)
	parser.add_argument('--mensa-annotations', help='mensa dataset support', type=Path, default=None)
	parser.add_argument('--multicam', type=int, help='mensa dataset support', default=None)
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

	mensa_annotations_path: Path = args.mensa_annotations
	if mensa_annotations_path:
		base_path = pathlib.Path(mensa_annotations_path)
		mensa_annotations = {}	# filename: [annotations]
		for f in os.listdir(base_path):
			file_name = base_path / f
			print('loading mensa annotations from', file_name)
			with open(file_name, 'r') as fp:
				csv_reader = csv.reader(fp, delimiter=' ')
				for row in csv_reader:
					if row[0].strip() == '#':
						continue
					label = file_name.stem.split('_')[1]
					fn = row[0]
					x, y = int(row[2]), int(row[3])
					w, h = int(row[4]), int(row[5])

					if row[0] not in mensa_annotations:
						mensa_annotations[row[0]] = []
					mensa_annotations[row[0]].append({
						'bounding_box': (x, y, x+w, y+h),
						'label': label
					})
		# print('mensa annotations loaded', mensa_annotations)

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

	for seq, (depth, colour, colour_trans) in enumerate(img_iter):
		print('processing seq', seq)
		if colour is not None and colour.shape[2] == 3:
			print('colour image has only 3 channels. Adding alpha channel...')
			colour = cv2.cvtColor(colour, cv2.COLOR_BGR2BGRA)
		if colour_trans is None:
			print('no colour_trans image found, use colour image instead')
			colour_trans = colour
		if depth is not None and len(depth.shape) >=3 and depth.shape[2] == 3:
			print('depth image has 3 channels. Converting to grayscale...')
			depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

		raw_depth = depth
		if preview:
			img_preview = cv2.putText(cv2.cvtColor(
				cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
				cv2.COLOR_GRAY2BGRA
				), 'NORM', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

		if args.bg_dataset is not None:
			bg_threshold: int = args.bg_threshold
			depth = bg_subtract(depth, bg_img, bg_threshold)
			if preview:
				img_preview = np.concatenate(
					(img_preview, cv2.putText(cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA),
						'BG SUB', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)), axis=1)
		else:
			print('no background removal')

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
					(img_preview, cv2.putText(cv2.cvtColor(cv2.normalize(
						depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
						cv2.COLOR_GRAY2BGRA), 'SLICE', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)), axis=1)

		cluster_size_min: int = args.cluster_size_min
		cluster_size_max: int = args.cluster_size_max
		cluster_idx: Optional[int] = None
		downsample: int = args.downsample
		cluster_tolerance: Optional[int] = args.cluster_tolerance
		cluster_exclude_centroid: Tuple[int, int, int, int] = [
			int(x) for x in args.cluster_exclude_centroid.strip().split(',')]
		multicam: Optional[int] = args.multicam
		if args.cluster_idx is not None:
			cluster_idx = args.cluster_idx
		if not cluster_idx and cluster_size_min is None and cluster_size_max is None:
			print('no clustering')
		else:
			if preview:
				img_preview_base = img_preview
			for i, (depth, label) in enumerate(clustering(depth,
				cluster_size_min, cluster_size_max,
				cluster_exclude_centroid,
				tolerance=cluster_tolerance,
				bounding_boxes=mensa_annotations.get(f'seq0_{seq:04d}_{multicam}', []),
				downsample=args.downsample)):
				print('received cluster', i)
				if cluster_idx and i != cluster_idx:
					print('skipping cluster', i)
					continue
				if preview:
					img_preview = np.concatenate(
						(img_preview_base, cv2.putText(cv2.cvtColor(cv2.normalize(
							depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1),
							cv2.COLOR_GRAY2BGRA), 'CLUSTERING', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)), axis=1)

				mask = depth
				mask[mask > 0] = 0xff
				final = cv2.bitwise_and(colour_trans, colour_trans, mask=mask.astype(np.uint8))
				if preview:
					img_preview = np.concatenate(
							(img_preview, cv2.putText(final, 'FINAL', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)), axis=1)

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
		output_dataset.append(colour_trans)

print('saving...')
df = output_dataset
out: np.ndarray = np.asarray(df)
print('output array size', out.shape)
np.save(f'{output_path}', out)
if dump_original:
	origial: np.ndarray = np.asarray(output_original)
	np.save(f'{dump_original}', origial)
if dump_raw_depth:
	raw_depth: np.ndarray = np.asarray(output_raw_depth)
	np.save(f'{dump_raw_depth}', raw_depth)
print('done')
