from pathlib import Path
import numpy as np


def get_images_from_recording(playback):
	transformer = playback.get_calibration().transformation_create()
	for cap in playback.get_capture_iter():
		print('processing new capture...')
		colour = cap.get_colour_image()
		depth = cap.get_depth_image()
		if colour is None or depth is None:
			print('colour or depth image is missing. skip this capture')
			continue
		try:
			colour_trans = transformer.color_image_to_depth_camera(colour, depth)
		except:
			colour_trans = None
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
