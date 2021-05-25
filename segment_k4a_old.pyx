import logging
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.string cimport memset
from libc.stdlib cimport malloc, free
import numpy as np

import site
site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

root_logger = logging.getLogger(__name__)


cdef extern from "k4a/k4a.h":
	ctypedef struct k4a_calibration_t:
		pass
	ctypedef enum k4a_depth_mode_t:
		K4A_DEPTH_MODE_OFF,
		K4A_DEPTH_MODE_NFOV_2X2BINNED,
		K4A_DEPTH_MODE_NFOV_UNBINNED,
		K4A_DEPTH_MODE_WFOV_2X2BINNED,
		K4A_DEPTH_MODE_WFOV_UNBINNED,
		K4A_DEPTH_MODE_PASSIVE_IR
	ctypedef enum k4a_color_resolution_t:
		K4A_COLOR_RESOLUTION_OFF,
		K4A_COLOR_RESOLUTION_720P,
		K4A_COLOR_RESOLUTION_1080P,
		K4A_COLOR_RESOLUTION_1440P,
		K4A_COLOR_RESOLUTION_1536P,
		K4A_COLOR_RESOLUTION_2160P,
		K4A_COLOR_RESOLUTION_3072P,
	ctypedef struct k4a_transformation_t:
		pass
	ctypedef enum k4a_result_t:
		K4A_RESULT_SUCCEEDED,
		K4A_RESULT_FAILED
	ctypedef struct k4a_device_t:
		pass
	ctypedef enum k4a_buffer_result_t:
		K4A_BUFFER_RESULT_SUCCEEDED,
		K4A_BUFFER_RESULT_FAILED,
		K4A_BUFFER_RESULT_TOO_SMALL
	k4a_result_t k4a_device_open( uint32_t index, k4a_device_t * device_handle )
	void k4a_device_close ( k4a_device_t device_handle )
	k4a_result_t k4a_device_get_calibration (k4a_device_t device_handle,
		const k4a_depth_mode_t depth_mode,
		const k4a_color_resolution_t color_resolution,
		k4a_calibration_t * calibration )
	float k4a_capture_get_temperature_c ( k4a_capture_t capture_handle )
	ctypedef struct k4a_image_t:
		pass
	k4a_image_t k4a_capture_get_color_image ( k4a_capture_t capture_handle )
	k4a_image_t k4a_capture_get_depth_image ( k4a_capture_t capture_handle )
	size_t k4a_image_get_size ( k4a_image_t image_handle )
	void k4a_image_release ( k4a_image_t image_handle )
	uint64_t k4a_image_get_device_timestamp_usec ( k4a_image_t image_handle )
	ctypedef enum k4a_image_format_t:
		K4A_IMAGE_FORMAT_COLOR_MJPG,
		K4A_IMAGE_FORMAT_COLOR_NV12,
		K4A_IMAGE_FORMAT_COLOR_YUY2,
		K4A_IMAGE_FORMAT_COLOR_BGRA32,
		K4A_IMAGE_FORMAT_DEPTH16,
		K4A_IMAGE_FORMAT_IR16,
		K4A_IMAGE_FORMAT_CUSTOM8,
		K4A_IMAGE_FORMAT_CUSTOM16,
		K4A_IMAGE_FORMAT_CUSTOM
	k4a_image_format_t k4a_image_get_format ( k4a_image_t image_handle )
	int k4a_image_get_height_pixels ( k4a_image_t image_handle )
	int k4a_image_get_width_pixels ( k4a_image_t image_handle )
	int k4a_image_get_stride_bytes ( k4a_image_t image_handle )
	uint8_t * k4a_image_get_buffer ( k4a_image_t image_handle )
	k4a_transformation_t k4a_transformation_create ( const k4a_calibration_t * calibration )
	k4a_result_t k4a_transformation_depth_image_to_color_camera ( k4a_transformation_t transformation_handle,
		const k4a_image_t depth_image,
		k4a_image_t transformed_depth_image
	 )
	k4a_result_t k4a_image_create ( k4a_image_format_t format,
		int width_pixels,
		int height_pixels,
		int stride_bytes,
		k4a_image_t * image_handle 
	 )
	k4a_result_t k4a_transformation_color_image_to_depth_camera ( k4a_transformation_t transformation_handle,
		const k4a_image_t depth_image,
		const k4a_image_t color_image,
		k4a_image_t transformed_color_image
	 )

cdef extern from "k4arecord/playback.h":
	ctypedef struct k4a_playback_t:
		pass
	k4a_result_t k4a_playback_open ( const char * path, k4a_playback_t * playback_handle)
	void k4a_playback_close ( k4a_playback_t playback_handle)
	k4a_result_t k4a_playback_get_calibration ( k4a_playback_t playback_handle,
		k4a_calibration_t * calibration )
	size_t k4a_playback_get_track_count ( k4a_playback_t playback_handle)
	k4a_buffer_result_t k4a_playback_get_track_name ( k4a_playback_t playback_handle,
		size_t track_index,
		char * track_name,
		size_t * track_name_size)
	ctypedef enum k4a_stream_result_t:
		K4A_STREAM_RESULT_SUCCEEDED
		K4A_STREAM_RESULT_FAILED
		K4A_STREAM_RESULT_EOF
	ctypedef struct k4a_capture_t:
		pass
	k4a_stream_result_t k4a_playback_get_next_capture ( k4a_playback_t playback_handle,
		k4a_capture_t * capture_handle )


cdef class K4a(object):
	cdef logger

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def device_open(self, idx):
		dev = K4aDevice()
		self.logger.debug('Opening device %s' % idx)
		if k4a_result_t.K4A_RESULT_SUCCEEDED != k4a_device_open(idx, &dev._dev):
			self.logger.error('Open device %s failed' % idx)
			raise K4aException()
		self.logger.info('Device %s opened' % idx)
		return dev

	def playback_open(self, path):
		playback = K4aPlayback()
		self.logger.debug('Opening playback %s' % path)
		if k4a_result_t.K4A_RESULT_SUCCEEDED != k4a_playback_open(path.encode('UTF-8'), &playback._playback):
			self.logger.error('Open playback %s failed' % path)
			raise K4aException()
		self.logger.info('Playback %s opened' % path)
		return playback


cdef class K4aDevice(object):
	cdef logger
	cdef k4a_device_t _dev

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def get_calibration(self):
		self.logger.debug('Getting calibration from camera with depth mode %s, colour resolution %s' % ('?', '?'))
		calib = K4aCalibration()
		if k4a_result_t.K4A_RESULT_SUCCEEDED != k4a_device_get_calibration(
			self._dev,
			k4a_depth_mode_t.K4A_DEPTH_MODE_WFOV_UNBINNED,
			k4a_color_resolution_t.K4A_COLOR_RESOLUTION_1080P,
			&calib._calib
			):
			self.logger.error('Get calibration from camera failed')
			raise K4aException()
		self.logger.info('Get calibration success')
		return calib

	def close(self):
		self.logger.debug('Closing device')
		k4a_device_close(self._dev)
		self.logger.debug('Device closed')


cdef class K4aPlayback(object):
	cdef logger
	cdef k4a_playback_t _playback

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def get_calibration(self):
		self.logger.debug('Getting calibration from playback with depth mode %s, colour resolution %s' % ('?', '?'))
		calib = K4aCalibration()
		if k4a_result_t.K4A_RESULT_SUCCEEDED != k4a_playback_get_calibration(
			self._playback,
			&calib._calib
			):
			self.logger.error('Get calibration from playback failed')
			raise K4aException()
		self.logger.info('Get calibration success')
		return calib

	def get_track_count(self):
		return k4a_playback_get_track_count(self._playback)

	def get_track_name(self, idx):
		self.logger.debug('Getting track name for track %d' % idx)
		cdef size_t buf_size

		if k4a_buffer_result_t.K4A_BUFFER_RESULT_TOO_SMALL != k4a_playback_get_track_name(
			self._playback,
			idx,
			NULL,
			&buf_size
			):
			self.logger.error('Get track name size failed')
			raise K4aException()
		cdef char *buf = <char*>malloc(buf_size)
		memset(buf, 0x00, buf_size)
		if k4a_buffer_result_t.K4A_BUFFER_RESULT_SUCCEEDED != k4a_playback_get_track_name(
			self._playback,
			idx,
			buf,
			&buf_size
			):
			self.logger.error('Get track name failed')
			raise K4aException()
		str = buf.decode('UTF-8')
		free(buf)
		self.logger.debug('Track %d has name %s' % (idx, str))
		return str

	def get_record_configuration(self):
		pass

	def get_next_capture(self):
		self.logger.debug('Getting next capture')
		capture = K4aCapture()
		cdef k4a_stream_result_t ret = k4a_playback_get_next_capture(
			self._playback,
			&capture._capture
			)
		if ret == K4A_STREAM_RESULT_FAILED:
			self.logger.error('Get next capture failed with unknown reason')
			raise K4aException()
		elif ret == K4A_STREAM_RESULT_EOF:
			self.logger.error('Get next capture failed with EOF')
			raise K4aException()
		self.logger.info('Get next capture success')
		return capture

	def get_capture_iter(self):
		self.logger.debug('Getting capture iterator')
		cdef k4a_stream_result_t ret
		while True:
			capture = K4aCapture()
			ret = k4a_playback_get_next_capture(
				self._playback,
				&capture._capture
				)
			if ret == K4A_STREAM_RESULT_FAILED:
				self.logger.error('Get next capture failed with unknown reason')
				raise K4aException()
			elif ret == K4A_STREAM_RESULT_EOF:
				self.logger.error('Get next capture failed with EOF')
				break
			self.logger.info('Get next capture success')
			yield capture


cdef class K4aCapture(object):
	cdef logger
	cdef k4a_capture_t _capture

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def get_temperature_c(self):
		self.logger.debug('Getting temperature')
		cdef float temp = k4a_capture_get_temperature_c(self._capture)
		self.logger.debug('Temperature %f' % temp)
		return temp

	def get_colour_image(self):
		self.logger.debug('Getting colour image')
		cdef k4a_image_t img = k4a_capture_get_color_image(self._capture)
		img_wrap = K4aImage()
		img_wrap._image = img
		self.logger.debug('Get colour image success')
		return img_wrap

	def get_depth_image(self):
		self.logger.debug('Getting depth image')
		cdef k4a_image_t img = k4a_capture_get_depth_image(self._capture)
		img_wrap = K4aImage()
		img_wrap._image = img
		self.logger.debug('Get depth image success')
		return img_wrap


cdef class K4aImage(object):
	cdef logger
	cdef k4a_image_t _image

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def __bool__(self):
		return <int>self._image != 0x00
	
	def __enter__(self):
		return self

	def __exit__(self, exec_type, exec_val, exec_tb):
		self.close()

	def close(self):
		self.logger.debug('Close image')
		if not bool(self):
			self.logger.debug('Image is empty, skipped')
		else:
			k4a_image_release(self._image)
			self.logger.debug('Image closed')

	def get_size(self):
		self.logger.debug('Getting image size')
		cdef size_t size
		size = k4a_image_get_size(self._image)
		self.logger.debug('Get image size success with size of %d' % size)
		return size

	def get_device_timestamp_usec(self):
		self.logger.debug('Getting device timestamp')
		cdef uint64_t time = k4a_image_get_device_timestamp_usec(self._image)
		self.logger.debug('Get device timestamp success, ts %s' % time)
		return time

	def get_format(self):
		self.logger.debug('Getting image format')
		cdef uint32_t fmt = k4a_image_get_format(self._image)
		self.logger.debug('Get image format success, fmt %s' % fmt)
		return fmt

	def get_height_pixels(self):
		self.logger.debug('Getting image height')
		cdef int height = k4a_image_get_height_pixels(self._image)
		self.logger.debug('Get image height success, height %s' % height)
		return height

	def get_width_pixels(self):
		self.logger.debug('Getting image width')
		cdef int width = k4a_image_get_width_pixels(self._image)
		self.logger.debug('Get image width success, width %s' % width)
		return width

	def get_stride_bytes(self):
		self.logger.debug('Getting image stride bytes')
		cdef int bytes = k4a_image_get_stride_bytes(self._image)
		self.logger.debug('Get image stride bytes success, stride bytes %s' % bytes)
		return bytes

	def get_buffer(self):
		self.logger.debug('Getting image buffer')
		cdef int size = self.get_size()
		self.logger.debug('size: %s' % size)
		cdef uint8_t *buf = k4a_image_get_buffer(self._image)
		self.logger.debug('Getting image buffer success')
		return memoryview(<uint8_t[:size]>buf)

	def as_np(self):
		self.logger.debug('Converting to np array')

		buf = self.get_buffer()

		cdef k4a_image_format_t fmt = self.get_format()
		cdef int width = self.get_width_pixels()
		cdef int height = self.get_height_pixels()
		if fmt == K4A_IMAGE_FORMAT_DEPTH16 or fmt == K4A_IMAGE_FORMAT_IR16 or fmt == K4A_IMAGE_FORMAT_CUSTOM16:
			self.logger.debug('Convert and return 16UC1 image')
			img = np.frombuffer(buffer=buf, dtype=np.uint16).reshape((height, width))
			return img
		elif fmt == K4A_IMAGE_FORMAT_COLOR_BGRA32:
			self.logger.debug('Convert and return RGBA image')
			img = np.frombuffer(buffer=buf, dtype=np.uint8).reshape((height, width, 4))
			return img
		elif fmt == K4A_IMAGE_FORMAT_COLOR_MJPG:
			self.logger.debug('Convert and return JPEG image')
			img = cv2.imdecode(np.frombuffer(buffer=buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
			return img
		self.logger.debug('Unknown image, return raw data')
		img = np.frombuffer(buffer=buf, dtype=np.uint8)
		return img

	def create_empty(self):
		cdef k4a_image_format_t fmt = self.get_format()
		cdef int width = self.get_width_pixels()
		cdef int height = self.get_height_pixels()
		cdef int stride = self.get_stride_bytes()
		return self.create_empty_fmt(fmt, width, height, stride)

	def create_empty_ext(self, k4a_image_format_t fmt, int width, int height, int stride):
		img = K4aImage()
		if K4A_RESULT_SUCCEEDED != k4a_image_create(fmt, width, height, stride, &img._image):
			self.logger.error('Create image failed')
			raise K4aException()
		return img


cdef class K4aCalibration(object):
	cdef logger
	cdef k4a_calibration_t _calib;

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def transformation_create(self):
		self.logger.debug('Creating transformation')
		cdef k4a_transformation_t trans = k4a_transformation_create(&self._calib)
		self.logger.debug('Creating transformation success')
		transform = K4aTransformation()
		transform._transform = trans
		return transform


cdef class K4aTransformation(object):
	cdef logger
	cdef k4a_transformation_t _transform;

	def __init__(self):
		self.logger = root_logger.getChild(self.__class__.__name__)

	def depth_image_to_color_camera(self, K4aImage depth, K4aImage colour):
		self.logger.debug('Transforming depth image to colour camera')
		cdef K4aImage transformed = colour.create_empty_ext(depth.get_format(),
			colour.get_width_pixels(),
			colour.get_height_pixels(),
			colour.get_stride_bytes()
			)

		if K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_color_camera(
			self._transform,
			depth._image,
			transformed._image
			):
			self.logger.error('Transform depth image to colour camera failed')
			raise K4aException()
		return transformed

	def color_image_to_depth_camera(self, K4aImage colour, K4aImage depth):
		self.logger.debug('Transforming colour image to depth camera')
		cdef K4aImage transformed = depth.create_empty_ext(
			K4A_IMAGE_FORMAT_COLOR_BGRA32,
			depth.get_width_pixels(),
			depth.get_height_pixels(),
			depth.get_width_pixels()*4
			)
		if K4A_RESULT_SUCCEEDED != k4a_transformation_color_image_to_depth_camera(
			self._transform,
			depth._image,
			colour._image,
			transformed._image
			):
			self.logger.error('Transform colour image to depth camera failed')
			raise K4aException()
		return transformed


class K4aException(Exception):
	pass