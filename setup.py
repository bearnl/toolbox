from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
	Extension("segment_k4a", ["segment_k4a.pyx"],
		include_dirs=[
			'C:/Program Files/Azure Kinect SDK v1.4.1/sdk/include',
			'./kinectsdk/build/native/include'
		],
		library_dirs=[
			'C:/Program Files/Azure Kinect SDK v1.4.1/sdk/windows-desktop/amd64/release/lib',
			'./kinectsdk/linux/lib/native/x64/release'
		],
		libraries=['k4a', 'k4arecord'],
		),
]
setup(
	name='segment k4a',
	ext_modules=cythonize(extensions),
	zip_safe=False
)
