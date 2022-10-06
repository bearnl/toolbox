import sys
import argparse
from os import listdir
from os.path import isfile, join
import site
try:
  import cv2
except ImportError:
  site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
  import cv2
import numpy as np
import pathlib


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Pewview NPY file')
  parser.add_argument('input_path', action='store', help='path to the input npy file')
  args = parser.parse_args()

  npy_filename = args.input_path
  if not isfile(npy_filename):
    print('file not found:', npy_filename)
    sys.exit(1)

  print('reading', npy_filename)
  df = np.load(npy_filename)
  print('reading complete. lines=', len(df), 'shape', df.shape, 'type', df.dtype)
  for i in range(len(df)):
    print('previewing', i, '/', len(df))
    cv2.imshow('preview', df[i])
    if cv2.waitKey(0) == ord('q'):
      break
  print('done')
