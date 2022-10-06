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


def read_images(path, mode=cv2.IMREAD_COLOR, resize=None, multicam=None):
  base_path = pathlib.Path(path)
  img_list = []
  for f in listdir(path):
    name = base_path / f
    if multicam:
      name_tmp = name.stem.split('_')
      if name_tmp.pop() != multicam:
        continue
    if not isfile(name):
      continue
    print('reading', path, f, end=' ')
    img = cv2.imread('%s/%s' % (path, f), mode)
    print('shape', img.shape)
    if resize is not None:
      tmp = resize.split('x')
      img = cv2.resize(img, (int(tmp[1]), int(tmp[0])), interpolation=cv2.INTER_AREA)
    img_list.append(img)
  return np.array(img_list)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert images to npy')
  parser.add_argument('input_path', action='store', help='the directory continas the images')
  parser.add_argument('output_path', action='store', help='path to the npy file')
  parser.add_argument('--mode', choices=['anydepth', 'color', 'grayscale'],
    help='specify image type', default='color')
  parser.add_argument('--resize', help='downsample image size, e.g. 128x256', default=None)
  parser.add_argument('--multicam', help='mensa dataset support', default=None)
  args = parser.parse_args()

  path = args.input_path
  npy_filename = args.output_path
  mode = cv2.IMREAD_COLOR
  if args.mode == 'anydepth':
    mode = cv2.IMREAD_ANYDEPTH
  elif args.mode == 'color':
    mode = cv2.IMREAD_COLOR
  elif args.mode == 'grayscale':
    mode = cv2.IMREAD_GRAYSCALE

  arr = read_images(path, mode, args.resize, args.multicam)
  print('reading complete. lines=', len(arr), 'saving...')
  print('total matrix size', arr.shape, 'type', arr.dtype)
  np.save(npy_filename, arr)
  print('done')
