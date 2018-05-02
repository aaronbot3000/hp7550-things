#!/usr/bin/python3
import sys
import cv2
import numpy as np
import random

from collections import deque
from collections import namedtuple
from rtree import index as rindex


RESOLUTION = 0.025  # mm/quanta
DECIMATE = 1

# Tabloid
#PAGE_HEIGHT = 254  # mm
#PAGE_WIDTH = 398.37  # mm

# Letter
PAGE_HEIGHT = 184  # mm
PAGE_WIDTH = 254  # mm
PAGE_HEIGHT_OFFSET = 11.5  # mm

DITHER = False

COLORS = [
    (0, 0, 0),  # black
    (212, 100, 45),  # blue
    (0, 0, 255),  # red
    (64, 195, 35),  # green
    (30, 82, 140),  # brown
    (211, 24, 211),  # magenta
    (0, 255, 255),  # yellow
]

# Image size
IMAGE_HEIGHT = PAGE_HEIGHT
IMAGE_WIDTH = PAGE_WIDTH

ScaleInfo = namedtuple('ScaleInfo', ['width_offset',
                                     'height_offset',
                                     'output_scale'])

def clamp(value, min_v, max_v):
  return min(max(value, min_v), max_v)

def collect_dots(image, color_list, dither=True):
  cols = len(image[0])
  rows = len(image)

  color_dots = []
  for _ in color_list:
    color_dots.append(deque())
  for row in range(rows):
    if row % 10 == 0:
      print('row %d of %d' % (row, rows))
    for column in range(cols):
      for color, color_deque in zip(color_list, color_dots):
        if np.all(image[row][column] == color):
          if dither:
            noise_r = clamp(random.random() * 1.5 - 0.75, 0, rows)
            noise_c = clamp(random.random() * 1.5 - 0.75, 0, cols)
            color_deque.append((row + noise_r, column + noise_c))
          else:
            color_deque.append((row, column))

  return color_dots

def dot_to_box(dot):
  return (dot[0], dot[1], dot[0], dot[1])

def prepare_rtree(dot_list):
  p = rindex.Property()
  p.leaf_capacity = 1000
  p.variant = rindex.RT_Star
  p.fill_factor = 0.02
  
  def points():
    for index in range(len(dot_list)):
      dot = dot_list[index]
      yield (index, dot_to_box(dot), None)

  rtree = rindex.Index(points(), properties=p)
  return rtree

def traverse_rtree(rtree, dot_list, starting_dot=[0, 0]):
  sorted_dots = deque()
  num_dots = len(dot_list)

  previous_dot = starting_dot
  for i in range(num_dots):
    if i % 500 == 0:
      print('{:} of {:}'.format(i, num_dots))

    nearest_id = next(rtree.nearest(dot_to_box(previous_dot)))
    nearest_dot = dot_list[nearest_id]
    previous_dot = nearest_dot

    sorted_dots.append(nearest_dot)

    rtree.delete(nearest_id, dot_to_box(nearest_dot))

  return sorted_dots

def scale_dot(dot, scale):
  return (int(dot[1] * scale), int(dot[0] * scale))

def dist(dot1, dot2):
  return abs(dot1[1]-dot2[1])+abs(dot1[0]-dot2[0])

def draw_lines(image, dot_list, scale, color):
  previous_dot = scale_dot(dot_list[0], scale)
  for dot in dot_list:
    scaled_dot = scale_dot(dot, scale)
    if dist(scaled_dot, previous_dot) < 5 / RESOLUTION:
      cv2.line(image, previous_dot, scaled_dot, color, 2)
    previous_dot = scaled_dot

  return image

def dot_to_coord(dot, scale_info):
  return (
      int((dot[1] * scale_info.output_scale + scale_info.width_offset)
          / RESOLUTION),
      int((dot[0] * scale_info.output_scale +
        scale_info.height_offset + PAGE_HEIGHT_OFFSET)
          / RESOLUTION))

def write_hpgl(outfile, dot_list, pen_index, scale_info):
  if not dot_list:
    return

  outfile.write('SP%d' % pen_index)
  first_coord = dot_to_coord(dot_list[0], scale_info)
  outfile.write('PU%d,%dPD' % first_coord)

  for dot in dot_list:
    outfile.write('%d,%d,' % dot_to_coord(dot, scale_info))

def main():
  source_name = sys.argv[1]
  dest = open(sys.argv[2], 'w')

  if not source_name:
    return

  image = cv2.imread(source_name)

  if len(image) > len(image[0]):
    image = np.transpose(image, (1, 0, 2))
  else:
    image = np.fliplr(image)

  image_rows = len(image)
  image_cols = len(image[0])


  scale = 5

  final_image = np.ones((image_rows * scale, image_cols * scale, 3), np.uint8) * 255

  output_scale = min(IMAGE_WIDTH / image_cols, IMAGE_HEIGHT / image_rows)
  width_offset = (PAGE_WIDTH - (image_cols * output_scale)) / 2
  height_offset = (PAGE_HEIGHT - (image_rows * output_scale)) / 2

  scale_info = ScaleInfo(width_offset, height_offset, output_scale)

  all_dots = collect_dots(image, COLORS, DITHER)
  for i, color in zip(range(len(COLORS)), COLORS):
    dot_deque = all_dots[i]
    if not dot_deque:
      continue

    dot_list = list(dot_deque)[::DECIMATE]

    print(len(dot_list))
    rtree = prepare_rtree(dot_list)

    print('made rtree')

    sorted_dots = traverse_rtree(rtree, dot_list, dot_list[0])
    write_hpgl(dest, sorted_dots, i+1, scale_info)

    final_image = draw_lines(final_image, sorted_dots, scale, color)

  dest.write('PUSP0;\n')
  dest.close()

  cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  cv2.imshow('image', final_image)
  while cv2.waitKey(0) != 0x0A:
    pass
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
