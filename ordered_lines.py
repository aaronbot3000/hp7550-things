#!/usr/bin/python3

from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import plotter_lib
from plotter_lib import Point

kPageSize = 'letter'  # letter or tabloid

kLineSpacing = int(3 / plotter_lib.kResolution)
kLineResolution = int(3 / plotter_lib.kResolution)

kPen = 0

def main():
  # Read input image.
  source_name = sys.argv[1]
  color_image = cv2.imread(source_name)
  image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
  image = np.rot90(image, 2)
  image = np.flip(image, 1)

  image_dim = np.array((len(image[0]), len(image)), np.int32)

  # Set up paper bounds.
  if kPageSize == 'tabloid':
    x_limit = plotter_lib.kTabloidX
    y_limit = plotter_lib.kTabloidY
  else:
    x_limit = plotter_lib.kLetterX
    y_limit = plotter_lib.kLetterY

  # Center and get image scaling.
  image_to_plot = min((x_limit - kLineResolution) / image_dim[0],
                      (y_limit - kLineResolution) / image_dim[1])
  plot_to_image = 1 / image_to_plot
  image_origin = ((np.array((x_limit, y_limit), np.int32) - 
                  image_dim * image_to_plot) / 2)

  # Blur the image.
  line_size_in_image = int(min(kLineSpacing, kLineResolution) * plot_to_image)
  if line_size_in_image % 2 == 0:
    line_size_in_image += 1
  filter_image = cv2.blur(image, (line_size_in_image, line_size_in_image))

  # Set up start of line pattern.
  start = np.copy(image_origin)
  image_dim_plot = image_dim * image_to_plot

  # Horizontal lines
  rounded_dim_x = (math.floor(image_dim_plot[0] / kLineSpacing) *
      kLineSpacing)
  rounded_dim_y = (math.floor(image_dim_plot[1] / kLineResolution) *
      kLineResolution)

  # Centering the lines on the paper.
  remainder = (image_dim_plot -
      np.array((rounded_dim_x, rounded_dim_y), np.int32))
  start += remainder / 2

  line_limits = image_origin + image_dim_plot

  shapes = deque()
  for y in range(int(start[1]), int(line_limits[1]), kLineSpacing):
    for x in range(int(start[0]), int(line_limits[0]), kLineResolution):
      center = np.array((x, y), np.int32)
      image_position = np.round((center - image_origin) * plot_to_image, 0)
      if not (0 <= image_position[0] < image_dim[0] and
              0 <= image_position[1] < image_dim[1]):
        continue

      thresh1 = 220.0
      #thresh2 = 120.0
      #thresh3 = 60
      thresh2 = 100.0
      thresh3 = 0
      value = filter_image[int(image_position[1]), int(image_position[0])]
      line_length = int(kLineResolution * min(((thresh1 - value) / (thresh1 - thresh2)), 1))
      if line_length <= 0:
        continue

      start_point = Point(center[0] - line_length / 2, center[1] + line_length / 2)
      end_point = Point(center[0] + line_length / 2, center[1] - line_length / 2)
      shapes.append(plotter_lib.OpenPolyline((start_point, end_point), kPen))

      if value >= thresh2:
        continue

      line_length = int(kLineResolution * min(((thresh2 - value) / (thresh2 - thresh3)), 1))
      if line_length <= 0:
        continue

      start_point = Point(center[0] + line_length / 2, center[1] + line_length / 2)
      end_point = Point(center[0] - line_length / 2, center[1] - line_length / 2)
      shapes.append(plotter_lib.OpenPolyline((start_point, end_point), kPen))

#      if value >= thresh3:
#        continue
#
#      line_length = int(kLineResolution * min(((thresh3 - value) / thresh3), 1))
#      if line_length <= 0:
#        continue
#
#      start_point = Point(center[0] - line_length / 2, center[1])
#      end_point = Point(center[0] + line_length / 2, center[1])
#      shapes.append(plotter_lib.OpenPolyline((start_point, end_point), kPen))


  cv2.namedWindow('filter', cv2.WINDOW_AUTOSIZE)
  filter_image_display = np.flip(filter_image, 0)
  cv2.imshow('filter', filter_image_display)

  print('Line segments: %d' % len(shapes))

  pen_map = [(0, 0, 0)]
  if not plotter_lib.ShowPreview(shapes, kPageSize, pen_map):
    return 0
  
  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.7, kPageSize)

if __name__ == "__main__":
  main()
