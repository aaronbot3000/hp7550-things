#!/usr/bin/python3

from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import plotter_lib
from plotter_lib import Point

kCircleDiameter = int(5 / plotter_lib.kResolution)
kPageSize = 'tabloid'  # letter or tabloid
kPattern = 'staggered_x'  # grid, staggered_x, staggered_y, random
kRandomRotation = True
kRandomCircles = 6000
kStaggeredPacking = 0.82

kChord = 2 * math.pi / 4
kCirclePlan = (
    (240, None),
    (120, 0.95),
    (40,  0.70),
    (20,  0.45),
    (0,   0.25))

kPen = 1

random.seed(0)

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
  image_to_plot = min((x_limit - kCircleDiameter) / image_dim[0],
                      (y_limit - kCircleDiameter) / image_dim[1])
  plot_to_image = 1 / image_to_plot
  image_origin = ((np.array((x_limit, y_limit), np.int32) - 
                  image_dim * image_to_plot) / 2)

  # Blur the image.
  circle_size_in_image = int(kCircleDiameter * plot_to_image)
  if circle_size_in_image % 2 == 0:
    circle_size_in_image += 1
  filter_image = cv2.blur(image, (circle_size_in_image, circle_size_in_image))


  # Set up start of circle pattern.
  # Find out how many circles we can put in the picture
  start = np.copy(image_origin)
  image_dim_plot = image_dim * image_to_plot

  # Staggered allows for higher packing.
  # Consider staggered for the x axis.
  if kPattern == 'staggered_x':
    short_dim = kCircleDiameter * kStaggeredPacking
    rounded_dim_x = math.floor(image_dim_plot[0] / short_dim) * short_dim
  else:
    rounded_dim_x = (math.floor(image_dim_plot[0] / kCircleDiameter) *
        kCircleDiameter)
  
  # Consider staggering for the y axis.
  if kPattern == 'staggered_y':
    short_dim = kCircleDiameter * kStaggeredPacking
    rounded_dim_y = math.floor(image_dim_plot[1] / short_dim) * short_dim
  else:
    rounded_dim_y = (math.floor(image_dim_plot[1] / kCircleDiameter) *
        kCircleDiameter)

  # Centering the circles on the paper.
  remainder = (image_dim_plot -
      np.array((rounded_dim_x, rounded_dim_y), np.int32))
  start += remainder / 2
  center = np.copy(start)

  circle_limits = image_origin + image_dim_plot

  shapes = deque()
  x_count = 0
  y_count = 0
  circle_count = 0
  while True:
    image_position = np.round((center - image_origin) * plot_to_image, 0)
    if (0 <= image_position[0] < image_dim[0] and
        0 <= image_position[1] < image_dim[1]):
      value = filter_image[int(image_position[1]), int(image_position[0])]

      for i in range(1, len(kCirclePlan)):
        upper_thresh = kCirclePlan[i - 1][0]
        lower_thresh = kCirclePlan[i][0]
        circle_size = kCirclePlan[i][1]
        diameter = int(kCircleDiameter *
            min(circle_size * (upper_thresh - value) /
              (upper_thresh - lower_thresh), circle_size))
        if diameter <= 0:
          break
        start_angle = 0
        if kRandomRotation:
          start_angle = random.random() * math.pi * 2
        shapes.append(
            plotter_lib.Arc(Point(center[0], center[1]),
              diameter,
              start_angle,
              math.pi * 2,
              kChord,
              kPen))

    if kPattern == 'grid':
      center[0] += kCircleDiameter
      if center[0] > circle_limits[0]:
        center[0] = start[0]
        center[1] += kCircleDiameter
      if center[1] > circle_limits[1]:
        break
    elif kPattern == 'staggered_y':
      center[0] += kCircleDiameter
      if center[0] > circle_limits[0]:
        y_count += 1
        center[0] = start[0]
        if y_count % 2 == 1:
          center[0] += kCircleDiameter / 2
        center[1] += int(kCircleDiameter * kStaggeredPacking)
      if center[1] > circle_limits[1]:
        break
    elif kPattern == 'staggered_x':
      center[1] += kCircleDiameter
      if center[1] > circle_limits[1]:
        x_count += 1
        center[1] = start[1]
        if x_count % 2 == 1:
          center[1] += kCircleDiameter / 2
        center[0] += int(kCircleDiameter * kStaggeredPacking)
      if center[0] > circle_limits[0]:
        break
    elif kPattern == 'random':
      center[0] = (image_origin[0] +
          random.randint(0, int(image_dim_plot[0] - 1)))
      center[1] = (image_origin[1] + random.randint(0, int(image_dim_plot[1] -
        1)))
      circle_count += 1
      if circle_count > kRandomCircles:
        break

  print('Used %d circles.' % len(shapes))

  cv2.namedWindow('filter', cv2.WINDOW_AUTOSIZE)
  filter_image_display = np.flip(filter_image, 0)
  cv2.imshow('filter', filter_image_display)

  pen_map = {1: (0, 0, 0)}
  if not plotter_lib.ShowPreview(shapes, kPageSize, pen_map):
    return 0
  
  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.3, kPageSize)

if __name__ == "__main__":
  main()
