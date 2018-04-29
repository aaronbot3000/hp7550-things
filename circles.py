#!/usr/bin/python3

from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import pens
import plotter_lib
import program_lib
from plotter_lib import Point

kCircleDiameter = 6 / plotter_lib.kResolution
kPageSize = 'tabloid'  # letter or tabloid
kPattern = 'staggered_x'  # grid, staggered_x, staggered_y, random_x, random_y
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

kPenMap = [pens.SARASA['black']]

kImageMargin = kCircleDiameter / 2
kBlur = kCircleDiameter

random.seed(0)

class Circles(program_lib.Program):
  def __init__(self, page_size):
    super().__init__(page_size)

  def _ProcessPosition(self, position, circle_plan):
    circles_here = deque()

    image_position = self.GetImagePosition(position)
    if image_position is None:
      return circles_here

    value = self._filter_image[int(image_position[1]), int(image_position[0])]

    if type(value) == np.uint8:
      for i in range(1, len(circle_plan)):
        upper_thresh = circle_plan[i - 1][0]
        lower_thresh = circle_plan[i][0]
        circle_size = circle_plan[i][1]
        diameter = (self._diameter *
            min(circle_size * (upper_thresh - value) /
              (upper_thresh - lower_thresh), circle_size))
        if diameter <= 0:
          break
        start_angle = 0
        if self._random_rotation:
          start_angle = random.random() * math.pi * 2
        circles_here.append(
            plotter_lib.Arc(Point(position[0], position[1]),
              diameter,
              start_angle,
              2 * math.pi,
              self._chord,
              0))
    return circles_here

  def _GetNextCenter(self, center, start):
    if self._pattern == 'grid':
      center[0] += self._diameter
      if center[0] > self._circle_limits[0]:
        center[0] = start[0]
        center[1] += self._diameter
      if center[1] > self._circle_limits[1]:
        return None
    elif self._pattern == 'staggered_y':
      center[0] += self._diameter
      if center[0] > self._circle_limits[0]:
        self._y_count += 1
        center[0] = start[0]
        if self._y_count % 2 == 1:
          center[0] += self._diameter / 2
        center[1] += self._diameter * kStaggeredPacking
      if center[1] > self._circle_limits[1]:
        return None
    elif self._pattern == 'staggered_x':
      center[1] += self._diameter
      if center[1] > self._circle_limits[1]:
        self._x_count += 1
        center[1] = start[1]
        if self._x_count % 2 == 1:
          center[1] += self._diameter / 2
        center[0] += self._diameter * kStaggeredPacking
      if center[0] > self._circle_limits[0]:
        return None
    elif self._pattern == 'random_x':
      center[1] += self._diameter
      if center[1] > self._circle_limits[1]:
        center[1] = start[1] + random.random() * self._diameter
        center[0] += self._diameter
      if center[0] > self._circle_limits[0]:
        return None
    elif self._pattern == 'random_y':
      center[0] += self._diameter
      if center[0] > self._circle_limits[0]:
        center[0] = start[0] + random.random() * self._diameter
        center[1] += self._diameter
      if center[1] > self._circle_limits[1]:
        return None
    else:
      assert False, 'Unknown pattern.'
    return center

  def FillImageWithCircles(self, image, pattern, diameter, chord, plan,
      random_rotation, pen_map):
    self._filter_image = self._InitializeImage(image, kImageMargin, kBlur)

    self._pattern = pattern
    self._diameter = diameter
    self._chord = chord
    self._random_rotation = random_rotation
    self._pen_map = pen_map

    # Set up start of circle pattern.
    # Find out how many circles we can put in the picture
    start = np.copy(self._image_origin)

    # Staggered allows for higher packing.
    # Consider staggered for the x axis.
    if pattern == 'staggered_x':
      short_dim = diameter * kStaggeredPacking
      rounded_dim_x = math.floor(self._image_dim_plot[0] / short_dim) * short_dim
    else:
      rounded_dim_x = (math.floor(self._image_dim_plot[0] / diameter) * diameter)
    
    # Consider staggering for the y axis.
    if pattern == 'staggered_y':
      short_dim = diameter * kStaggeredPacking
      rounded_dim_y = math.floor(self._image_dim_plot[1] / short_dim) * short_dim
    else:
      rounded_dim_y = (math.floor(self._image_dim_plot[1] / diameter) * diameter)

    # Centering the circles on the paper.
    remainder = (self._image_dim_plot -
        np.array((rounded_dim_x, rounded_dim_y), np.int32))
    start += remainder / 2
    center = np.copy(start)

    self._circle_limits = self._image_origin + self._image_dim_plot

    shapes = deque()
    self._x_count = 0
    self._y_count = 0
    while True:
      shapes.extend(
          self._ProcessPosition(center, plan))
      center = self._GetNextCenter(center, start)
      if center is None:
        break
    return shapes

def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  circle_maker = Circles(kPageSize)
  circles = circle_maker.FillImageWithCircles(image, kPattern, kCircleDiameter,
      kChord, kCirclePlan, kRandomRotation, kPenMap)

  print('Contains %d "circles".' % len(circles))
  if not plotter_lib.ShowPreview(circles, kPageSize, kPenMap):
    return 0
  
  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.3, kPageSize)

if __name__ == "__main__":
  main()
