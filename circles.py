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

kCircleDiameter = 4 / plotter_lib.kResolution
kPageSize = 'letter'  # letter or tabloid
kPattern = 'random_x'  # grid, staggered_x, staggered_y, random_x, random_y
kRandomRotation = True
kRandomCircles = 6000
kStaggeredPacking = 0.82

kChord = 2 * math.pi / 4
#kCirclePlan = (
#    (240, None),
#    (120, 0.95),
#    (40,  0.70),
#    (20,  0.45),
#    (0,   0.25))

kCirclePlan = (
    (240, None),
    (120, 0.95),
    (40,  0.95),
    (20,  0.95),
    (20,  0.95))

kPenMap = [pens.SARASA['pink'],
           pens.SARASA['blue'],
           pens.SARASA['light blue'],
           pens.SARASA['orange'],
           pens.SARASA['light green'],
           pens.SARASA['red'],
           pens.SARASA['violet'],
           pens.SARASA['mahogany'],
           ]

#kPenMap = [pens.PURE['black'],
#           pens.PURE['red'],
#           pens.PURE['green'],
#           pens.PURE['blue'],
#           pens.PURE['cyan'],
#           pens.PURE['magenta'],
#           pens.PURE['yellow'],
#           ]

kImageMargin = kCircleDiameter / 2
kBlur = kCircleDiameter

random.seed(0)

class Circles(program_lib.Program):
  def __init__(self, page_size):
    super().__init__(page_size)

  def _NaiveScore(self, inv_color, inv_pen):
    scale = None
    for c, p in zip(inv_color, inv_pen):
      # lol division by zero
      if c == 0 and p == 0:
        continue
      elif c == 0 and p != 0:
        scale = 0
      elif c != 0 and p == 0:
        continue
      else:
        this_scale = c / p
        if scale is None or this_scale < scale:
          scale = this_scale

    difference = inv_color - inv_pen * scale
    return (np.sum(difference ** 2), scale)

  def _LeastSquareScore(self, inv_color, inv_pen):
    a = np.sum(np.power(inv_pen, 2))
    b = np.sum(np.multiply(inv_color, inv_pen))
    intercept = b / 2 * a
    if intercept < 0:
      return (float('inf'), intercept)

    return (a * intercept ** 2 - b * intercept + np.sum(inv_color), intercept)

  def _BestColor(self, inv_color):
    best_result = (None, None)
    best_value = None

    for pen_index in range(len(self._pen_map)):
      inv_pen = 255 - np.array(self._pen_map[pen_index])

      this_value, scale = self._NaiveScore(inv_color, inv_pen)

      if best_value is None or this_value < best_value:
        best_result = (pen_index, scale)
        best_value = this_value

    return best_result

  def _ProcessPosition(self, position):
    circles_here = deque()

    image_position = self.GetImagePosition(position)
    if image_position is None:
      return circles_here

    value = self._filter_image[int(image_position[1]), int(image_position[0])]

    if type(value) == np.uint8:  # Grayscale image.
      for i in range(1, len(self._circle_plan)):
        upper_thresh = self._circle_plan[i - 1][0]
        lower_thresh = self._circle_plan[i][0]
        circle_size = self._circle_plan[i][1]
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
    else:  # Color image
      working_value = np.copy(value)
      inv_color = 255 - value
      #print('one circle')
      for i in range(1, len(self._circle_plan)):
        pen, scale = self._BestColor(inv_color)
        actual_scale = min(self._circle_plan[i][1], scale)
        scaled_color = actual_scale * (255 - np.array(self._pen_map[pen]))
        inv_color = inv_color - scaled_color

        if scale is None:
          break
        diameter = self._diameter * actual_scale
        if diameter < .2 / plotter_lib.kResolution:
          break;

        start_angle = 0
        if self._random_rotation:
          start_angle = random.random() * math.pi * 2
        circles_here.append(
            plotter_lib.Arc(Point(position[0], position[1]),
              diameter,
              start_angle,
              2 * math.pi,
              self._chord,
              pen))

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
    self._circle_plan = plan
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
          self._ProcessPosition(center))
      center = self._GetNextCenter(center, start)
      if center is None:
        break
    return shapes

def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  circle_maker = Circles(kPageSize)
  circles = circle_maker.FillImageWithCircles(image, kPattern, kCircleDiameter,
      kChord, kCirclePlan, kRandomRotation, kPenMap)

  print('Contains %d "circles".' % len(circles))
  if not plotter_lib.ShowPreview(circles, kPageSize, kPenMap):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, circles, 0.3, kPageSize)

if __name__ == "__main__":
  main()
