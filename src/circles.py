#!/usr/bin/python3
from collections import deque
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cmc
import cv2
import math
import numpy as np
import random
import sys

import lib.pens as pens
import lib.plotter as plotter
import lib.program as program

kCircleDiameter = 3.0 / plotter.kResolution
kPageSize = 'letter'  # letter or tabloid
kPattern = 'staggered_x'  # grid, staggered_x, staggered_y, random_x, random_y
kRandomRotation = True
kRandomCircles = 6000
kStaggeredPacking = 0.82

kChord = 2 * math.pi / 4
kGrayscaleCirclePlan = (
    None,
    0.95,
    0.70,
    0.45,
    0.25)

kColorCirclePlan = (1, 1, 1)

kTestColors = [
    pens.PURE['red'],
    pens.PURE['green'],
    pens.PURE['blue'],
    pens.PURE['cyan'],
    pens.PURE['magenta'],
    pens.PURE['yellow'],
    pens.PURE['black'],
]

kBluesGreens = [
    pens.SARASA['light blue'],
    pens.SARASA['blue'],
    pens.SARASA['cobalt'],
    pens.SARASA['forest green'],
    pens.SARASA['kelly green'],
    pens.PIGMA_MICRON['yellow'],
    pens.SARASA['mahogany'],
    pens.SARASA['black'],
    ]

kBluesBrowns = [
    pens.SARASA['light blue'],
    pens.SARASA['blue'],
    pens.SARASA['orange'],
    pens.SARASA['light green'],
    pens.SARASA['violet'],
    pens.SARASA['red'],
    pens.SARASA['mahogany'],
    pens.SARASA['black'],
    ]

kPenMap = kBluesBrowns

kImageMargin = kCircleDiameter
kBlur = kCircleDiameter

random.seed(0)

class Circles(program.Program):
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

  def _WeightedNaiveScore(self, inv_color, inv_pen):
    weights = np.array((0.3, 0.59, 0.11), dtype=np.float)
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

    difference = np.multiply((inv_color - inv_pen), weights)
    return (np.sum(difference ** 2), scale)

  def _BestColor(self, inv_color):
    best_result = (None, None)
    best_value = None

    for pen_index in range(len(self._pen_map)):
      inv_pen = 255 - np.array(self._pen_map[pen_index])

      #this_value, scale = self._NaiveScore(inv_color, inv_pen)
      this_value, scale = self._WeightedNaiveScore(inv_color, inv_pen)

      if best_value is None or this_value < best_value:
        best_result = (pen_index, scale)
        best_value = this_value

    return best_result

  def _ProcessInRgb(self, position, value):
    circles_here = deque()
    inv_color = 255 - value
    for plan_scale in self._circle_plan:
      pen, scale = self._BestColor(inv_color)
      actual_scale = min(plan_scale, scale)
      scaled_color = actual_scale * (255 - np.array(self._pen_map[pen]))
      inv_color = inv_color - scaled_color

      if scale is None:
        break
      diameter = self._diameter * actual_scale
      if diameter < .2 / plotter.kResolution:
        break

      start_angle = 0
      if self._random_rotation:
        start_angle = random.random() * math.pi * 2
      circles_here.append(
          plotter.Arc(plotter.Point(position[0], position[1]),
            diameter,
            start_angle,
            2 * math.pi,
            self._chord,
            pen))
    return circles_here

  def _CIELabScore(self, value):
    best_result = (None, None)
    max_diff = 30
    best_value = max_diff

    for pen_index in range(len(self._pen_map)):
      color_scaled = np.array([[value]], dtype=np.float32) * (1 / 255.0)
      color_lab = cv2.cvtColor(color_scaled, cv2.COLOR_RGB2Lab)[0][0]
      color_labcolor = LabColor(color_lab[0], color_lab[1], color_lab[2])

      difference = delta_e_cmc(self._pen_labcolor[pen_index], color_labcolor)

      if best_value is None or difference < best_value:
        best_value = difference
        score = 0
        score += max((max_diff - difference) / max_diff, 0) / 2
        score += min(color_labcolor.lab_l / self._pen_labcolor[pen_index].lab_l, 1) / 2
        best_result = (pen_index, score)
  
    if best_result[0] == len(self._pen_map) - 1:
      return (None, None)
    return best_result

  def _ProcessInCIELab(self, position, value):
    circles_here = deque()

    working_value = np.zeros(3, np.int32)
    np.copyto(working_value, value)

    for i in range(len(self._circle_plan)):
      result = self._CIELabScore(working_value)
      if result[0] is None:
        return circles_here

      circle_size = self._circle_plan[i] * result[1]

      inv_color = (255 - np.array(self._pen_map[result[0]]))
      working_value = np.minimum(working_value + inv_color, 255)
      
      diameter = (self._diameter * circle_size)
      if diameter <= 0:
        break

      start_angle = 0
      if self._random_rotation:
        start_angle = random.random() * math.pi * 2
      circles_here.append(
          plotter.Arc(plotter.Point(position[0], position[1]),
            diameter,
            start_angle,
            2 * math.pi,
            self._chord,
            result[0]))

      start_angle = 0
      if self._random_rotation:
        start_angle = random.random() * math.pi * 2
      circles_here.append(
          plotter.Arc(plotter.Point(position[0], position[1]),
            diameter / 2,
            start_angle,
            2 * math.pi,
            self._chord,
            result[0]))

    return circles_here

  def _ProcessPosition(self, position):

    image_position = self.GetImagePosition(position)
    if image_position is None:
      return deque()

    value = self._filter_image[int(image_position[1]), int(image_position[0])]

    if self._conv_type == 'grayscale':  # Grayscale image.
      circles_here = deque()
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
            plotter.Arc(plotter.Point(position[0], position[1]),
              diameter,
              start_angle,
              2 * math.pi,
              self._chord,
              0))
      return circles_here
    elif self._conv_type == 'color':
      return self._ProcessInRgb(position, value)
    elif self._conv_type == 'cielab':
      self._pen_labcolor = []
      for pen in self._pen_map:
        pen_scaled = np.array([[pen]], dtype=np.float32) * (1 / 255.0)
        pen_lab = cv2.cvtColor(pen_scaled, cv2.COLOR_RGB2Lab)[0][0]
        self._pen_labcolor.append(LabColor(pen_lab[0], pen_lab[1], pen_lab[2]))

      return self._ProcessInCIELab(position, value)

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

  def FillImageWithCircles(self, conv_type, image, pattern, diameter, chord,
      plan, random_rotation, pen_map):
    self._filter_image = self._InitializeImage(image, kImageMargin, kBlur)

    self._pattern = pattern
    self._diameter = diameter
    self._chord = chord
    self._circle_plan = plan
    self._random_rotation = random_rotation
    self._pen_map = pen_map
    self._conv_type = conv_type

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
  # Read conversion type: grayscale or color.
  conv_type = sys.argv[1]
  # Read input image.
  source_name = sys.argv[2]

  if conv_type == 'grayscale':
    image = cv2.imread(source_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    no_bg_white = image[image < 240]
    quantiles = [ i / 4 for i in range(5) ]
    thresholds = np.flip(np.unique(np.quantile(no_bg_white, quantiles)))
    print('Image thresholds: %s' % str(thresholds))

    plan = [ z for z in zip(thresholds, kGrayscaleCirclePlan) ]
    print(plan)
    pen_map = [pens.SARASA['black']]
  elif conv_type == 'color':
    image = cv2.imread(source_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plan = kColorCirclePlan
    pen_map = kPenMap
  elif conv_type == 'cielab':
    image = cv2.imread(source_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plan = kColorCirclePlan
    pen_map = kPenMap + [pens.PURE['white']]
  else:
    print('Unknown conversion type %s' % conv_type)
    return -1

  circle_maker = Circles(kPageSize)
  circles = circle_maker.FillImageWithCircles(conv_type, image, kPattern,
      kCircleDiameter, kChord, plan, kRandomRotation, pen_map)

  print('Contains %d "circles".' % len(circles))
  if not plotter.ShowPreview(circles, kPageSize, pen_map):
    return 0

  with open(sys.argv[3], 'wb') as dest:
    plotter.SortAllAndWrite(dest, circles, 0.3, kPageSize)

if __name__ == "__main__":
  main()
