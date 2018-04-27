#!/usr/bin/python3

from collections import deque
import cv2
import hsluv
import math
import numpy as np
import random
import sys

import plotter_lib
from pens import SARASA
from plotter_lib import Point

kCircleDiameter = int(3 / plotter_lib.kResolution)
kPageSize = 'letter'  # letter or tabloid
kPattern = 'staggered_x'  # grid, staggered_x, staggered_y, random_x, random_y
kRandomRotation = True
kRandomCircles = 6000
kStaggeredPacking = 0.82

kChord = 2 * math.pi / 20
kCirclePlan = (
    (240, None),
    (120, 0.95),
    (40,  0.70),
    (20,  0.45),
    (0,   0.25))

kPens = [ color for _, color in SARASA.items() ]

random.seed(0)

kMaxHue = 360
kMaxPixel = 255.0
def bgr2hsl(triplet):
  return hsluv.rgb_to_hsluv((triplet[2] / kMaxPixel,
                             triplet[1] / kMaxPixel,
                             triplet[0] / kMaxPixel))
def rgb2hsl(triplet):
  return hsluv.rgb_to_hsluv((triplet[0] / kMaxPixel,
                             triplet[1] / kMaxPixel,
                             triplet[2] / kMaxPixel))

def hue_distance(a, b):
  return min(abs(a - b), kMaxHue - abs(a - b))

def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
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

  while True:
    image_position = np.round((center - image_origin) * plot_to_image, 0)
    if (0 <= image_position[0] < image_dim[0] and
        0 <= image_position[1] < image_dim[1]):
      hue, saturation, lightness = bgr2hsl(
          filter_image[int(image_position[1]), int(image_position[0])])

      best_pen = None
      if lightness < 90:
        if lightness < 7 or 1508 * math.pow(lightness, -1.43) > saturation:
          best_pen = 3
        else:
          best_distance = float('inf')
          best_pen = 3
          for i in range(len(kPens)):
            p_hue, p_saturation, p_lightness = rgb2hsl(kPens[i])
            distance = hue_distance(hue, p_hue) + 0.4 * abs(saturation - p_saturation) + 0.25 * abs(lightness - p_lightness)
            #distance = hue_distance(hue, p_hue)
            #distance = abs(p_saturation - saturation)
            if distance < best_distance:
              best_distance = distance
              best_pen = i

      if best_pen is not None:
        p_hue, p_saturation, p_lightness = rgb2hsl(kPens[best_pen])

        if best_pen == 3:
          value = lightness * 2.55
        else:
          value = 255 - (saturation / p_saturation * 120)

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
                best_pen))

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
    elif kPattern == 'random_x':
      center[1] += kCircleDiameter
      if center[1] > circle_limits[1]:
        x_count += 1
        center[1] = start[1] + random.random() * kCircleDiameter
        center[0] += kCircleDiameter
      if center[0] > circle_limits[0]:
        break
    elif kPattern == 'random_y':
      center[0] += kCircleDiameter
      if center[0] > circle_limits[0]:
        x_count += 1
        center[0] = start[0] + random.random() * kCircleDiameter
        center[1] += kCircleDiameter
      if center[1] > circle_limits[1]:
        break
  print('Used %d circles.' % len(shapes))

  cv2.namedWindow('filter', cv2.WINDOW_AUTOSIZE)
  filter_image_display = np.flip(filter_image, 0)
  cv2.imshow('filter', filter_image_display)

  if not plotter_lib.ShowPreview(shapes, kPageSize, kPens):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.3, kPageSize)

if __name__ == "__main__":
  main()
