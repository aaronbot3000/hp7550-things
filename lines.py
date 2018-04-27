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
kImageMargin = int(3 / plotter_lib.kResolution)

kLineSpacing = int(3 / plotter_lib.kResolution)
kBlur = int(3 / plotter_lib.kResolution)

pattern = 'swoopy'

if pattern == 'diamonds':
  kAngles = [ x * math.pi / 12 for x in range(12) ]
  kLineMap = (
      (kAngles[3], (170, 240)),
      (kAngles[7], (110, 190)),
      (kAngles[11], (100, 140)),
      (kAngles[0], (60, 110)),
      (kAngles[2], (40, 70)),
      (kAngles[4], (0, 50)),
  )

if pattern == 'circles':
  kAngles = [ x * math.pi / 6 for x in range(12) ]
  kLineMap = (
      (kAngles[1], (170, 240)),
      (kAngles[5], (110, 190)),
      (kAngles[2], (100, 140)),
      (kAngles[0], (60, 110)),
      (kAngles[4], (40, 70)),
      (kAngles[3], (0, 50)),
  )

if pattern == 'swoopy':
  kAngles = [ x * math.pi / 20 for x in range(12) ]
  kLineMap = (
      (kAngles[3], (170, 240)),
      (kAngles[4], (110, 190)),
      (kAngles[5], (100, 140)),
      (kAngles[8], (60, 110)),
      (kAngles[9], (40, 70)),
      (kAngles[10], (0, 50)),
  )

kLineWidthVariance = 0 / plotter_lib.kResolution
kMinLineWidth = 4 / plotter_lib.kResolution

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
  image_to_plot = min((x_limit - kImageMargin) / image_dim[0],
                      (y_limit - kImageMargin) / image_dim[1])
  plot_to_image = 1 / image_to_plot
  image_origin = ((np.array((x_limit, y_limit), np.int32) -
                  image_dim * image_to_plot) / 2)

  # Blur the image.
  line_size_in_image = int(kBlur * plot_to_image)
  if line_size_in_image % 2 == 0:
    line_size_in_image += 1
  filter_image = cv2.blur(image, (line_size_in_image, line_size_in_image))

  # Set up start of line pattern.
  image_dim_plot = image_dim * image_to_plot
  line_limits = image_origin + image_dim_plot
  center = image_origin + (image_dim_plot / 2)

  # Figure out overestimate of number of lines to make.
  longest_dimension = math.sqrt(math.pow(image_dim_plot[0], 2) +
                                math.pow(image_dim_plot[1], 2))
  num_lines = int((longest_dimension / 2) / kLineSpacing) + 1

  shapes = deque()
  for angle, color_range in kLineMap:
    line_step = np.array((-1 * math.sin(angle), math.cos(angle)))
    dash_step = np.array((math.cos(angle), math.sin(angle)))

    for _ in range(2):
      for line in range(num_lines):
        line_center = center + (line + 0.5) * kLineSpacing * line_step
        #line_center = center + line * kLineSpacing * line_step

        dash_offset = random.random() * 2 * kLineWidthVariance - kLineWidthVariance
        dash_center = line_center + dash_offset * dash_step

        for i in range(2):
          while True:
            dash_width = random.random() * kLineWidthVariance + kMinLineWidth
            dash_center += (dash_width / 2) * dash_step

            image_position = np.round((dash_center - image_origin) * plot_to_image, 0)
            if not (0 <= image_position[0] < image_dim[0] and
                    0 <= image_position[1] < image_dim[1]):
              break

            value = filter_image[int(image_position[1]), int(image_position[0])]

            upper_thresh = color_range[1]
            lower_thresh = color_range[0]

            line_length_percent = ((upper_thresh - value) /
                                   (upper_thresh - lower_thresh))

            if line_length_percent > 0.95:
              line_length_percent = 1

            line_length = int(dash_width * line_length_percent)

            if line_length > 0.1 / plotter_lib.kResolution:
              point_delta = dash_step * line_length / 2
              start_point = Point(dash_center[0] + point_delta[0],
                                  dash_center[1] + point_delta[1])
              end_point = Point(dash_center[0] - point_delta[0],
                                dash_center[1] - point_delta[1])
              shapes.append(plotter_lib.OpenPolyline((start_point, end_point), kPen))
            dash_center += (dash_width / 2) * dash_step

          dash_center = line_center + dash_offset * dash_step
          dash_step *= -1
      line_step *= -1




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
