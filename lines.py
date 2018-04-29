#!/usr/bin/python3

from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import plotter_lib
import program_lib
from pens import SARASA
from plotter_lib import Point

kPageSize = 'letter'  # letter or tabloid

kLineSpacing = int(3 / plotter_lib.kResolution)
kLineSpacingVariance = 0 / plotter_lib.kResolution

kLineWidth = 6 / plotter_lib.kResolution
kLineWidthVariance = 0 / plotter_lib.kResolution

kImageMargin = kLineWidth / 2 + kLineWidthVariance

kBlur = int(3 / plotter_lib.kResolution)

pattern = 'swoopy'

if pattern == 'jagged':
  kOffset = 0.5
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
  kOffset = 0.5
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
  kOffset = 0.
  kAngles = [ x * math.pi / 20 for x in range(12) ]
  kLineMap = (
      (kAngles[3], (170, 240)),
      (kAngles[4], (110, 190)),
      (kAngles[5], (100, 140)),
      (kAngles[8], (60, 110)),
      (kAngles[9], (40, 70)),
      (kAngles[10], (0, 50)),
  )

if pattern == 'evenhex':
  kOffset = 0
  kAngles = [ x * math.pi / 5 for x in range(12) ]
  kLineMap = (
      (kAngles[0], (170, 240)),
      (kAngles[1], (110, 190)),
      (kAngles[2], (100, 140)),
      (kAngles[3], (60, 110)),
      (kAngles[4], (40, 70)),
      (kAngles[5], (0, 50)),
  )

kPen = 0


class Hatch(program_lib.Program):
  def __init__(self, paper_type):
    super().__init__(paper_type)

  def _DashInDirection(self, dash_start_ref, dash_step, color_range, pen):
    dash_start = np.copy(dash_start_ref)
    dashes_in_direction = deque()
    while True:
      dash_width = (random.random() * self._line_width_variance +
                    self._line_width)
      dash_center = dash_start + (dash_width / 2) * dash_step

      image_position = self.GetImagePosition(dash_center)
      if image_position is None:
        break

      value = self._filter_image[int(image_position[1]), int(image_position[0])]

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
        dashes_in_direction.append(
            plotter_lib.OpenPolyline((start_point, end_point), pen))

      dash_start += dash_width * dash_step
    return dashes_in_direction

  def _DrawDashLine(self, line_center, dash_step, color_range, pen):
    one_line = deque()

    dash_offset = (2 * random.random() * self._line_width_variance -
                   self._line_width_variance)
    dash_start = line_center + dash_offset * dash_step

    one_line.extend(
        self._DashInDirection(dash_start, dash_step, color_range, pen))
    one_line.extend(
        self._DashInDirection(dash_start, -1 * dash_step, color_range, pen))

    return one_line

  def _GenerateLineCenter(self, center, line, line_step, direction):
    random_line_spacing = (random.random() * self._line_spacing_variance -
                           self._line_spacing_variance / 2.0) * line_step

    if direction == 'positive':
      offset = self._line_offset
      step = line_step
    else:
      offset = 1 - self._line_offset
      step = -1 * line_step

    return (center +
        (line + offset) * self._line_spacing * step +
        random_line_spacing)

  def _DrawAllDashLines(self, center, angle, num_lines, color_range, pen):
    line_step = np.array((-1 * math.sin(angle), math.cos(angle)))
    dash_step = np.array((math.cos(angle), math.sin(angle)))

    parallel_lines = deque()
    for line in range(num_lines):
      positive_center = self._GenerateLineCenter(center, line, line_step,
          'positive')
      parallel_lines.extend(
        self._DrawDashLine(positive_center, dash_step, color_range, pen))

      negative_center = self._GenerateLineCenter(center, line, line_step,
          'negative')
      parallel_lines.extend(
          self._DrawDashLine(negative_center, dash_step, color_range, pen))

    return parallel_lines

  def CrosshatchLines(self, image, pen, image_margin, image_blur, line_spacing,
      line_spacing_variance, line_width, line_width_variance, line_offset,
      line_map):

    self._filter_image = self._InitializeImage(image, image_margin, image_blur)

    self._line_spacing = line_spacing
    self._line_spacing_variance = line_spacing_variance
    self._line_width = line_width
    self._line_width_variance = line_width_variance
    self._line_offset = line_offset

    # Set up start of line pattern.
    center = self._image_origin + (self._image_dim_plot / 2)

    # Figure out overestimate of number of lines to make.
    longest_dimension = math.sqrt(math.pow(self._image_dim_plot[0], 2) +
                                  math.pow(self._image_dim_plot[1], 2))
    num_lines = int((longest_dimension / 2) / self._line_spacing) + 1

    shapes = deque()
    for angle, color_range in line_map:
      shapes.extend(
          self._DrawAllDashLines(center, angle, num_lines, color_range, pen))

    return shapes


def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  hatch = Hatch(kPageSize)
  shapes = hatch.CrosshatchLines(image, 0, kImageMargin, kBlur, kLineSpacing,
      kLineSpacingVariance, kLineWidth, kLineWidthVariance, kOffset,
      kLineMap)

  print('Line segments: %d' % len(shapes))

  pen_map = [SARASA['black'],
      ]
  if not plotter_lib.ShowPreview(shapes, kPageSize, pen_map):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.7, kPageSize)

if __name__ == "__main__":
  main()
