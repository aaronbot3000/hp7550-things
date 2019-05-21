#!/usr/bin/python3
from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import lib.plotter as plotter
import lib.program as program
import lib.pens as pens

kPageSize = 'letter'  # letter or tabloid

kMaxCircleDiameter = 4 / plotter.kResolution

kImageMargin = kMaxCircleDiameter
kChord = 2 * math.pi / 4;

random.seed(1)
np.random.seed(1)

class Hatch(program.Program):
  def __init__(self, paper_type):
    super().__init__(paper_type)

  def _PlaceHatches(self, threshold):
    shapes = deque()

    threshold_image = np.less(self._filter_image, threshold)
    possible_locations = np.nonzero(threshold_image)
    num_locations = len(possible_locations[0])

    for i in range(min(int(num_locations / 2), 5000)):
      i = random.randint(0, num_locations - 1)

      diameter = max(kMaxCircleDiameter * random.random(), 1);
      position = self.Image2Plotter(
          possible_locations[1][i], possible_locations[0][i])
      shapes.append(plotter.Arc(
        position,
        diameter,
        0, #random.random() * 2 * math.pi,
        2 * math.pi,
        kChord,
        self._pen))
    return shapes

  def Run(self, image, image_margin):
    self._filter_image = self._InitializeImage(image, image_margin, blur=None)
    self._pen = 0

    no_bg_white = self._filter_image[self._filter_image < 240]
    # Take the middle 90%
    levels = np.flip(np.unique(np.percentile(no_bg_white, range(5, 99, 8))))
    print(levels)
    all_shapes = []
    for level in levels:
      all_shapes.extend(self._PlaceHatches(level))

    return all_shapes

def main():
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  pen_map = [pens.SARASA['black']]

  hatch = Hatch(kPageSize)
  shapes = hatch.Run(image, kImageMargin)

  print('Contains %d "shapes".' % len(shapes))
  if not plotter.ShowPreview(shapes, kPageSize, pen_map):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter.SortAllAndWrite(dest, shapes, 0.3, kPageSize)

if __name__ == "__main__":
  main()
