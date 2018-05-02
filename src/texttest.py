#!/usr/bin/python3
from collections import deque
import math
import sys

import lib.plotter as plotter
from lib.plotter import Point
from lib.plotter import ShowPreview
from lib.plotter import SortAllAndWrite
from lib.plotter import WriteShapes

def main():
  shapes = deque()

  side = 625

  x_offset = 1092 + side / 2
  y_offset = 1017.5 + side / 2

  for i in range(22):
    for j in range(13):
      x_pos = x_offset + side * i
      y_pos = y_offset + side * j

      shapes.append(
          plotter.Arc(Point(x_pos, y_pos),
            side,
            (i + j) * 2 * math.pi / (22 + 13),
            2,
            None,
            1))

  pen_map = {1: (0, 0, 0), 2: (0, 0, 255)}

  if not ShowPreview(shapes, 'tabloid', pen_map):
    return 0

#  for i in range(5):
#    for j in range(5):
#      shapes.append(
#          plotter.Label('loLOgy',
#            Point(500 * i, 1000 * j),
#            (i * 0.2 + 0.2, j * 0.2 + 0.2),
#            0,
#            1))

  with open(sys.argv[1], 'w') as dest:
    SortAllAndWrite(dest, shapes, 0.3)
    #WriteShapes(dest, 1, shapes, 0.3)

if __name__ == "__main__":
  main()
