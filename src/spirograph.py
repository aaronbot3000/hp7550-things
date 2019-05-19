#!/usr/bin/python3
import math
import numpy as np
import sys
from collections import deque

import lib.plotter as plotter
import lib.pens as pens

kNumPoints = 97
kStep = 35

kRadius = 75 / plotter.kResolution

def draw_dottyline(radius, num_points, step, pen):
  center = np.array((plotter.kLetterX, plotter.kLetterY)) / 2

  endpoints = []

  for i in range(num_points):
    angle = i / num_points * 2 * math.pi
    #if i % 10 < 3:
      #continue
    #if i % 40 < 20:
      #continue
    endpoints.append(plotter.Point(radius * math.cos(angle) + center[0],
                                   radius * math.sin(angle) + center[1]))

  points = deque()
  cursor = 0
  while True:
    points.append(endpoints[cursor])
    cursor = (cursor + step) % len(endpoints)
    if cursor == 0:
      break

  return plotter.ClosedPolyline(points, pen)

def main():
  shapes = deque()
  shapes.append(draw_dottyline(75 / plotter.kResolution, 240, 37, 0))
  shapes.append(draw_dottyline(60 / plotter.kResolution, 97, 30, 0))
  shapes.append(draw_dottyline(25 / plotter.kResolution, 97, 25, 0))
  shapes.append(draw_dottyline(12 / plotter.kResolution, 21, 25, 0))

  pen_map = [
      pens.SARASA['black'],
      pens.SARASA['light green'],
      pens.SARASA['light blue'],
      pens.SARASA['mahogany'],
      pens.SARASA['pink'],
  ]

  if not plotter.ShowPreview(shapes, 'letter', pen_map):
    return 0

  with open(sys.argv[1], 'wb') as dest:
    plotter.SortAllAndWrite(dest, shapes, 0.7, 'letter', reorder=False)

if __name__ == "__main__":
  main()
