#!/usr/bin/python3
from collections import deque
import copy
import math
import numpy as np
import random
import sys

import plotter_lib
from plotter_lib import Arc
from plotter_lib import Label
from plotter_lib import Point
from plotter_lib import ShowPreview
from plotter_lib import SortAllAndWrite
from plotter_lib import Square

# 40 points per mm

random.seed(1)
kPen = 0

def main():
  shapes = deque()

  x_limit = plotter_lib.kLetterX
  y_limit = plotter_lib.kLetterY

  # horizontal lines
  line_start = Point(0, 0)
  while True:
    line_end = Point(line_start.x + 200, line_start.y)
    shapes.append(plotter_lib.OpenPolyline((line_start, line_end), kPen))
    line_start.y += 200
    if line_start.y > y_limit:
      break

  line_start = Point(300, 0)
  delta = 40
  while True:
    line_end = Point(line_start.x + 200, line_start.y)
    shapes.append(plotter_lib.OpenPolyline((line_start, line_end), kPen))
    line_start.y += delta
    delta *= 1.1
    if line_start.y > y_limit:
      break

  # vertical lines
  line_start = Point(600, 0)
  direction = True
  while True:
    line_end = Point(line_start.x, line_start.y + 200)
    if direction:
      shapes.append(plotter_lib.OpenPolyline((line_start, line_end), kPen))
    else:
      shapes.append(plotter_lib.OpenPolyline((line_end, line_start), kPen))

    direction = not direction

    line_start.x += 200
    if line_start.x > x_limit:
      break

  line_start = Point(600, 300)
  delta = 40
  while True:
    line_end = Point(line_start.x, line_start.y + 200)
    if direction:
      shapes.append(plotter_lib.OpenPolyline((line_start, line_end), kPen))
    else:
      shapes.append(plotter_lib.OpenPolyline((line_end, line_start), kPen))

    direction = not direction

    line_start.x += delta
    delta *= 1.1
    if line_start.x > x_limit:
      break

  # boxes
  box_start = Point(700, 700)
  gap = 300
  while True:
    if direction:
      box_center = copy.copy(box_start)
      box_center.y += 200
      shapes.append(Square(200, box_center, 0, kPen))
    else:
      shapes.append(Square(200, box_start, 0, kPen))

    direction = not direction

    box_start.x += gap
    gap = max(gap - 5, 100)
    if box_start.x > x_limit:
      break

  # circles
  circle_start = Point(700, 1200)
  gap = 300
  while True:
    if direction:
      circle_center = copy.copy(circle_start)
      circle_center.y += 200
      shapes.append(Arc(circle_center, 200, 0, 2 * math.pi, None, kPen))
    else:
      shapes.append(Arc(circle_start, 200, 0, 2 * math.pi, None, kPen))

    direction = not direction

    circle_start.x += gap
    gap = max(gap - 5, 100)
    if circle_start.x > x_limit:
      break

  # circles that become squares
  circle_start = Point(700, 1700)
  gap = 300
  angle = 0
  while True:
    if direction:
      circle_center = copy.copy(circle_start)
      circle_center.y += 200
      shapes.append(Arc(circle_center, 200, angle, 2 * math.pi,
        2 * math.pi / 3.2, kPen))
    else:
      shapes.append(Arc(circle_start, 200, angle, 2 * math.pi,
        2 * math.pi / 3.2, kPen))

    direction = not direction
    angle += math.pi / 5

    circle_start.x += gap
    gap = max(gap - 5, 100)
    if circle_start.x > x_limit:
      break

  # arcs
  circle_start = Point(700, 2200)
  gap = 300
  angle = 0
  while True:
    if direction:
      circle_center = copy.copy(circle_start)
      circle_center.y += 200
      shapes.append(Arc(circle_center, 200, angle, math.pi / 3.1,
        None, kPen))
    else:
      shapes.append(Arc(circle_start, 200, angle, math.pi / 2.1,
        None, kPen))

    direction = not direction
    angle += math.pi / 4

    circle_start.x += gap
    gap = max(gap - 5, 100)
    if circle_start.x > x_limit:
      break


  pen_map = [(0, 0, 0)]
  if not plotter_lib.ShowPreview(shapes, 'letter', pen_map):
    return 0

  with open(sys.argv[1], 'wb') as dest:
    plotter_lib.SortAllAndWrite(dest, shapes, 0.7, 'letter',
        False)

if __name__ == "__main__":
  main()
