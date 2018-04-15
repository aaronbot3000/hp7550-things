from collections import deque
import math
import random
import sys

from plotter_lib import Label
from plotter_lib import Point
from plotter_lib import SortAllAndWrite
from plotter_lib import Square

# 10 inch page width (cols)
# 15.684 inch page height (rows)

def main():
  shapes = deque()

  side = 625

  rows = 22
  cols = 13

  x_offset = 1092 + side / 2
  y_offset = 1017.5 + side / 2

  for r in range(rows):
    for c in range(cols):
      scale_factor = 9.88939

      phi_range = math.radians(r * 1.5 + 1)
      trans_range = r * 1.2 * scale_factor
      red_thresh = math.log(r / 22 + 0.95) * 0.4

      phi = random.uniform(-1 * phi_range, phi_range)
      rand_x = random.uniform(-1 * trans_range, trans_range)
      rand_y = random.uniform(-1 * trans_range, trans_range)
      if random.random() < red_thresh:
        pen = 2
      else:
        pen = 1

      x_pos = x_offset + side * r + rand_x
      y_pos = y_offset + side * c + rand_y
      shapes.append(Square(side, Point(x_pos, y_pos), phi, pen))

  shapes.append(
      Label('SYMBOLIC DISARRAY', Point(15900, 1016), (1, 0.5), math.pi / 2, 1))

  with open(sys.argv[1], 'w') as dest:
    SortAllAndWrite(dest, shapes, 0.3)

if __name__ == "__main__":
  main()
