#!/usr/bin/python3
from collections import deque

import lib.plotter as plotter
from lib.plotter import ClosedPolyline
from lib.plotter import OpenPolyline
from lib.plotter import Point

vert = deque()
vert.append(Point(0, 0))
vert.append(Point(0, 1))
vert.append(Point(0, 2))
vert.append(Point(0, 3))
vert.append(Point(0, 4))
vert.append(Point(0, 3))
vert.append(Point(0, 2))
vert.append(Point(0, 1))
line = ClosedPolyline(vert, 0)
print(line)

horiz = deque()
horiz.append(Point(0, 0))
horiz.append(Point(1, 0))
horiz.append(Point(2, 0))
horiz.append(Point(3, 0))
horiz.append(Point(4, 0))
horiz.append(Point(3, 0))
horiz.append(Point(2, 0))
horiz.append(Point(1, 0))
line = ClosedPolyline(horiz, 0)
print(line)

diag = deque()
diag.append(Point(0, 0))
diag.append(Point(100000, 100000))
diag.append(Point(100001, 100002))
diag.append(Point(10000, 0))
diag.append(Point(0, 0))
diag.append(Point(1, 100))
diag.append(Point(2, 200))
diag.append(Point(0, 0))
line = OpenPolyline(diag, 0)
print(line)

horiz = []
horiz.append(Point(0, 0))
horiz.append(Point(1, 0))
horiz.append(Point(2, 0))
line = OpenPolyline(horiz, 0)
print(line)
