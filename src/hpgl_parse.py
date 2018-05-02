#!/usr/bin/python3
from collections import deque
import sys

import lib.plotter
from lib.plotter import ClosedPolyline
from lib.plotter import OpenPolyline
from lib.plotter import Point
from lib.plotter import PointDistance
from lib.plotter import SortAllAndWrite

kMergeDistance = 0.3 / lib.plotter.kResolution  # input in mm


class PenParser:
  def __init__(self):
    self._current_pen = 0
    self._last_position = Point(0, 0)
    self._pen_down = False
    self._parameter_cleaner = str.maketrans(',+', '  ')
    self._all_shapes = []

    self.open_counter = 0
    self.closed_counter = 0

  def _split_point_list(self, argstring):
    cleaned_params = argstring.translate(self._parameter_cleaner)
    split_params = cleaned_params.split()
    return split_params

  def _parse_point_list(self, argstring):
    split_params = self._split_point_list(argstring)
    if not split_params:
      return deque()

    points = deque()
    for i in range(0, len(split_params), 2):
      new_point = Point(int(split_params[i]), int(split_params[i + 1]))
      points.append(new_point)

    return points

  def _add_shape(self):
    if (len(self._current_point_list) > 2 and
        PointDistance(self._current_point_list[0],
                      self._current_point_list[-1]) < kMergeDistance):
      new_shape = ClosedPolyline(self._current_point_list[1:],
          self._current_pen)
      self.closed_counter += 1
    else:
      new_shape = OpenPolyline(self._current_point_list, self._current_pen)
      self.open_counter += 1
    self._all_shapes.append(new_shape)

  def _parse(self, command, parameters):
    # End of the line.
    if command != 'pd' and self._pen_down == True:
      self._add_shape()

    if command == 'lt':
      if not parameters:
        return
      print('Attempted to change line type with parameters %s.' % parameters)
    elif command == 'in':
      return
    elif command == 'sp':
      new_pen = int(parameters) - 1
      if new_pen != self._current_pen:
        self._pen_down = False
        self._current_pen = new_pen
      return
    elif command == 'pu':
      self._pen_down = False
      points = self._parse_point_list(parameters)
      if not points:
        return

      # Don't care about pen up movements except the last destination.
      self._last_position = points[-1]
      return
    elif command == 'pd':
      points = self._parse_point_list(parameters)
    
      if not points and self._pen_down:
        return
        
      if not self._pen_down:
        self._pen_down = True

        self._current_point_list = []
        self._current_point_list.append(self._last_position)

      self._current_point_list.extend(points)
      self._last_position = points[-1]
      return
    else:
      print('Unrecognized command: %s' % command)

  def read_file(self, infile):
    while True:
      line = infile.readline();
      if not line:
        break

      command_start = 0
      while True:
        # Find next command
        while command_start < len(line) and not line[command_start].isalpha():
          command_start += 1

        if command_start >= len(line):
          break
        command = line[command_start : command_start + 2].lower()

        # Read parameters
        param_end = command_start + 2
        while line[param_end].isdigit() or line[param_end] in ', -+.':
          param_end += 1
        params = line[command_start + 2 : param_end]
        command_start = param_end

        self._parse(command, params)

      if self._pen_down:
        self._add_shape()

    return self._all_shapes


def main():
  parser = PenParser()

  with open(sys.argv[1], 'r') as source:
    all_shapes = parser.read_file(source)

  print('Closed polygons: %d' % parser.closed_counter)
  print('Open polygons: %d' % parser.open_counter)

  with open(sys.argv[2], 'wb') as dest:
    if len(sys.argv) == 4 and sys.argv[3] == 'letter':
      SortAllAndWrite(dest, all_shapes, kMergeDistance, 'letter')
    else:
      SortAllAndWrite(dest, all_shapes, kMergeDistance, 'tabloid')

if __name__ == "__main__":
  main()
