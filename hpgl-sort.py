#!/usr/bin/python3

from collections import deque
from collections import namedtuple
from rtree import index as rindex
import itertools
import sys
import math

NUM_PENS = 16
RESOLUTION = 0.025  # mm/quanta
MERGE_DIST = 0.3  # mm

Point = namedtuple('Point', ['x', 'y'])


class Line:
  def __init__(self, start_point):
    self._points = deque()
    self._points.append(start_point)
    self._reversed = False

  def points(self):
    return self._points

  def first_as_box(self):
    point = self._points[0]
    return [point.x, point.y, point.x, point.y]

  def last_as_box(self):
    point = self._points[-1]
    return [point.x, point.y, point.x, point.y]

  def reverse(self):
    self._points.reverse()

  def merge(self, other_line):
    self._points.extend(other_line.points())

  def add_points(self, list_of_points):
    self._points.extend(list_of_points)

  def distance(self, other_line):
    other_point = other_line.points()[0]
    return math.sqrt(math.pow(self._points[-1].x - other_point.x, 2) + 
                     math.pow(self._points[-1].y - other_point.y, 2))

  def __repr__(self):
    return '\n[{:},{:} to {:},{:}, {:} points]'.format(
      self._points[0].x, self._points[0].y,
      self._points[-1].x, self._points[-1].y,
      len(self._points))

class PenParser:
  def __init__(self):
    self._current_pen = 0
    self._last_position = Point(0, 0)
    self._pen_down = False
    self._parameter_cleaner = str.maketrans(',+', '  ')

    self._line_lists = []
    for i in range(NUM_PENS):
      self._line_lists.append(deque())

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

  def parse(self, command, parameters):
    if command == 'sp':
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
        new_line = Line(self._last_position)
        self._line_lists[self._current_pen].append(new_line)

      self._line_lists[self._current_pen][-1].add_points(points)
      self._last_position = points[-1]
      return

  def _prepare_rtree(self, line_list):
    p = rindex.Property()
    p.leaf_capacity = 1000
    p.variant = rindex.RT_Star
    p.fill_factor = 0.02
    
    def points():
      for index in range(len(line_list)):
        offset_index = index + 1
        line = line_list[index]
        yield (offset_index, line.first_as_box(), None)
        yield (offset_index * -1, line.last_as_box(), None)

    rtree = rindex.Index(points(), properties=p)
    return rtree

  def _sort(self, line_deque):
    if not line_deque:
      return None

    merge_dist_units = MERGE_DIST / RESOLUTION

    # bootstrap
    sorted_lines = deque()
    sorted_lines.append(line_deque[0])

    # copy to array for fast random access
    line_array = list(line_deque)
    rtree = self._prepare_rtree(line_array[1:])

    num_lines = len(line_array) - 1
    for i in range(num_lines):
      if i % 100 == 0:
        print('{:} of {:}'.format(i, num_lines))

      nearest_id = next(rtree.nearest(sorted_lines[-1].last_as_box()))

      array_index = nearest_id
      if array_index < 0:
        array_index *= -1
      nearest_line = line_array[array_index]

      if nearest_id < 0:
        rtree.delete(nearest_id, nearest_line.last_as_box())
        rtree.delete(-1 * nearest_id, nearest_line.first_as_box())
      else:
        rtree.delete(nearest_id, nearest_line.first_as_box())
        rtree.delete(-1 * nearest_id, nearest_line.last_as_box())

      if nearest_id < 0:
        nearest_line.reverse()

      if sorted_lines[-1].distance(nearest_line) < merge_dist_units:
        sorted_lines[-1].merge(nearest_line)
      else:
        sorted_lines.append(nearest_line)
    return sorted_lines

  def sort_all_and_save(self, outfile):
    pen_number = 0
    for line_list in self._line_lists:
      pen_number += 1
      if not line_list:
        continue

      starting_length = len(line_list)
      sorted_lines = self._sort(line_list)

      print('pen {:}'.format(pen_number))
      print('starting length: {:}'.format(starting_length))
      print('ending length: {:}'.format(len(sorted_lines)))
      outfile.write('SP%d' % pen_number)
      for line in sorted_lines:
        points = line.points()
        first_point = True
        for point in points:
          if first_point:
            outfile.write('PU%d,%d' % (point.x, point.y))
            outfile.write('PD')
            first_point = False
          else:
            outfile.write('%d,%d,' % (point.x, point.y))


def main():
  source = open(sys.argv[1], 'r')
  dest = open(sys.argv[2], 'w')

  parser = PenParser()
  while True:
    line = source.readline();
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

      parser.parse(command, params)

  source.close()
  dest.write('IN')
  parser.sort_all_and_save(dest)
  dest.write('PUSP0;\n')
  dest.close()

if __name__ == "__main__":
  main()
