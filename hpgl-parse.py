#!/usr/bin/python3

from plotter_lib import Line
from plotter_lib import sort_all
from plotter_lib import write_shapes

MERGE_DIST = 0.3  # mm


class PenParser:
  def __init__(self):
    self._current_pen = 0
    self._last_position = Point(0, 0)
    self._pen_down = False
    self._parameter_cleaner = str.maketrans(',+', '  ')
    self._all_shapes = []

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

  def _parse(self, command, parameters):
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
        new_line = Line(self._last_position, self._current_pen)
        self._all_shapes.append(new_line)

      self._all_shapes[-1].add_points(points)
      self._last_position = points[-1]
      return
    else:
      print 'Unrecognized command: %s' % command

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

    return self._all_shapes


def main():
  parser = PenParser()

  with open(sys.argv[1], 'r') as source:
    all_shapes = parser.read_file(source)

  sorted_shapes = sort_all(all_shapes(), MERGE_DIST)

  with open(sys.argv[2], 'w') as dest:
    dest.write('IN')

    for _, shapes in sorted_shapes.items():
      write_shapes(shapes, dest)

    dest.write('PUSP0;\n')

if __name__ == "__main__":
  main()
