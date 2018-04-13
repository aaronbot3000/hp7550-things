from collections import deque
from collections import namedtuple
from rtree import index as rindex
import itertools
import sys
import math

RESOLUTION = 0.025  # mm/quanta

Point = namedtuple('Point', ['x', 'y'])


class Line:
  def __init__(self, start_point, pen=0):
    self._points = deque()
    self._points.append(start_point)
		self._pen = pen

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

	def pen(self):
		return self._pen

  def __repr__(self):
    return '\n[{:},{:} to {:},{:}, {:} points]'.format(
      self._points[0].x, self._points[0].y,
      self._points[-1].x, self._points[-1].y,
      len(self._points))


def _prepare_rtree(line_list):
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


def _sort(line_deque, merge_dist):
	if not line_deque:
		return None

	merge_dist_units = merge_dist / RESOLUTION

	# bootstrap with first line.
	sorted_lines = deque()
	sorted_lines.append(line_deque[0])

	# copy to array for fast random access
	line_array = list(line_deque)
	rtree = _prepare_rtree(line_array[1:])

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


def sort_all(mixed_shapes, merge_dist=0):
	# Sort shapes by pen.
	categorized_shapes = {}
	sorted_shapes = {}
	for shape in mixed_shapes:
		pen = shape.pen
		if not pen in categorized_shapes:
			categorized_shapes[pen] = deque()
		categorized_shapes[pen].append(shape)
	
	for pen, shape_list in categorized_shapes.items():
		sorted_shapes[pen] = _sort(shape_list, merge_dist)

		print('pen %d' % pen)
		print('starting length: %d' % len(shape_list))
		print('ending length: %d' % len(sorted_shapes[pen]))

	return sorted_shapes


def write_shapes(shapes, outfile):
	if not shapes:
		return

	outfile.write('SP%d' % shapes[0].pen + 1)

	for line in shapes:
		points = line.points()
		first_point = True
		for point in points:
			if first_point:
				outfile.write('PU%d,%d' % (point.x, point.y))
				outfile.write('PD')
				first_point = False
			else:
				outfile.write('%d,%d,' % (point.x, point.y))
