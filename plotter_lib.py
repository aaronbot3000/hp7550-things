from collections import deque
from collections import namedtuple
from rtree import index as rindex
import abc
import itertools
import math
import sys

RESOLUTION = 0.025  # mm/quanta

Point = namedtuple('Point', ['x', 'y'])


def GenerateRtreeIndex(shape_id, point_id):
  return shape_id << 16 | point_id


def GetShapeId(rtree_index):
  return rtree_index >> 16


def GetPointId(rtree_index):
  return rtree_index & 0xFFFF


def PointDistance(a, b):
  return math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))


def PointToBox(point):
  return (point.x, point.y, point.x, point.y)


class Polyline(abc.ABC):
  def __init__(self, points, pen=0):
    self._points = deque()
    self._points.extend(points)
    self._pen = pen

  @abc.abstractmethod
  def BoxesForRtree(self, shape_id):
    """Yield points as boxes for rtree."""

  @abc.abstractmethod
  def LastBox(self):
    """Return last point in this polyline as a box."""

  @abc.abstractmethod
  def Points(self):
    """Yield points."""

  @abc.abstractmethod
  def StartWith(self, point_id):
    """Rearranges points to start with this point id."""

  def Pen(self):
    return self._pen


class ClosedPolyline(Polyline):
  def __init__(self, points, pen):
    super().__init__(points, pen)
    self._start = 0

  def BoxesForRtree(self, shape_id):
    for i in range(len(self._points)):
      point = self._points[i]
      yield (
        GenerateRtreeIndex(shape_id, i),
        PointToBox(point),
        None)

  def StartWith(self, point_id):
    self._start = point_id

  def Points(self):
    modifier = 0
    if len(self._points) > 2:
      modifier = 1
    for i in range(len(self._points) + modifier):
      index = (i + self._start) % len(self._points)
      yield self._points[index]

  def LastBox(self):
    last_index = (len(self._points) + self._start - 1) % len(self._points)
    return PointToBox(self._points[last_index])


class OpenPolyline(Polyline):
  def __init__(self, points, pen):
    super().__init__(points, pen)
    self._forward_direction = True

  def BoxesForRtree(self, shape_id):
    yield (GenerateRtreeIndex(shape_id, 0), PointToBox(self._points[0]), None)
    yield (GenerateRtreeIndex(shape_id, 1), PointToBox(self._points[-1]), None)

  def StartWith(self, point_id):
    if point_id == 0:
      self._forward_direction = True
    else:
      self._forward_direction = False

  def Points(self):
    if self._forward_direction:
      for point in self._points:
        yield point
    else:
      for point in reversed(self._points):
        yield point

  def LastBox(self):
    if self._forward_direction:
      return PointToBox(self._points[0])
    else:
      return PointToBox(self._points[-1])


def _PrepareRtree(shape_list):
  p = rindex.Property()
  p.leaf_capacity = 1000
  p.variant = rindex.RT_Star
  p.fill_factor = 0.02

  def Points():
    for shape_id in range(len(shape_list)):
      shape = shape_list[shape_id]
      for box in shape.BoxesForRtree(shape_id):
        yield box

  rtree = rindex.Index(Points(), properties=p)
  return rtree


def _Sort(shape_deque, merge_dist):
  if not shape_deque:
    return None

  merge_dist_units = merge_dist / RESOLUTION

  # bootstrap with first line.
  sorted_shapes = deque()
  sorted_shapes.append(shape_deque[0])

  # copy to array for fast random access
  shape_array = list(shape_deque)
  rtree = _PrepareRtree(shape_array[1:])

  num_shapes = len(shape_array) - 1
  for i in range(num_shapes):
    if i % 500 == 0:
      print('{:} of {:}'.format(i, num_shapes))

    rtree_id = next(rtree.nearest(sorted_shapes[-1].LastBox()))
    shape_id = GetShapeId(rtree_id)

    nearest_shape = shape_array[shape_id]
    nearest_shape.StartWith(GetPointId(rtree_id))

    sorted_shapes.append(nearest_shape)

    for id, box, _ in nearest_shape.BoxesForRtree(shape_id):
      rtree.delete(id, box)

  return sorted_shapes


def sort_all(mixed_shapes, merge_dist=0):
  # Sort shapes by pen.
  categorized_shapes = {}
  sorted_shapes = {}
  for shape in mixed_shapes:
    pen = shape.Pen()
    if not pen in categorized_shapes:
      categorized_shapes[pen] = deque()
    categorized_shapes[pen].append(shape)

  for pen, shape_list in categorized_shapes.items():
    sorted_shapes[pen] = _Sort(shape_list, merge_dist)

  return sorted_shapes


def write_shapes(shapes, merge_dist, outfile):
  if not shapes:
    return

  outfile.write('SP%d' % (shapes[0].Pen() + 1))

  last_point = None
  for shape in shapes:
    point_counter = 0
    for point in shape.Points():
      if point_counter == 0:
        if (last_point is None or
            PointDistance(point, last_point) > merge_dist):
          outfile.write('PU%d,%d' % (point.x, point.y))
          outfile.write('PD')
      elif point_counter == 1:
        outfile.write('%d,%d' % (point.x, point.y))
      else:
        outfile.write(',%d,%d' % (point.x, point.y))
      point_counter += 1
      last_point = point
