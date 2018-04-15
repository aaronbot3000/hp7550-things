"""Plotter tools.

For the HPGL plotters, X axis is always the longer dimension of the page.
"""
from collections import deque
from collections import namedtuple
from rtree import index as rindex
import abc
import cv2
import itertools
import math
import numpy as np
import sys

RESOLUTION = 0.025  # mm/quanta

Point = namedtuple('Point', ['x', 'y'])


def PointDistance(a, b):
  return math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))


def PointToBox(point):
  return (point.x, point.y, point.x, point.y)


_NEXT_RTREE_ID = 0

class Shape(abc.ABC):
  def __init__(self, pen):
    self._pen = pen

  @abc.abstractmethod
  def BoxesForRtree(self, shape_id):
    """Yield valid start points as boxes for rtree."""

  @abc.abstractmethod
  def FirstPoint(self):
    """Return first point in this shape."""

  @abc.abstractmethod
  def LastPoint(self):
    """Return last point in this shape."""

  def LastBox(self):
    return PointToBox(self.LastPoint())

  @abc.abstractmethod
  def WriteToFile(self, outfile, need_traverse):
    """Write this shape to file."""

  @abc.abstractmethod
  def StartWith(self, point_id):
    """Rearranges points to start with this point id."""

  @abc.abstractmethod
  def Chainable(self):
    """Whether this operation leaves the pen down for chaining operations."""

  @abc.abstractmethod
  def DrawIn(self, image, pen_map, scale):
    """Draw this shape in the given array."""

  def Pen(self):
    return self._pen


class Label(Shape):
  def __init__(self, text, start_position, size, angle, pen):
    """Write a label.

    256 character max.
    size is (width, height) in cm
    angle in radians.
    """
    super().__init__(pen)
    self._text = text
    self._start_position = start_position
    self._size = size
    self._angle = angle

    self._last_point = self.GenerateLastPoint()
    
    global _NEXT_RTREE_ID
    self._id = _NEXT_RTREE_ID
    _NEXT_RTREE_ID += 1

  def WriteToFile(self, outfile, _):
    outfile.write('PU%d,%d' % (self._start_position.x, self._start_position.y))
    outfile.write('SI%.3f,%.3f' % (self._size[0], self._size[1]))
    run = math.cos(self._angle)
    rise = math.sin(self._angle)
    outfile.write('DI%.3f,%.3f' % (run, rise))
    outfile.write('LB%s\3' % self._text)

  def FirstPoint(self):
    return self._start_position

  def GenerateLastPoint(self):
    cm_to_quanta = 100 / RESOLUTION

    length = len(self._text) * self._size[0] * cm_to_quanta * 1.5
    height = self._size[1] * cm_to_quanta

    # Consider the length of the string.
    run = int(math.cos(self._angle) * length)
    rise = int(math.sin(self._angle) * length)
    
    # Consider half the height of the string.
    run += int(math.cos(self._angle + math.pi / 2) * height / 2)
    rise += int(math.sin(self._angle + math.pi / 2) * height / 2)

    return Point(self._start_position.x + run, self._start_position.y + rise)

  def LastPoint(self):
    return self._last_point

  def BoxesForRtree(self, shape_id):
    yield (self._id,
        PointToBox(self._start_position),
        (shape_id, 0))

  def StartWith(self, point_id):
    pass

  def Chainable(self):
    return False


class Polyline(Shape):
  @abc.abstractmethod
  def Points(self):
    """Yield points."""

  def WriteToFile(self, outfile, need_traverse):
    point_counter = 0
    for point in self.Points():
      if point_counter == 0 and need_traverse:
        outfile.write('PU%d,%dPD' % (point.x, point.y))
      elif point_counter == 1 and need_traverse:
        outfile.write('%d,%d' % (point.x, point.y))
      else:
        outfile.write(',%d,%d' % (point.x, point.y))
      point_counter += 1

  def Chainable(self):
    return True

  def DrawIn(self, image, pen_map, scale):
    all_points = deque()
    for point in self.Points():
      all_points.append([int(point.x * scale),
                         int(point.y * scale)])
    np_points = np.array(all_points, np.int32)
    np_points = np_points.reshape((-1, 1, 2))
    cv2.polylines(image, [np_points], False, pen_map[self._pen])


class ClosedPolyline(Polyline):
  def __init__(self, points, pen):
    super().__init__(pen)
    self._points = []
    self._points.extend(points)
    self._start = 0

    global _NEXT_RTREE_ID
    self._id = _NEXT_RTREE_ID
    _NEXT_RTREE_ID += len(self._points)

  def BoxesForRtree(self, shape_id):
    for i in range(len(self._points)):
      point = self._points[i]
      yield (self._id + i, PointToBox(point), (shape_id, i))

  def StartWith(self, point_id):
    self._start = point_id

  def Points(self):
    modifier = 0
    if len(self._points) > 2:
      modifier = 1

    for i in range(len(self._points) + modifier):
      index = (i + self._start) % len(self._points)
      yield self._points[index]

  def FirstPoint(self):
    return self._points[self._start]

  def LastPoint(self):
    last_index = (len(self._points) + self._start - 1) % len(self._points)
    return self._points[last_index]


class OpenPolyline(Polyline):
  def __init__(self, points, pen):
    super().__init__(pen)
    self._points = deque()
    self._points.extend(points)
    self._forward_direction = True

    global _NEXT_RTREE_ID
    self._id = _NEXT_RTREE_ID
    _NEXT_RTREE_ID += 2

  def BoxesForRtree(self, shape_id):
    yield (self._id, PointToBox(self._points[0]), (shape_id, 0))
    yield (self._id + 1, PointToBox(self._points[-1]), (shape_id, 1))

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

  def FirstPoint(self):
    if self._forward_direction:
      return self._points[0]
    else:
      return self._points[-1]

  def LastPoint(self):
    if self._forward_direction:
      return self._points[-1]
    else:
      return self._points[0]


class Square(ClosedPolyline):
  def __init__(self, length, center, rotation, pen):
    """Make a square. Rotation in radians."""
    half = length / 2.0

    def RotX(x, y):
      return int(x * math.cos(rotation) - y * math.sin(rotation))
    def RotY(x, y):
      return int(x * math.sin(rotation) + y * math.cos(rotation))

    points = []
    points.append(Point(RotX(-1 * half, -1 * half) + center.x,
                        RotY(-1 * half, -1 * half) + center.y));
    points.append(Point(RotX(half, -1 * half) + center.x,
                        RotY(half, -1 * half) + center.y));
    points.append(Point(RotX(half, half) + center.x,
                        RotY(half, half) + center.y));
    points.append(Point(RotX(-1 * half, half) + center.x,
                        RotY(-1 * half, half) + center.y));

    super().__init__(points, pen)


all_ids = set()
all_box = set()

def _PrepareRtree(shape_list):
  p = rindex.Property()
  p.leaf_capacity = 1000
  p.variant = rindex.RT_Star
  p.fill_factor = 0.02

  def Points():
    for shape_id in range(1, len(shape_list)):
      shape = shape_list[shape_id]
      for box in shape.BoxesForRtree(shape_id):
        all_ids.add(box[0])
        all_box.add(box[1])
        yield box

  rtree = rindex.Index(Points(), properties=p)
  return rtree


def _Sort(shape_deque):
  if not shape_deque:
    return None

  # bootstrap with first line.
  sorted_shapes = deque()
  sorted_shapes.append(shape_deque[0])

  if len(shape_deque) == 1:
    return sorted_shapes
    
  # copy to array for fast random access
  shape_array = list(shape_deque)
  rtree = _PrepareRtree(shape_array)

  num_shapes = len(shape_array) - 1
  for i in range(num_shapes):
    if i % 500 == 0:
      print('Sorted %d of %d.' % (i, num_shapes))

    nearest = next(rtree.nearest(sorted_shapes[-1].LastBox(), objects='raw'))
    shape_id = nearest[0]

    nearest_shape = shape_array[shape_id]
    nearest_shape.StartWith(nearest[1])

    sorted_shapes.append(nearest_shape)

    for id, box, _ in nearest_shape.BoxesForRtree(0):
      rtree.delete(id, box)

  return sorted_shapes


def SortAll(mixed_shapes):
  # Sort shapes by pen.
  categorized_shapes = {}
  sorted_shapes = {}
  for shape in mixed_shapes:
    pen = shape.Pen()
    if not pen in categorized_shapes:
      categorized_shapes[pen] = deque()
    categorized_shapes[pen].append(shape)

  for pen, shape_list in categorized_shapes.items():
    sorted_shapes[pen] = _Sort(shape_list)

  return sorted_shapes


def WriteShapes(outfile, pen, shapes, merge_dist):
  if not shapes:
    return

  outfile.write('SP%d' % pen)

  last_point = None
  need_traverse = True

  for shape in shapes:
    need_traverse = (last_point is None or
        PointDistance(last_point, shape.FirstPoint()) > merge_dist)

    shape.WriteToFile(outfile, need_traverse)

    if shape.Chainable():
      last_point = shape.LastPoint()
    else:
      last_point = None

def WriteAllShapes(outfile, sorted_shapes, merge_dist):
  outfile.write('IN')
  for pen, shapes in sorted_shapes.items():
    WriteShapes(outfile, pen, shapes, merge_dist)
  outfile.write('PUSP0;\n')

def SortAllAndWrite(outfile, mixed_shapes, merge_dist):
  all_sorted = SortAll(mixed_shapes)
  WriteAllShapes(outfile, all_sorted, merge_dist)

PREVIEW_X = 1081  # pixels

LETTER_X = 10 * 1016  # points
LETTER_Y = 8.154 * 1016  # points

TABLOID_X = 15.684 * 1016  # points
TABLOID_Y = 10 * 1016  # points

def ShowPreview(mixed_shapes, page_size, pen_map):
  if page_size == 'letter':
    scale = PREVIEW_X / LETTER_X
    preview_y = int(LETTER_Y * scale)
  elif page_size == 'tabloid':
    scale = PREVIEW_X / TABLOID_X
    preview_y = int(TABLOID_Y * scale)
  else:
    assert 'Bruh you typoed page size: %s' % page_size

  preview_dims = (preview_y, PREVIEW_X, 3)
  image = np.ones(preview_dims, np.uint8) * 255

  for shape in mixed_shapes:
    shape.DrawIn(image, pen_map, scale)

  cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Preview', image)

  return_value = False
  while True:
    key = cv2.waitKey(0)
    print(key)
    if key == 1048586:
      return_value = True
      break;
    elif key == -1 or key == 1048603 or key == 1048689:
      break

  cv2.destroyAllWindows()
  return return_value
