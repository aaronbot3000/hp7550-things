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

kResolution = 0.025  # mm/quanta

kLetterX = 10 * 1016  # points
kLetterY = int(183.5 / kResolution)  # points
kLetterYOffset = int(12 / kResolution)  # points

kTabloidX = int(15.684 * 1016)  # points
kTabloidY = 10 * 1016  # points


Point = namedtuple('Point', ['x', 'y'])


def PointDistance(a, b):
  return math.sqrt(math.pow(a.x - b.x, 2) + math.pow(a.y - b.y, 2))


def PointToBox(point):
  return (point.x, point.y, point.x, point.y)


def RotX(x, y, rotation):
  return int(x * math.cos(rotation) - y * math.sin(rotation))


def RotY(x, y, rotation):
  return int(x * math.sin(rotation) + y * math.cos(rotation))


_NEXT_RTREE_ID = 0

class Shape(abc.ABC):
  def __init__(self, pen):
    self._pen = pen

    global _NEXT_RTREE_ID
    self._id = _NEXT_RTREE_ID
    _NEXT_RTREE_ID += self.NumberOfRtreeIdsConsumed()

  @abc.abstractmethod
  def NumberOfRtreeIdsConsumed(self):
    """The number of rtree IDs needed for this shape."""

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
  def WriteToFile(self, outfile, need_traverse, letter_offset):
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
    self._text = text
    self._start_position = start_position
    self._size = size
    self._angle = angle
    self._last_point = self.GenerateLastPoint()
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return 1

  def WriteToFile(self, outfile, _, letter_offset):
    yoffset = 0
    if letter_offset:
      yoffset = kLetterYOffset

    outfile.write(b'PU%d,%d' % (self._start_position.x,
                               self._start_position.y + yoffset))

    outfile.write(b'SI%.2f,%.2f' % (self._size[0], self._size[1]))
    run = math.cos(self._angle)
    rise = math.sin(self._angle)
    outfile.write(b'DI%.2f,%.2f' % (run, rise))
    outfile.write(b'LB%s\3' % self._text)

  def FirstPoint(self):
    return self._start_position

  def GenerateLastPoint(self):
    cm_to_quanta = 100 / kResolution

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

  def DrawIn(self, image, pen_map, scale):
    pass


class Arc(Shape):
  def __init__(self, center, diameter, start, sweep, chord, pen):
    """Draw an arc or circle.

    start: angle it starts the arc. negative is ok
      0 degrees is positive X axis, counterclockwise.. Enter None for random.
    """
    self._center = center
    self._start = start
    self._radius = diameter / 2.0
    self._sweep = sweep
    self._chord = chord

    self._forward_direction = True
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return 2

  def WriteToFile(self, outfile, need_traverse, letter_offset):
    start_point = self.FirstPoint()

    yoffset = 0
    if letter_offset:
      yoffset = kLetterYOffset

    if self._forward_direction:
      sweep = self._sweep
    else:
      sweep = -1 * self._sweep

    if need_traverse:
      outfile.write(b'PU%d,%dPD' % (start_point.x, start_point.y + yoffset))
    else:
      outfile.write(b'PD%d,%d' % (start_point.x, start_point.y + yoffset))

    outfile.write(b'AA%d,%d,%.2f' %
        (self._center.x, self._center.y + yoffset, np.degrees(sweep)))

    if self._chord is not None:
      outfile.write(b',%d' % np.degrees(self._chord))

  def FirstPoint(self):
    if self._forward_direction:
      return Point(self._center.x + RotX(self._radius, 0, self._start),
                   self._center.y + RotY(self._radius, 0, self._start))
    else:
      start = self._start + self._sweep
      return Point(self._center.x + RotX(self._radius, 0, start),
                   self._center.y + RotY(self._radius, 0, start))

  def LastPoint(self):
    if self._forward_direction:
      end = self._start + self._sweep
      return Point(self._center.x + RotX(self._radius, 0, end),
                   self._center.y + RotY(self._radius, 0, end))
    else:
      start = self._start + self._sweep
      return Point(self._center.x + RotX(self._radius, 0, self._start),
                   self._center.y + RotY(self._radius, 0, self._start))

  def BoxesForRtree(self, shape_id):
    forward_point = Point(self._center.x + RotX(self._radius, 0, self._start),
                          self._center.y + RotY(self._radius, 0, self._start))

    angle = self._start + self._sweep
    reverse_point = Point(self._center.x + RotX(self._radius, 0, angle),
                          self._center.y + RotY(self._radius, 0, angle))

    yield (self._id, PointToBox(forward_point), (shape_id, 0))
    yield (self._id + 1, PointToBox(reverse_point), (shape_id, 1))

  def StartWith(self, point_id):
    if point_id == 0:
      self._forward_direction = True
    else:
      self._forward_direction = False

  def Chainable(self):
    return True

  def DrawIn(self, image, pen_map, scale):
    chord = 5
    if self._chord is not None:
      chord = self._chord

    polyline = cv2.ellipse2Poly(
        (int(self._center.x * scale), int(self._center.y * scale)),
        (int(self._radius * scale), int(self._radius * scale)),
        0,  # no rotation of the circle.
        int(np.degrees(self._start)),
        int(np.degrees(self._start + self._sweep)),
        int(np.degrees(chord)))
    cv2.polylines(image,
        [polyline],
        False,
        pen_map[self._pen])


class Polyline(Shape):
  @abc.abstractmethod
  def Points(self):
    """Yield points."""

  def _Encode(self, number):
    base = 64

    # Add sign bit.
    number = int(2 * number)  # no funny business
    if number < 0:
      number *= -1
      number += 1

    digits = []  # little endian
    while number >= base:
      digits.append(63 + (number % base))
      number = int(number / base)
    digits.append(number + 191)
    return bytes(digits)

  def WriteToFile(self, outfile, need_traverse, letter_offset):
    yoffset = 0
    if letter_offset:
      yoffset = kLetterYOffset

    zero = self._Encode(0)

    last_point = None
    for point in self.Points():
      if last_point is None:
        x = self._Encode(point.x)
        y = self._Encode(point.y + yoffset)
        if need_traverse:
          outfile.write(b'PE<=')
          outfile.write(x)
          outfile.write(y)
          outfile.write(zero)
          outfile.write(zero)
        else:
          outfile.write(b'PE=')
          outfile.write(x)
          outfile.write(y)
      else:
        x = self._Encode(point.x - last_point.x)
        y = self._Encode(point.y - last_point.y)
        outfile.write(x)
        outfile.write(y)
      last_point = point

    outfile.write(b';')

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
    self._points = []
    self._points.extend(points)
    self._start = 0
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return len(self._points)

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
    self._points = deque()
    self._points.extend(points)
    self._forward_direction = True
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return 2

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

    points = []
    points.append(Point(RotX(-1 * half, -1 * half, rotation) + center.x,
                        RotY(-1 * half, -1 * half, rotation) + center.y));
    points.append(Point(RotX(half, -1 * half, rotation) + center.x,
                        RotY(half, -1 * half, rotation) + center.y));
    points.append(Point(RotX(half, half, rotation) + center.x,
                        RotY(half, half, rotation) + center.y));
    points.append(Point(RotX(-1 * half, half, rotation) + center.x,
                        RotY(-1 * half, half, rotation) + center.y));

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

  print('Remaining items in rtree: %d' % rtree.count((-300000, -300000,
                                                      300000, 300000)))
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


def WriteShapes(outfile, pen, shapes, merge_dist, page_size):
  if not shapes:
    return

  outfile.write(b'SP%d' % pen)

  last_point = None
  need_traverse = True

  for shape in shapes:
    need_traverse = (last_point is None or
        PointDistance(last_point, shape.FirstPoint()) > merge_dist)

    shape.WriteToFile(outfile, need_traverse, page_size == 'letter')

    if shape.Chainable():
      last_point = shape.LastPoint()
    else:
      last_point = None

def WriteAllShapes(outfile, sorted_shapes, merge_dist, page_size):
  outfile.write(b'IN')
  for pen, shapes in sorted_shapes.items():
    WriteShapes(outfile, pen, shapes, merge_dist, page_size)
  outfile.write(b'PUSP0;\n')

def SortAllAndWrite(outfile, mixed_shapes, merge_dist, page_size):
  all_sorted = SortAll(mixed_shapes)
  WriteAllShapes(outfile, all_sorted, merge_dist, page_size)

PREVIEW_X = 1081  # pixels

def ShowPreview(mixed_shapes, page_size, pen_map):
  if page_size == 'letter':
    scale = PREVIEW_X / kLetterX
    preview_y = int(kLetterY * scale)
  elif page_size == 'tabloid':
    scale = PREVIEW_X / kTabloidX
    preview_y = int(kTabloidY * scale)
  else:
    assert 'Bruh you typoed page size: %s' % page_size

  # Numpy arrays are row, col notation, or y, x. Everything else is
  # x, y, so the y and x are reversed.
  preview_dims = (preview_y, PREVIEW_X, 3)
  image = np.ones(preview_dims, np.uint8) * 255

  for shape in mixed_shapes:
    shape.DrawIn(image, pen_map, scale)

  # Put the origin at the bottom left corner, z axis pointing out
  # of the screen.
  image = np.flip(image, 0)
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
