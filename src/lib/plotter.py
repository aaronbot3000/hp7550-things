"""Plotter tools.

For the HPGL plotters, X axis is always the longer dimension of the page.
Origin is the bottom left corner.
"""
from collections import deque
from collections import namedtuple
from rtree import index as rindex
import abc
import copy
import cv2
import itertools
import math
import numpy as np
import sys

kResolution = 0.025  # mm / point

kLetterX = 10 * 1016  # points
kLetterY = int(183.5 / kResolution)  # points
kLetterYOffset = int(12 / kResolution)  # points

kTabloidX = int(15.684 * 1016)  # points
kTabloidY = 10 * 1016  # points

kPreviewAlpha = 255
kLetterLineWidth = 2
kTabloidLineWidth = 1
kLineType = cv2.LINE_AA

kMaxSlopeDiff = 1


class Point(object):
  def __init__(self, x, y=None):
    if y == None:
      self.x = int(x[0])
      self.y = int(x[1])
    else:
      self.x = int(x)
      self.y = int(y)

  def __setattr__(self, name, value):
    super().__setattr__(name, int(value))

  def __str__(self):
    return 'Point: %d, %d' % (self.x, self.y)

  def __repr__(self):
    return self.__str__()


class PenState(object):
  def __init__(self, position, position_exact, pen_is_up, last_operation=None):
    self.position = position
    self.position_exact = position_exact
    self.pen_is_up = pen_is_up
    self.last_operation = last_operation


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
    self._y_offset = 0
    _NEXT_RTREE_ID += self.NumberOfRtreeIdsConsumed()

  def SetLetterOffset(self, letter_offset):
    if letter_offset:
      self._y_offset = kLetterYOffset

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
  def WriteToFile(self, outfile, pen_state, merge_dist):
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
    self._start_position = copy.copy(start_position)
    self._size = size
    self._angle = angle
    self._last_point = self.GenerateLastPoint()
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return 1

  def WriteToFile(self, outfile, pen_state, _):
    start_point = self.FirstPoint()
    if pen_state.last_operation != 'PU':
      outfile.write(b'PU')
    outfile.write(b'+%d%+dPD' % (start_point.x,
                                 start_point.y + self._y_offset))

    outfile.write(b'SI%.2f,%.2f' % (self._size[0], self._size[1]))
    run = math.cos(self._angle)
    rise = math.sin(self._angle)
    outfile.write(b'DI%.2f%+.2f' % (run, rise))
    outfile.write(b'LB')
    outfile.write(bytes(self._text, encoding='ascii'))
    outfile.write(b'\3')

    pen_state.position = self.LastPoint()
    pen_state.position_exact = False
    pen_state.pen_is_up = True
    pen_state.last_operation = 'LB'

    return pen_state

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

  def DrawIn(self, image, pen_map, scale, line_width):
    pass


class Arc(Shape):
  def __init__(self, center, diameter, start, sweep, chord, pen):
    """Draw an arc or circle.

    start: angle it starts the arc. negative is ok
      0 degrees is positive X axis, counterclockwise.
    chord: length of each arc approximation line, in radians.
    """
    self._center = copy.copy(center)
    self._start = start
    self._radius = diameter / 2.0
    self._sweep = sweep

    if chord is not None:
      self._chord = chord
    else:
      self._chord = None

    self._forward_direction = True
    super().__init__(pen)

  def NumberOfRtreeIdsConsumed(self):
    return 2

  def WriteToFile(self, outfile, pen_state, merge_dist):
    if self._forward_direction:
      sweep = self._sweep
    else:
      sweep = -1 * self._sweep

    start_point = self.FirstPoint()

    if (merge_dist is not None and
        PointDistance(pen_state.position, start_point) < merge_dist):
      if pen_state.last_operation != 'PD':
        outfile.write(b'PD')
      outfile.write(b'+%d%+d' % (start_point.x,
                                 start_point.y + self._y_offset))
    else:
      if pen_state.last_operation != 'PU':
        outfile.write(b'PU')
      outfile.write(b'+%d%+dPD' % (start_point.x,
                                   start_point.y + self._y_offset))

    cx = self._center.x - start_point.x
    cy = self._center.y - start_point.y
    outfile.write(b'AR%d%+d%+.2f' %
        (cx, cy, np.degrees(sweep)))

    if self._chord is not None:
      outfile.write(b'%+d' % math.ceil(np.degrees(self._chord)))

    pen_state.position = self.LastPoint(),
    pen_state.position_exact=False,
    pen_state.pen_is_up=False,
    pen_state.last_operation='AR'
    return pen_state

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

  def DrawIn(self, image, pen_map, scale, line_width):
    if self._chord is not None:
      chord = np.degrees(self._chord)
    else:
      chord = 5

    polyline = cv2.ellipse2Poly(
        (int(self._center.x * scale), int(self._center.y * scale)),
        (int(self._radius * scale), int(self._radius * scale)),
        0,  # no rotation of the circle.
        int(np.degrees(self._start)),
        int(np.degrees(self._start + self._sweep)),
        int(chord))
    cv2.polylines(image,
        [polyline],
        False,
        pen_map[self._pen] + (kPreviewAlpha,),
        thickness=line_width,
        lineType=kLineType)


class Polyline(Shape):
  @abc.abstractmethod
  def Points(self):
    """Yield points."""

  def _Diff(self, p2, p1, n):
    xp2 = p1.x - p2.x
    yp2 = p1.y - p2.y

    xp1 = n.x - p1.x
    yp1 = n.y - p1.y

    xn = n.x - p2.x
    yn = n.y - p2.y

    # Direction changed.
    if np.sign(xp2) != np.sign(xp1) or np.sign(yp2) != np.sign(yp1):
      return kMaxSlopeDiff + 1

    # Horizontal or vertical line or dot.
    if xp2 == 0 or yp2 == 0:
      return 0

    diff = abs((xn / xp2 * yp2) - yn)
    return diff

  def _CopyAndDedupe(self, origin):
    if len(origin) <= 2:
      return copy.deepcopy(origin)

    prev2 = origin[0]
    prev1 = origin[1]
    dest = deque()

    dest.append(copy.deepcopy(prev2))
    dest.append(copy.deepcopy(prev1))

    count = 0
    for next_point in origin:
      count += 1
      if count < 3:
        continue
      if self._Diff(prev2, prev1, next_point) < kMaxSlopeDiff:
        dest.pop()
        prev1 = next_point
      else:
        prev2 = prev1
        prev1 = next_point
      dest.append(copy.deepcopy(next_point))
    return dest

  def _Encode(self, number):
    base = 64

    # Add sign bit.
    scaled = abs(2 * int(number))  # no funny business
    if number < 0:
      scaled += 1

    digits = []  # little endian
    while scaled >= base:
      digits.append(63 + (scaled % base))
      scaled = int(scaled / base)
    digits.append(scaled + 191)
    return bytes(digits)

  def WriteFirstPoint(self, outfile, pen_state, merge_dist):
    zero = self._Encode(0)
    point = self.FirstPoint()

    # This is the first point.
    need_zeros = False
    if (merge_dist is not None and
        PointDistance(pen_state.position, start_point) < merge_dist):
      # This shape can be merged into the previous shape.
      # First check if we need to enter the polyline encoded command.
      if pen_state.last_operation != 'PE':
        outfile.write(b'PE')
    else:
      # This shape cannot be merged.
      if pen_state.last_operation != 'PE':
        outfile.write(b'PE')
      # Write the pen up command.
      outfile.write(b'<')
      need_zeros = True

    # See if we have a handle on the position.
    if pen_state.position_exact:
      outfile.write(self._Encode(point.x - pen_state.position.x))
      outfile.write(self._Encode(point.y - pen_state.position.y))
    else:
      # If not, write the next coordinate with absolute position.
      outfile.write(b'=')
      outfile.write(self._Encode(point.x))
      outfile.write(self._Encode(point.y + self._y_offset))

    # Write zeros if needed as recommended by the reference guide.
    if need_zeros:
      outfile.write(zero)
      outfile.write(zero)

  def WriteToFile(self, outfile, pen_state, merge_dist):
    last_point = None
    for point in self.Points():
      if last_point is None:
        self.WriteFirstPoint(outfile, pen_state, merge_dist)
      else:
        outfile.write(self._Encode(point.x - last_point.x))
        outfile.write(self._Encode(point.y - last_point.y))
      last_point = point

    pen_state.position = self.LastPoint()
    pen_state.position_exact = True
    pen_state.pen_is_up = False,
    pen_state.last_operation = 'PE'

    return pen_state

  def Chainable(self):
    return True

  def DrawIn(self, image, pen_map, scale, line_width):
    all_points = deque()
    for point in self.Points():
      all_points.append([int(point.x * scale),
                         int(point.y * scale)])
    np_points = np.array(all_points, np.int32)
    np_points = np_points.reshape((-1, 1, 2))
    cv2.polylines(image,
                  [np_points],
                  False,
                  pen_map[self._pen] + (kPreviewAlpha,),
                  thickness=line_width,
                  lineType=kLineType)


class ClosedPolyline(Polyline):
  def __init__(self, points, pen):
    assert len(points) > 1, 'No single point lines.'
    self._points = self._CopyAndDedupe(points)
    self._start = 0
    super().__init__(pen)

  def __str__(self):
    return str(self._points)

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
    return self.FirstPoint()


class OpenPolyline(Polyline):
  def __init__(self, points, pen):
    assert len(points) > 1, 'No single point lines.'
    self._points = deque()
    self._points = self._CopyAndDedupe(points)
    self._forward_direction = True
    super().__init__(pen)

  def __str__(self):
    return str(self._points)

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


def _PrepareRtree(shape_list):
  p = rindex.Property()
  p.leaf_capacity = 1000
  p.variant = rindex.RT_Star
  p.fill_factor = 0.02

  def Points():
    for shape_id in range(1, len(shape_list)):
      shape = shape_list[shape_id]
      for box in shape.BoxesForRtree(shape_id):
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


def CategorizeShapes(mixed_shapes):
  # Sort shapes by pen.
  categorized_shapes = {}
  for shape in mixed_shapes:
    pen = shape.Pen()
    if not pen in categorized_shapes:
      categorized_shapes[pen] = deque()
    categorized_shapes[pen].append(shape)
  return categorized_shapes


def SortAll(mixed_shapes, reorder=True):
  categorized_shapes = CategorizeShapes(mixed_shapes)
  if reorder:
    sorted_shapes = {}
    for pen, shape_list in categorized_shapes.items():
      sorted_shapes[pen] = _Sort(shape_list)
    return sorted_shapes
  else:
    return categorized_shapes


def WriteShapes(outfile, pen, shapes, merge_dist, page_size):
  if not shapes:
    return

  if pen < 0:
    return

  outfile.write(b'SP%d' % (pen + 1))

  pen_state = PenState(None, position_exact=False, pen_is_up=True)
  need_letter_offset = page_size == 'letter'

  previous_shape = None
  for shape in shapes:
    shape.SetLetterOffset(need_letter_offset)
    merge_dist_quanta = None
    if (previous_shape is not None and
        previous_shape.Chainable() and
        shape.Chainable()):
      merge_dist_quanta = merge_dist / kResolution

    if (pen_state.last_operation == 'PE' and
        not issubclass(type(shape), Polyline)):
      outfile.write(b';')
      pen_state.last_operation = None

    pen_state = shape.WriteToFile(outfile, pen_state, merge_dist_quanta)

  if pen_state.last_operation == 'PE':
    outfile.write(b';')


def WriteAllShapes(outfile, sorted_shapes, merge_dist, page_size):
  outfile.write(b'INPA')
  for pen, shapes in sorted_shapes.items():
    WriteShapes(outfile, pen, shapes, merge_dist, page_size)
  outfile.write(b'PUSP0;\n')


def SortAllAndWrite(outfile, mixed_shapes, merge_dist, page_size, reorder=True):
  all_sorted = SortAll(mixed_shapes, reorder)
  WriteAllShapes(outfile, all_sorted, merge_dist, page_size)

PREVIEW_X = 1200  # pixels
RENDER_X = 2400  # pixels

def ShowPreview(mixed_shapes, page_size, pen_map, override_line_width=None):
  if page_size == 'letter':
    scale = RENDER_X / kLetterX
    render_y = int(kLetterY * scale)
    preview_y = int(kLetterY * PREVIEW_X / kLetterX)
    line_width = kLetterLineWidth
  elif page_size == 'tabloid':
    scale = RENDER_X / kTabloidX
    render_y = int(kTabloidY * scale)
    preview_y = int(kTabloidY * PREVIEW_X / kTabloidX)
    line_width = kTabloidLineWidth
  else:
    assert 'Bruh you typoed page size: %s' % page_size

  if override_line_width is not None:
    line_width = override_line_width

  # Numpy arrays are row, col notation, or y, x. Everything else is
  # x, y, so the y and x are reversed.
  render_dims = (render_y, RENDER_X)
  accumulator = np.zeros(render_dims + (4,), np.float64)
  acc_r, acc_g, acc_b, acc_a = np.dsplit(accumulator, 4)

  print('Sorting shapes.')
  categorized_shapes = CategorizeShapes(mixed_shapes)
  color_planes = []
  print('Drawing shapes.')
  for _, shape_list in categorized_shapes.items():
    color_planes.append(np.zeros(render_dims + (4,), np.uint8))
    for shape in shape_list:
      shape.DrawIn(color_planes[-1], pen_map, scale, line_width)

  # Accumulate colors.
  alphacount = np.zeros(render_dims + (1,))
  temp = np.zeros(render_dims + (4,))
  plane_count = 0
  for plane in color_planes:
    plane_count += 1
    print('Summing plane %d of %d.' % (plane_count, len(color_planes)))
    np.copyto(temp, plane)
    plane_r, plane_g, plane_b, plane_a = np.dsplit(temp, 4)

    np.multiply(plane_r, plane_a, out=plane_r)
    np.add(acc_r, plane_r, out=acc_r)

    np.multiply(plane_g, plane_a, out=plane_g)
    np.add(acc_g, plane_g, out=acc_g)

    np.multiply(plane_b, plane_a, out=plane_b)
    np.add(acc_b, plane_b, out=acc_b)

    np.add(acc_a, plane_a, acc_a)

    # Keep track of how many non-zero alpha values for each pixel.
    np.add(alphacount, 1, out=alphacount, where=np.greater(plane_a, 0))

  print('Averaging.')
  # Average colors.
  nonzeros = np.greater(alphacount, 0)
  np.divide(acc_r, acc_a, out=acc_r, where=nonzeros)
  np.divide(acc_g, acc_a, out=acc_g, where=nonzeros)
  np.divide(acc_b, acc_a, out=acc_b, where=nonzeros)
  np.divide(acc_a, alphacount, out=acc_a, where=nonzeros)

  print('Applying transparency and adding background.')
  # Multiply alpha transparency.
  np.multiply(acc_r, acc_a, out=acc_r)
  np.divide(acc_r, 255, out=acc_r)

  np.multiply(acc_g, acc_a, out=acc_g)
  np.divide(acc_g, 255, out=acc_g)

  np.multiply(acc_b, acc_a, out=acc_b)
  np.divide(acc_b, 255, out=acc_b)

  inv_alpha = np.subtract(255, acc_a)

  # Add white background.
  np.add(acc_r, inv_alpha, out=acc_r)
  np.add(acc_g, inv_alpha, out=acc_g)
  np.add(acc_b, inv_alpha, out=acc_b)

  # Reshape into non-alpha image and convert to uint8 for opencv's UI.
  values, _ = np.dsplit(accumulator, [3])
  needs_casting = values.reshape(render_dims + (3,))
  combined_output = np.array(needs_casting, np.uint8)

  print('Done')
  preview = cv2.resize(combined_output, (PREVIEW_X, preview_y),
      interpolation=cv2.INTER_AREA)

  # Put the origin at the bottom left corner, z axis pointing out
  # of the screen.
  preview = np.flip(preview, 0)
  # Fix the stupid RGB to BGR.
  preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
  cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('Preview', preview)

  return_value = False
  while True:
    key = cv2.waitKey(0)
    if key == 10:  # Enter
      return_value = True
      break;
    elif key == -1 or key == 27 or key == 113:  # ESC or q
      break

  cv2.destroyAllWindows()
  return return_value
