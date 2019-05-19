#!/usr/bin/python3
from collections import deque
import cv2
import math
import numpy as np
import random
import sys

import lib.plotter as plotter
import lib.program as program
import lib.pens as pens

kPageSize = 'letter'  # letter or tabloid

kPitch = int(2.5 / plotter.kResolution)

kWiggleRadius = 1 / plotter.kResolution
kWiggles = 2

kCenterPosition = 'center'  # center corner random very_random

kFillCorners = True

kMaxStepover = 2 / plotter.kResolution
kMaxAngle = 0.9

kStartDiameter = 1 / plotter.kResolution

kCmykColors = [
    pens.PURE['yellow'],
    pens.PURE['magenta'],
    pens.PURE['cyan'],
    pens.PURE['black'],
]

kPotooColors = [
    pens.SARASA['orange'],
    pens.SARASA['fuchsia'],
    pens.SARASA['light blue'],
    pens.SARASA['black'],
]

kPenMap = kPotooColors

random.seed(0)

class Spiraler(program.Program):
  def __init__(self, paper_type):
    super().__init__(paper_type)

  def _GetSpiralDiameter(self, angle, diameter_offset=0):
    return (self._pitch *angle /
        (2 * math.pi) + diameter_offset + kStartDiameter)

  def _GetSpiralPosition(self, angle, diameter, angle_offset=0):
    return np.array((
          diameter * math.cos(angle + angle_offset),
          diameter * math.sin(angle + angle_offset))) + self._center

  def _ProcessAngle(self, t, previous_angle, start_angle, pen, iteration):
    half_step = (t + previous_angle) / 2
    diameter = self._GetSpiralDiameter(half_step)
    point = self._GetSpiralPosition(half_step, diameter, start_angle)

    image_position = self.GetImagePosition(point)

    if image_position is None:
      if len(self._points) > 1:
        self._shapes.append(plotter.OpenPolyline(self._points, pen))
      self._points.clear()
      if t - self._last_angle_in_image > (2 * math.pi):
        return False
      return self._fill_corners

    self._last_angle_in_image = t
    pixel = self._filter_image[int(image_position[1]),
                               int(image_position[0])]
    inv_color = 255 - pixel

    if pen < 3:
      value = inv_color[pen]
      value -= min(inv_color)
    else:
      value = min(inv_color)

    value = 255 - value

    threshold = 170 * (1 - iteration / 2) + 50
    if (previous_angle is None or value >= threshold):
      if len(self._points) > 1:
        self._shapes.append(plotter.OpenPolyline(self._points, pen))
      self._points.clear()
      #self._points.append(plotter.Point(point))
      return True

    for i in range(self._num_wiggles):
      angle = (previous_angle +
          i / self._num_wiggles * (t - previous_angle))
      wiggle_amount = (
          self._wiggle_radius * self._mult * (threshold - value) / threshold)

      offset_diameter = self._GetSpiralDiameter(angle, wiggle_amount)
      spiral_point = self._GetSpiralPosition(
        angle, offset_diameter, start_angle)

      self._points.append(plotter.Point(spiral_point))
      self._mult *= -1
    return True

  def DrawSpiral(self, start_angle, pen, iteration):
    self._points = deque()
    self._shapes = deque()
    self._last_angle_in_image = 0

    t = 0
    previous_angle = start_angle
    while self._ProcessAngle(t, previous_angle, start_angle, pen, iteration):
      previous_angle = t
      t += min(kMaxStepover / self._GetSpiralDiameter(t), kMaxAngle)
    return self._shapes

  def DoSwirls(self, image, pitch, num_wiggles, wiggle_radius,
      spiral_center, fill_corners):
    self._filter_image = self._InitializeImage(
        image, pitch + wiggle_radius, pitch)
    self._pitch = pitch
    self._num_wiggles = num_wiggles
    self._wiggle_radius = wiggle_radius
    self._fill_corners = fill_corners

    self._mult = 1

    if spiral_center == 'center':
      self._center = self._image_origin + (self._image_dim_plot / 2)
    elif spiral_center == 'corner':
      self._center = self._image_origin
    elif spiral_center == 'random' or spiral_center == 'very_random':
      rand = np.array((random.random(), random.random()))
      self._center = self._image_origin + self._image_dim_plot * rand

    shapes = deque()

    for iteration in range(2):
      start_angle_offset = random.random() * math.pi
      #rand = np.array((random.random(), random.random()))
      #self._center = self._image_origin + self._image_dim_plot * rand
      shapes.extend(self.DrawSpiral(math.pi / 2 + start_angle_offset, 0, iteration))
      if spiral_center == 'very_random':
        rand = np.array((random.random(), random.random()))
        self._center = self._image_origin + self._image_dim_plot * rand
      shapes.extend(self.DrawSpiral(math.pi + start_angle_offset, 1, iteration))

      if spiral_center == 'very_random':
        rand = np.array((random.random(), random.random()))
        self._center = self._image_origin + self._image_dim_plot * rand
      shapes.extend(self.DrawSpiral(3 * math.pi / 2 + start_angle_offset, 2, iteration))

      if spiral_center == 'very_random':
        rand = np.array((random.random(), random.random()))
        self._center = self._image_origin + self._image_dim_plot * rand
      shapes.extend(self.DrawSpiral(0 + start_angle_offset, 3, iteration))

    return shapes


def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)

  spiraler = Spiraler(kPageSize)
  shapes = spiraler.DoSwirls(
      image, kPitch, kWiggles, kWiggleRadius, kCenterPosition, kFillCorners)

  if not plotter.ShowPreview(shapes, kPageSize, kPenMap):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter.SortAllAndWrite(dest, shapes, 0.7, kPageSize, reorder=False)

if __name__ == "__main__":
  main()
