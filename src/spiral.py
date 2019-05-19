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

kPitch = int(3 / plotter.kResolution)

kWiggleRadius = 1.4 / plotter.kResolution
kWiggles = 5

kCenterPosition = 'center'  # center corner random

kMaxStepover = 0.5 / plotter.kResolution
kMaxAngle = 0.3

kStartDiameter = 1 / plotter.kResolution

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

  def _ProcessAngle(self, t, previous_angle, start_angle, pen):
    half_step = (t + previous_angle) / 2
    diameter = self._GetSpiralDiameter(half_step)
    point = self._GetSpiralPosition(half_step, diameter, start_angle)

    image_position = self.GetImagePosition(point)

    if image_position is None:
      if t - self._last_angle_in_image > (2 * math.pi):
        return False
      if len(self._points) > 1:
        self._shapes.append(plotter.OpenPolyline(self._points, pen))
        self._points.clear()
      return True

    self._last_angle_in_image = t
    value = self._filter_image[int(image_position[1]),
                               int(image_position[0])]

    threshold = 200
    if (previous_angle is None or value >= threshold):
      self._points.append(plotter.Point(point))
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

  def DrawSpiral(self, pen, start_angle):
    self._points = deque()
    self._shapes = deque()
    self._last_angle_in_image = 0

    t = 0
    previous_angle = start_angle
    while self._ProcessAngle(t, previous_angle, start_angle, pen):
      previous_angle = t
      t += min(kMaxStepover / self._GetSpiralDiameter(t), kMaxAngle)
    return self._shapes

  def DoSwirls(self, image, pitch, num_wiggles, wiggle_radius,
      spiral_center=None):
    self._filter_image = self._InitializeImage(
        image, pitch + wiggle_radius, kMaxStepover)
    self._mult = 1

    if spiral_center is None or spiral_center == 'center':
      self._center = self._image_origin + (self._image_dim_plot / 2)
    elif spiral_center == 'corner':
      self._center = self._image_origin
    elif spiral_center == 'random':
      rand = np.array((random.random(), random.random()))
      self._center = self._image_origin + self._image_dim_plot * rand

    self._pitch = pitch
    self._num_wiggles = num_wiggles
    self._wiggle_radius = wiggle_radius

    return self.DrawSpiral(0, 0)


def main():
  # Read input image.
  source_name = sys.argv[1]
  image = cv2.imread(source_name)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  spiraler = Spiraler(kPageSize)
  shapes = spiraler.DoSwirls(
      image, kPitch, kWiggles, kWiggleRadius, kCenterPosition)

  pen_map = [
      pens.SARASA['black'],
      pens.SARASA['light blue'],
      pens.SARASA['pink'],
      pens.SARASA['light green'],
      pens.SARASA['mahogany'],
      pens.SARASA['cobalt'],
      pens.SARASA['kelly green'],
      pens.SARASA['violet'],
      ]
  if not plotter.ShowPreview(shapes, kPageSize, pen_map):
    return 0

  with open(sys.argv[2], 'wb') as dest:
    plotter.SortAllAndWrite(dest, shapes, 0.7, kPageSize, reorder=False)

if __name__ == "__main__":
  main()
