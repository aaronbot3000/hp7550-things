#!/usr/bin/python3

# inspired by https://turtletoy.net/turtle/b2fdeed2d8

from collections import deque
import math
import random
import noise
import sys

import lib.program as program
import lib.pens as pens
import lib.plotter as plotter

kPaperType = 'letter'
kDirection = 'landscape'

kStepRes = 0.6 # mm
kScanRes = 0.4 # mm

kNoiseScanZoom = 3
kNoiseStepZoom = 5
kNoiseScale = 2048
kNoiseType = 'perlin'  # simplex or perlin
kOctaves = 2

# steppes and plateaus
#kNoiseMask = 0x3B0

# steppes
#kNoiseMask = 0x1B0

# plains and plateaus
kNoiseMask = 0x2F0

# tall walls
#kNoiseMask = 0x6B0

# less tall walls
#kNoiseMask = 0x6B0 >> 1

kFloor = 200
kFloorDecimator = 2


class LandGen(program.Program):
  def __init__(self, paper_type):
    super().__init__(paper_type)

  def AppendPoint(self, storage, scan, step):
    if kDirection == 'landscape':
      storage.append(plotter.Point(scan, step))
    else:
      storage.append(plotter.Point(step, scan))

  def NoiseFunc(self, x, y):
    if kNoiseType == 'simplex':
      return noise.snoise2(x, y, octaves=kOctaves)
    elif kNoiseType == 'perlin':
      return noise.pnoise2(x, y, octaves=kOctaves)
    else:
      assert False, 'unrecognized noise %s' % kNoiseType

  def Run(self):
    all_lines = deque()

    if kDirection == 'landscape':
      scan_limit = self._x_limit
      step_limit = self._y_limit
      scan_scale = self._x_limit / self._y_limit
      step_scale = 1
    elif kDirection == 'portrait':
      scan_limit = self._y_limit
      step_limit = self._x_limit
      scan_scale = 1
      step_scale = self._x_limit / self._y_limit
    else:
      assert False, 'unrecognized direction %s' % kDirection

    step_res = math.ceil(kStepRes / plotter.kResolution)
    scan_res = math.ceil(kScanRes / plotter.kResolution)
    scan_points = math.floor(scan_limit / scan_res)
    step_points = math.floor(step_limit / step_res)

    view_line = [0] * scan_points
    for step in range(-100, step_points):
      if step % 30 == 0:
        print('On step %d of %d.' % (step, step_points))
      pen_up = False
      points = deque()
      floor_points = deque()
      for scan in range(scan_points):
        # height is in points.
        base_height = step * step_res

        normal_scan_pos = scan / scan_points * scan_scale
        normal_step_pos = step / step_points * step_scale
        add_noise = self.NoiseFunc(normal_scan_pos * kNoiseScanZoom,
                                   normal_step_pos * kNoiseStepZoom)

        rise = int(kNoiseScale * add_noise) & kNoiseMask
        rise = max(kFloor, rise)

        height = base_height + rise

        prev = scan - 1

        if height >= step_limit:
          if scan > 0:
            if not points and view_line[prev] < step_limit:
              self.AppendPoint(points, prev * scan_res, view_line[prev])

          view_line[scan] = step_limit

          if points:
            self.AppendPoint(points, scan * scan_res, step_limit)
            all_lines.append(plotter.OpenPolyline(points, 0))
          points = deque()
          if floor_points:
            if len(floor_points) > 1:
              all_lines.append(plotter.OpenPolyline(floor_points, 1))
            floor_points = deque()
        elif height <= view_line[scan]:
          if not points and scan > 0:
            if view_line[prev] >= step_limit and view_line[scan] < step_limit:
              self.AppendPoint(points, prev * scan_res, step_limit)

          if points:
            self.AppendPoint(points, scan * scan_res, view_line[scan])
            all_lines.append(plotter.OpenPolyline(points, 0))
          points = deque()
          if floor_points:
            if len(floor_points) > 1:
              all_lines.append(plotter.OpenPolyline(floor_points, 1))
            floor_points = deque()
        else:
          if rise == kFloor:
            if points:
              self.AppendPoint(points, prev * scan_res, height)
              all_lines.append(plotter.OpenPolyline(points, 0))
              points = deque()
            if (step % kFloorDecimator) == 0:
              self.AppendPoint(floor_points, prev * scan_res, height)
          else:
            if floor_points:
              if len(floor_points) > 1:
                all_lines.append(plotter.OpenPolyline(floor_points, 1))
              floor_points = deque()

            if not points and scan > 0:
              self.AppendPoint(points, prev * scan_res, view_line[prev])
            self.AppendPoint(points, scan * scan_res, height)
          view_line[scan] = height

      if len(floor_points) > 1:
        all_lines.append(plotter.OpenPolyline(floor_points, 1))

      if len(points) > 1:
        all_lines.append(plotter.OpenPolyline(points, 0))
    return all_lines

def main():
  land_gen = LandGen(kPaperType)
  shapes = land_gen.Run()

  print('Line segments: %d' % len(shapes))

  pen_map = [pens.SARASA['forest green'], pens.SARASA['light blue'], ]
  if not plotter.ShowPreview(shapes, kPaperType, pen_map,
      override_line_width=2):
    return 0

  with open(sys.argv[1], 'wb') as dest:
    plotter.SortAllAndWrite(dest, shapes, 0.5, kPaperType)

if __name__ == "__main__":
  main()
