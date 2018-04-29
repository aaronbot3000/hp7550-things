import cv2

import plotter_lib
import numpy as np

class Program(object):
  def __init__(self, paper_type):
    self._paper_type = paper_type

  def _InitializeImage(self, image, margin, blur):
    """Common image processing routine.

    Provides the following class members:
      self._image_to_plot: Scale factor to go from image coordinates to plotter
        coordinates.
      
      self._plot_to_image: Scale factor to go from plotter coordinates to image
        coordinates.

      self._image_origin: The centered image's origin on the paper in plotter
        coordinates.

      self._image_dim: The dimensions of the image in image coordinates.

      self._image_dim_plot: The dimensions of the image in plotter coordinates.

    Rotates and flips the image to align the image's x and y axis to the
    plotter x and y axis.

    Args:
      image: the input image as a numpy array.
      margin: how much margin to leave around the image, in plotter coordinates.
      blur: the window size of the blur filter in plotter coordinates. 

    Returns:
      A blurred, rotation and flip corrected version of the image.
    """

    image = np.rot90(image, 2)
    image = np.flip(image, 1)
    self._image_dim = np.array((len(image[0]), len(image)), np.int32)

    if self._paper_type == 'tabloid':
      x_limit = plotter_lib.kTabloidX
      y_limit = plotter_lib.kTabloidY
    else:
      x_limit = plotter_lib.kLetterX
      y_limit = plotter_lib.kLetterY

    # Center and get image scaling.
    self._image_to_plot = min((x_limit - margin) / self._image_dim[0],
                              (y_limit - margin) / self._image_dim[1])
    self._plot_to_image = 1 / self._image_to_plot
    self._image_origin = ((np.array((x_limit, y_limit), np.int32) -
                          self._image_dim * self._image_to_plot) / 2)

    self._image_dim_plot = self._image_dim * self._image_to_plot

    # Blur the image.
    line_size_in_image = int(blur * self._plot_to_image)
    if line_size_in_image % 2 == 0:
      line_size_in_image += 1
    return cv2.blur(image, (line_size_in_image, line_size_in_image))

  def GetImagePosition(self, position):
    image_position = np.round((position - self._image_origin) *
        self._plot_to_image, 0)
    if not (0 <= image_position[0] < self._image_dim[0] and
            0 <= image_position[1] < self._image_dim[1]):
      return None
    return image_position

