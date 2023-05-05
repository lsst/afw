# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

from lsst.utils import continueClass

from lsst.geom import Point2I
from ._detection import Footprint
from ..image import Image

__all__ = []  # import this module only for its side effects


@continueClass
class Footprint:  # noqa: F811
    def extractFluxFromArray(self, image: np.ndarray, xy0: Point2I = Point2I()) -> float:
        """

        Parameters
        ----------
        image:
            The array containing the image pixels to extract
        xy0:
            The origin of the image array.

        Returns
        flux:
            The flux from the image in pixels contained in the footprint
        """
        return self.spans.flatten(image, xy0).sum()

    def extractFluxFromImage(self, image: Image) -> float:
        """Calculate the total flux in the region of an Image contained in
        this Footprint.

        Parameters
        ----------
        image:
            The image to extract.

        Returns
        -------
        flux:
            The flux from the image in the pixels contained in the footprint.
        """
        return self.extractFluxFromArray(image.array, image.getBBox().getMin())
