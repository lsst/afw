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

__all__ = []  # import this module only for its side effects

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lsst.afw.image import Image

import numpy as np

from lsst.utils import continueClass

from lsst.geom import Point2I
from ._detection import Footprint


@continueClass
class Footprint:  # noqa: F811
    def computeFluxFromArray(self, image: np.ndarray, xy0: Point2I) -> float:
        """Calculate the total flux in the region of an image array
        contained in this Footprint.

        Parameters
        ----------
        image:
            Array containing the pixels to extract.
        xy0:
            The origin of the image array.

        Returns
        flux:
            Flux from the image in pixels contained in the footprint.
        """
        return self.spans.flatten(image, xy0).sum()

    def computeFluxFromImage(self, image: "Image") -> float:
        """Calculate the total flux in the region of an Image contained in
        this Footprint.

        Parameters
        ----------
        image:
            Image to extract.

        Returns
        -------
        flux:
            Flux from the image pixels contained in the footprint.
        """
        return self.computeFluxFromArray(image.array, image.getXY0())
