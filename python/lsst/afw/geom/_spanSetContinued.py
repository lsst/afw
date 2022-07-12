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
import numpy as np

from . import SpanSet
from lsst.utils import continueClass

__all__ = []


@continueClass
class SpanSet:
    def asArray(self, shape=None, xy0=None):
        """Convert a SpanSet into a numpy boolean array

        Parameters
        ----------
        shape : `tuple` of `int`
            The final shape of the output array.
            If `shape` is `None` then the extent of the bounding box is used.
        xy0 : `~lsst.geom.Box2I` or `tuple` of `int`
            The lower-left corner of the array that will contain the spans.
            If `xy0` is `None` then the origin of the bounding box is used.

        Returns
        -------
        result : `numpy.ndarray`
            The array with pixels contained in `spans` marked as `True`.
        """
        # prevent circular import
        from lsst.afw.image import Mask

        if shape is None and xy0 is None:
            # It's slightly faster to set the array with the Mask instead of
            # shifting the spans.
            bbox = self.getBBox()
            mask = Mask(bbox)
            self.setMask(mask, 1)
            result = mask.getArray().astype(bool)
        else:
            if shape is None:
                # Use the shape of the full SpanSet
                extent = self.getBBox().getDimensions()
                shape = extent[1], extent[0]

            if xy0 is None:
                xy0 = self.getBBox().getMin()
            offset = (-xy0[0], -xy0[1])

            result = np.zeros(shape, dtype=bool)
            yidx, xidx = self.spans.shiftedBy(*offset).indices()
            result[yidx, xidx] = 1
        return result
