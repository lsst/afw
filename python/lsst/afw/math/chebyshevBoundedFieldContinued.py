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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from lsst.utils import continueClass
from .chebyshevBoundedField import ChebyshevBoundedField, ChebyshevBoundedFieldControl

__all__ = []  # import this module only for its side effects

@continueClass  # noqa: F811
class ChebyshevBoundedField:
    @classmethod
    def approxBoundedField(cls, boundedField,
                           orderX=3, orderY=3,
                           nStepX=100, nStepY=100):
        """
        Approximate a bounded field as a ChebyshevBoundedField.

        Parameters
        ----------
        boundedField : `lsst.afw.math.BoundedField`
            A bounded field to approximate
        orderX : `int`, optional
            Order of the Chebyshev polynomial in the x direction.
            Default is 3.
        orderY : `int`, optional
            Order of the Chebyshev polynomial in the y direction.
            Default is 3.
        nStepX : `int`, optional
            Number of x steps to approximate boundedField.
            Default is 100.
        nStepY : `int`, optional
            Number of y steps to approximate boundedField.
            Default is 100.

        Returns
        -------
        chebyshevBoundedField : `lsst.afw.math.ChebyshevBoundedField`
        """

        ctrl = ChebyshevBoundedFieldControl()
        ctrl.orderX = orderX
        ctrl.orderY = orderY
        ctrl.triangular = False

        bbox = boundedField.getBBox()

        xSteps = np.linspace(bbox.getMinX(), bbox.getMaxX(), nStepX)
        ySteps = np.linspace(bbox.getMinY(), bbox.getMaxY(), nStepY)

        x = np.tile(xSteps, nStepY)
        y = np.repeat(ySteps, nStepX)

        return cls.fit(bbox, x, y, boundedField.evaluate(x, y), ctrl)
