#
# LSST Data Management System
# Copyright 2017 LSST/AURA.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import absolute_import, division, print_function

__all__ = ["PointKey", "CovarianceMatrixKey"]

from future.utils import with_metaclass
import numpy as np

from lsst.utils import TemplateMeta
from . import aggregates


class PointKey(with_metaclass(TemplateMeta, object)):
    TEMPLATE_PARAMS = ("dtype", "dim")
    TEMPLATE_DEFAULTS = (None, 2)


PointKey.register((np.int32, 2), aggregates.Point2IKey)
PointKey.register((np.float64, 2), aggregates.Point2DKey)


class CovarianceMatrixKey(with_metaclass(TemplateMeta, object)):
    TEMPLATE_PARAMS = ("dtype", "dim")


CovarianceMatrixKey.register((np.float32, 2), aggregates.CovarianceMatrix2fKey)
CovarianceMatrixKey.register((np.float32, 3), aggregates.CovarianceMatrix3fKey)
CovarianceMatrixKey.register((np.float32, 4), aggregates.CovarianceMatrix4fKey)
CovarianceMatrixKey.register((np.float32, None), aggregates.CovarianceMatrixXfKey)
CovarianceMatrixKey.register((np.float64, 2), aggregates.CovarianceMatrix2dKey)
CovarianceMatrixKey.register((np.float64, 3), aggregates.CovarianceMatrix3dKey)
CovarianceMatrixKey.register((np.float64, 4), aggregates.CovarianceMatrix4dKey)
CovarianceMatrixKey.register((np.float64, None), aggregates.CovarianceMatrixXdKey)
