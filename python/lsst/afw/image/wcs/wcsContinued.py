#
# LSST Data Management System
# Copyright 2008-2017 LSST/AURA.
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

__all__ = ["unpickleWcs"]

from lsst.utils import continueClass

from ..exposure import ExposureU
from .wcs import Wcs


def unpickleWcs(pick):
    import pickle
    exposure = pickle.loads(pick)
    return exposure.getWcs()


@continueClass
class Wcs:

    def __reduce__(self):
        import pickle
        exposure = ExposureU(1, 1)
        exposure.setWcs(self)
        return (unpickleWcs, (pickle.dumps(exposure),))

