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

__all__ = []  # import this module only for its side effects

from deprecated.sphinx import deprecated

from lsst.utils import continueClass

from ..base import Catalog
from .exposure import ExposureCatalog, ExposureRecord


@continueClass  # noqa F811
class ExposureRecord:
    @deprecated(reason="Replaced with getPhotoCalib (will be removed after v18)", category=FutureWarning)
    def getCalib(self, *args, **kwargs):
        return self._getCalib(*args, **kwargs)

    @deprecated(reason="Replaced with setPhotoCalib (will be removed after v18)", category=FutureWarning)
    def setCalib(self, *args, **kwargs):
        self._setCalib(*args, **kwargs)


Catalog.register("Exposure", ExposureCatalog)
