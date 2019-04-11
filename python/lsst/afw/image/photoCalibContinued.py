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
"""This file only exists to provide deprecation warnings for the deprecated
Calib-style interface.
"""

__all__ = []  # import this module only for its side effects

from lsst.utils import continueClass

from deprecated.sphinx import deprecated

from .photoCalib import PhotoCalib


@continueClass  # noqa: F811
class PhotoCalib:
    @staticmethod
    @deprecated("No-op: PhotoCalib never throws on negative instFlux (will be removed after v18).",
                category=FutureWarning)
    def setThrowOnNegativeFlux(raiseException):
        PhotoCalib._setThrowOnNegativeFlux(raiseException)

    @staticmethod
    @deprecated("No-op: PhotoCalib never throws on negative instFlux (will be removed after v18).",
                category=FutureWarning)
    def getThrowOnNegativeFlux():
        return PhotoCalib._getThrowOnNegativeFlux()

    @deprecated("For backwards compatibility with Calib; use `instFluxToMagnitude` instead"
                " (will be removed after v18).", category=FutureWarning)
    def getMagnitude(self, *args, **kwargs):
        return self._getMagnitude(*args, **kwargs)

    @deprecated("For backwards compatibility with Calib; use `magnitudeToInstFlux` instead"
                " (will be removed after v18).", category=FutureWarning)
    def getFlux(self, *args, **kwargs):
        return self._getFlux(*args, **kwargs)

    @deprecated("For backwards compatibility with Calib: use `getCalibrationMean`, `getCalibrationErr`, or "
                "`getInstFluxAtZeroMagnitude` (will be removed after v18).", category=FutureWarning)
    def getFluxMag0(self):
        return self._getFluxMag0()
