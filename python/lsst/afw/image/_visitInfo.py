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
"""This file only exists to deprecate the Filter and FilterProperty classes.
"""

from lsst.utils.deprecated import deprecate_pybind11
from lsst.utils import continueClass
from ._imageLib import VisitInfo


__all__ = []


@continueClass
class VisitInfo:  # noqa: F811

    def __deepcopy__(self, memo=None):
        return self

    def copyWith(
        self,
        exposureId=None,
        exposureTime=None,
        darkTime=None,
        date=None,
        ut1=None,
        era=None,
        boresightRaDec=None,
        boresightAzAlt=None,
        boresightAirmass=None,
        boresightRotAngle=None,
        rotType=None,
        observatory=None,
        weather=None,
        instrumentLabel=None,
        id=None,
        focusZ=None,
        observationType=None,
        scienceProgram=None,
        observationReason=None,
        object=None,
        hasSimulatedContent=None,
    ):
        if exposureId is None:
            # Note: exposureId is deprecated and `VisitInfo` no longer contains
            # an `exposureId` property, so we use the getter until
            # this is removed in DM-32138.
            exposureId = self.getExposureId()
        if exposureTime is None:
            exposureTime = self.exposureTime
        if darkTime is None:
            darkTime = self.darkTime
        if date is None:
            date = self.date
        if ut1 is None:
            ut1 = self.ut1
        if era is None:
            era = self.era
        if boresightRaDec is None:
            boresightRaDec = self.boresightRaDec
        if boresightAzAlt is None:
            boresightAzAlt = self.boresightAzAlt
        if boresightAirmass is None:
            boresightAirmass = self.boresightAirmass
        if boresightRotAngle is None:
            boresightRotAngle = self.boresightRotAngle
        if rotType is None:
            rotType = self.rotType
        if observatory is None:
            observatory = self.observatory
        if weather is None:
            weather = self.weather
        if instrumentLabel is None:
            instrumentLabel = self.instrumentLabel
        if id is None:
            id = self.id
        if focusZ is None:
            focusZ = self.focusZ
        if observationType is None:
            observationType = self.observationType
        if scienceProgram is None:
            scienceProgram = self.scienceProgram
        if observationReason is None:
            observationReason = self.observationReason
        if object is None:
            object = self.object
        if hasSimulatedContent is None:
            hasSimulatedContent = self.hasSimulatedContent

        return VisitInfo(
            exposureId=exposureId,
            exposureTime=exposureTime,
            darkTime=darkTime,
            date=date,
            ut1=ut1,
            era=era,
            boresightRaDec=boresightRaDec,
            boresightAzAlt=boresightAzAlt,
            boresightAirmass=boresightAirmass,
            boresightRotAngle=boresightRotAngle,
            rotType=rotType,
            observatory=observatory,
            weather=weather,
            instrumentLabel=instrumentLabel,
            id=id,
            focusZ=focusZ,
            observationType=observationType,
            scienceProgram=scienceProgram,
            observationReason=observationReason,
            object=object,
            hasSimulatedContent=hasSimulatedContent,
        )


VisitInfo.getExposureId = deprecate_pybind11(
    VisitInfo.getExposureId,
    reason="Replaced by VisitInfo.id for full focal plane identifiers and by ExposureInfo.id for "
           "detector-level identifiers. Will be removed after v25.",
    version="v24.0")
