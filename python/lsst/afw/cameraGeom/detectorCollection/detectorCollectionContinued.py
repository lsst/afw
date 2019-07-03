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
#

__all__ = []

# import lsst.afw.cameraGeom as afwGeom
from lsst.utils import continueClass, TemplateMeta
from ..detector import Detector
from .detectorCollection import DetectorCollectionDetectorBase, DetectorCollectionBuilderBase


class DetectorCollectionBase(metaclass=TemplateMeta):  # noqa: F811
    """!An immutable collection of Detectors that can be accessed by name or ID
    """

    def __iter__(self):
        for k, v in sorted(self.getIdMap().items()):
            yield v

    def __getitem__(self, key):
        r = self.get(key)
        if r is None:
            raise KeyError("Detector for key {} not found.".format(key))
        return r

    def getNameIter(self):
        """Get an iterator over detector names
        """
        for k, v in self.getNameMap().items():
            yield k

    def getIdIter(self):
        """Get an iterator over detector IDs
        """
        for k, v in self.getIdMap().items():
            yield k


DetectorCollectionBase.register(Detector, DetectorCollectionDetectorBase)


@continueClass  # noqa: F811
class DetectorCollectionBuilderBase():

    def fromDict(self, inputDict, translationDict=None):
        if translationDict is not None:
            for key in translationDict.keys():
                if key in inputDict:
                    alias = translationDict[key]
                    inputDict[alias] = inputDict[key]

        self.setName(inputDict.get('name', "Undefined Camera"))
        #        self.setPlateScale(inputDict.get('plateScale', 1.0))
        #        self.setNativeSys(afwGeom.FOCAL_PLANE)  # This is fixed somewhere.
        #        self.setTransforms(inputDict('transformDict', None))
        #        import pdb
        #        pdb.set_trace()

        if 'CCDs' in inputDict:
            for name, ccd in inputDict['CCDs'].items():
                detBuilder = self.add(name, ccd['id'])
                detBuilder.fromDict(ccd)
