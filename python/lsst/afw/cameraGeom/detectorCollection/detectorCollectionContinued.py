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
import os.path
from lsst.utils import continueClass, TemplateMeta
from ..amplifier import Amplifier
from ..detector import Detector
from .detectorCollection import DetectorCollectionDetectorBase, DetectorCollectionBuilderBase
from ...table import BaseCatalog


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

        import pdb
        pdb.set_trace()
        if 'transformDict' in inputDict:
            setTransforms(self, inputDict['transformDict'])

        print("CZW: Define Camera to sky transforms.")
        if 'CCDs' in inputDict:
            for name, ccd in inputDict['CCDs'].items():
                detBuilder = self.add(name, ccd['id'])
                detBuilder.fromDict(ccd)
        elif 'detectorList' in inputDict:
            for detInd, ccd in inputDict['detectorList'].items():
                name = ccd['name']
                detBuilder = self.add(name, ccd['id'])
                detBuilder.fromDict(ccd)

        # We need to manually load amplifier data for each detector.
        if 'ampInfoPath' in inputDict:
            cameraNameMap = self.getNameMap()
            for detName, detector in cameraNameMap.items():
                detName.replace(" ", "_")  # GetShortCcdName
                ampCatPath = os.path.join(inputDict['ampInfoPath'], detName + ".fits")
                catalog = BaseCatalog.readFits(ampCatPath)
                for record in catalog:
                    amp = Amplifier.Builder.fromRecord(record)
                    detector.append(amp)

        # Transforms


def setTransforms(cameraBuilder, transformDict):
    from ..cameraGeomLib import FIELD_ANGLE, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS
    from lsst.afw.cameraGeom.transformConfig import transformDictFromYaml
    cameraSysList = [FIELD_ANGLE, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS]
    cameraSysMap = dict((sys.getSysName(), sys) for sys in cameraSysList)

    nativeSysName = transformDict['nativeSys']
    nativeSys = cameraSysMap.get(nativeSysName, FOCAL_PLANE)
    assert nativeSys == FOCAL_PLANE, "Received incorrect nativeSys"

    import pdb
    pdb.set_trace()
    print("CZW:")
    transformDict = transformDictFromYaml(transformDict.get('plateScale', 1.0), transformDict)
    focalPlaneToField = transformDict[FIELD_ANGLE]

    for toSys, transform in transformDict.items():
        cameraBuilder.setTransformFromFocalPlaneTo(toSys, transform)

    return focalPlaneToField
