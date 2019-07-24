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

__all__ = ["ReadoutCornerValNameDict", "ReadoutCornerNameValDict"]

from lsst.utils import continueClass
import lsst.geom
from .amplifier import Amplifier, ReadoutCorner, AssemblyState


ReadoutCornerValNameDict = {
    ReadoutCorner.LL: "LL",
    ReadoutCorner.LR: "LR",
    ReadoutCorner.UR: "UR",
    ReadoutCorner.UL: "UL",
}
ReadoutCornerNameValDict = {val: key for key, val in
                            ReadoutCornerValNameDict.items()}


@continueClass  # noqa: F811
class Amplifier:
    def getDataBBox(self):
        """Return a consistent bounding box accessor for the science data region.

        Returns
        -------
        dataBBox : `lsst.geom.Box2I`
            Either Amplifier.bbox or Amplifier.rawDataBBox, based on the current AssemblyState.

        Raises
        ------
        RuntimeError :
            Raised if the Assembly State is not listed/is not handled.
        """
        myState = self.getAssemblyState()
        if myState == AssemblyState.SPLIT:
            return self.getRawDataBBox()
        elif myState == AssemblyState.RAW:
            bbox = self.getRawDataBBox()
            bbox.shift(self.getRawXYOffset())
            return bbox
        elif myState == AssemblyState.ENGINEERING:
            bbox = self.getRawDataBBox()
            bbox.shift(self.getRawXYOffset())
            return bbox
        elif myState == AssemblyState.SCIENCE:
            return self.getBBox()
        else:
            raise RuntimeError("Unknown Amplifier.AssemblyState: %s" % (myState))

    def getEngineeringBBox(self):
        """Return a consistent bounding box accessor for the science data
        region, rotated such that the ReadoutCorner = LL.

        CZW: This may not yield something usable. :(  It's possible we'll want
             to return a image view that has been properly indexed to be useful,
             not a BBox at all.

        Returns
        -------
        dataBBox : `lsst.geom.Box2I`
            Appropriate science data bounding box, flipped to have a
            consistent ReadoutCorner value.

        Raises
        ------
        RuntimeError :
            Raised if the ReadoutCorner is not listed/is not handled.
        """
        myCorner = self.getReadoutCorner()
        dataBBox = self.getDataBBox()
        if myCorner == ReadoutCorner.LL:
            return dataBBox
        elif myCorner == ReadoutCorner.LR:
            return dataBBox.flipLR(self.getRawBBox()[0])
        elif myCorner == ReadoutCorner.UL:
            return dataBBox.flipTB(self.getRawBBox()[1])
        elif myCorner == ReadoutCorner.UR:
            return dataBBox.flipLR(self.getRawBBox()[0]).flipTB(self.getRawBBox()[1])
        else:
            raise RuntimeError("Unknown Amplifier.ReadoutCorner: %s" % (myCorner))

    def fromDict(self, inputDict, translationDict=None):
        if translationDict is not None:
            for key in translationDict.keys():
                if key in inputDict:
                    alias = translationDict[key]
                    inputDict[alias] = inputDict[key]

        self.setName(inputDict.get('name', "Undefined Amplifier"))
        self.setGain(inputDict.get('gain', 1.0))
        self.setReadNoise(inputDict.get('readNoise', 0.0))
        self.setSaturation(inputDict.get('saturation', float('nan')))
        self.setSuspectLevel(inputDict.get('suspect', float('nan')))
        self.setReadoutCorner(ReadoutCornerNameValDict.get(inputDict.get('readCorner', 0)))

        # Linearity is a special case
        print("PRE", self.getLinearityType(), self.getLinearityCoeffs())
        if 'linearityCoeffs' in inputDict.keys():
            self.setLinearityCoeffs([float(val) for val in inputDict['linearityCoeffs']])
        self.setLinearityType(inputDict.get('linearityType', "PROPORTIONAL"))
        self.setLinearityThreshold(inputDict.get('linearityThreshold', 0.0))
        self.setLinearityMaximum(inputDict.get('linearityMax', self.getSaturation()))
        self.setLinearityUnits("DN")  # This likely never will be set.
        if self.getLinearityType() == "PROPORTIONAL":
            self.setLinearityCoeffs([float(self.getLinearityThreshold()),
                                     float(self.getLinearityMaximum()),
                                     float('nan'), float('nan')])
        print("POST", self.getLinearityType(), self.getLinearityCoeffs())

        # Set up geometries
        ix, iy = inputDict.get('ixy', (0, 0))
        flipX, flipY = inputDict.get('flipXY', (False, False))
        xRawExtent, yRawExtent = self.getRawBBox().getDimensions()

        perAmpData = inputDict.get('perAmpData', True)
        if perAmpData:
            x0, y0 = 0, 0
        else:
            x0, y0 = ix*xRawExtent, iy*yRawExtent

        #        self.setHasRawInfo(True)
        self.setRawFlipX(flipX)
        self.setRawFlipY(flipY)
        self.setRawXYOffset(lsst.geom.Extent2I(x0, y0))

        # BBoxes: These are passed from yaml as ((x0, y0), (x_extent, y_extent))
        #        self.setBBox(makeBBoxFromList(inputDict.get('bbox', None), x0, y0))
        self.setRawBBox(makeBBoxFromList(inputDict.get('rawBBox', None), x0, y0))
        self.setRawDataBBox(makeBBoxFromList(inputDict.get('rawDataBBox', None), x0, y0))

        # Overscan should go to the edge of the detector, so if MAX(OS) != MAX(AMP) make it so.
        self.setRawHorizontalOverscanBBox(
            makeBBoxFromList(inputDict.get('rawSerialOverscanBBox', None), x0, y0))
        self.setRawVerticalOverscanBBox(
            makeBBoxFromList(inputDict.get('rawParallelOverscanBBox', None), x0, y0))
        self.setRawHorizontalPrescanBBox(
            makeBBoxFromList(inputDict.get('rawSerialPrescanBBox', None), x0, y0))
        self.setRawVerticalPrescanBBox(
            makeBBoxFromList(inputDict.get('rawParallelOverscanBBox', None), x0, y0))

        # Update boxes
        self.setBBox(lsst.geom.BoxI(lsst.geom.PointI(x0, y0), self.getRawDataBBox().getDimensions()))

        x_extent = (self.getRawHorizontalPrescanBBox().getDimensions()[0] +
                    self.getRawDataBBox().getDimensions()[0] +
                    self.getRawHorizontalOverscanBBox().getDimensions()[0])
        y_extent = (self.getRawHorizontalPrescanBBox().getDimensions()[1] +
                    self.getRawDataBBox().getDimensions()[1] +
                    self.getRawHorizontalOverscanBBox().getDimensions()[1])
        if x_extent < self.getRawBBox().getDimensions()[0]:
            dx = self.getRawBBox().getDimensions()[0] - x_extent
            self.setRawHorizontalOverscanBBox(lsst.geom.BoxI(
                self.getRawHorizontalOverscanBBox().getMin(),
                self.getRawHorizontalOverscanBBox().getDimensions() +
                lsst.geom.ExtentI(dx, 0)))
        if y_extent < self.getRawBBox().getDimensions()[1]:
            dy = self.getRawBBox().getDimensions()[1] - y_extent
            self.setRawVerticalOverscanBBox(lsst.geom.BoxI(self.getRawVerticalOverscanBBox().getMin(),
                                                           self.getRawVerticalOverscanBBox().getDimensions() +
                                                           lsst.geom.ExtentI(0, dy)))

    def toDict(self):
        configDict = dict()

        configDict['name'] = self.getName()
        configDict['gain'] = self.getGain()
        configDict['readNoise'] = self.getReadNoise()
        configDict['saturation'] = self.getSaturation()
        configDict['suspectLevel'] = self.getSuspectLevel()
        configDict['readoutCorner'] = self.getReadoutCorner()
        configDict['linearityCoeffs'] = self.getLinearityCoeffs()
        configDict['linearityThreshold'] = self.getLinearityThreshold()
        configDict['linearityMax'] = self.getLinearityMaximum()
        configDict['linearityUnits'] = self.getLinearityUnits()

        configDict['bbox'] = self.getBBox()
        configDict['rawBBox'] = self.getRawBBox()
        configDict['rawDataBBox'] = self.getRawDataBBox()
        configDict['rawHorizontalOverscanBBox'] = self.getRawHorizontalOverscanBBox()
        configDict['rawVerticalOverscanBBox'] = self.getRawVerticalOverscanBBox()
        configDict['rawHorizontalPrescanBBox'] = self.getRawHorizontalPrescanBBox()
        configDict['rawVerticalPrescanBBox'] = self.getRawVerticalPrescanBBox()

        # configDict[perAmpData] = self.get
        configDict['raw_flip_x'] = self.getRawFlipX()
        configDict['raw_flip_y'] = self.getRawFlipY()
        configDict['raw_xyoffset'] = self.getRawXYOffset()
        configDict['assembly_state'] = self.getAssemblyState()

        return configDict


def makeBBoxFromList(inlist, x0=0, y0=0):
    """Given a list [(x0, y0), (xsize, ysize)], probably from a yaml file,
    return a BoxI.
    """
    if inlist is None:
        return lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.ExtentI(0, 0))
    else:
        (xs, ys), (xsize, ysize) = inlist
        box = lsst.geom.BoxI(lsst.geom.PointI(xs, ys), lsst.geom.ExtentI(xsize, ysize))
        box.shift(lsst.geom.Extent2I(x0, y0))
        return box
