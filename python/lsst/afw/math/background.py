# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import os
import sys
import lsst.daf.base as dafBase
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.fits import FitsError, MemFileManager, reduceToFits
import mathLib as afwMath

class BackgroundList(object):
    """A list-like class to contain a list of (afwMath.Background, interpStyle, undersampleStyle) tuples

In deference to the deprecated-but-not-yet-removed Background.getImage() API, we also accept a single
afwMath.Background and extract the interpStyle and undersampleStyle from the as-used values
    """
    
    def __init__(self, *args):
        self._backgrounds = []
        for a in args:
            self.append(a)

    def __getitem__(self, *args):
        """Return an item"""
        #
        # Set any previously-unknown Styles (they are set by bkgd.getImage())
        #
        for i, val in enumerate(self._backgrounds):
            bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder = val
            if interpStyle is None or undersampleStyle is None:
                interpStyle = bkgd.getAsUsedInterpStyle()
                undersampleStyle = bkgd.getAsUsedUndersampleStyle()
                actrl = bkgd.getBackgroundControl().getApproximateControl()
                approxStyle = actrl.getStyle()
                approxOrder = actrl.getOrderX()
                self._backgrounds[i] = (bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder)
        #
        # And return what they wanted
        #
        return self._backgrounds.__getitem__(*args)

    def __len__(self, *args):
        return self._backgrounds.__len__(*args)

    def append(self, val):
        try:
            bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder = val
        except TypeError:
            bkgd = val
            interpStyle = None
            undersampleStyle = None
            approxStyle = None
            approxOrder = None

        # Check to see if the Background is actually a BackgroundMI.
        # Such special treatment is not generally a good idea as it is against the whole idea of subclassing.
        # However, lsst.afw.math.makeBackground() returns a Background, even though it's really a BackgroundMI
        # under the covers.  Persistence requires that the type python sees is the actual type under the covers
        # (or it will call the wrong python class's python persistence methods).
        # The real solution is to not use makeBackground() in python but call the constructor directly;
        # however there is already code using makeBackground(), so this is an attempt to assist the user.
        subclassed = afwMath.cast_BackgroundMI(bkgd)
        if subclassed is not None:
            bkgd = subclassed
        else:
            from lsst.pex.logging import getDefaultLog
            getDefaultLog().warn("Unrecognised Background object %s may be unpersistable." % (bkgd,))

        self._backgrounds.append((bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder))

    def writeFits(self, fileName, flags=0):
        """Save our list of Backgrounds to a file
        @param fileName         FITS file to write
        @param flags            Flags to control details of writing; currently unused,
                                but present for consistency with
                                afw.table.BaseCatalog.writeFits.
        """

        for i, bkgd in enumerate(self):
            bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder = bkgd

            statsImage = bkgd.getStatsImage()

            md = dafBase.PropertyList()
            md.set("INTERPSTYLE", interpStyle)
            md.set("UNDERSAMPLESTYLE", undersampleStyle)
            md.set("APPROXSTYLE", approxStyle)
            md.set("APPROXORDER", approxOrder)
            bbox = bkgd.getImageBBox()
            md.set("BKGD_X0", bbox.getMinX())
            md.set("BKGD_Y0", bbox.getMinY())
            md.set("BKGD_WIDTH", bbox.getWidth())
            md.set("BKGD_HEIGHT", bbox.getHeight())

            statsImage.getImage().writeFits(   fileName, md, "w" if i == 0 else "a")
            statsImage.getMask().writeFits(    fileName, md, "a")
            statsImage.getVariance().writeFits(fileName, md, "a")

    @staticmethod
    def readFits(fileName, hdu=0, flags=0):
        """Read a our list of Backgrounds from a file
        @param fileName         FITS file to read
        @param hdu              First Header/Data Unit to attempt to read from
        @param flags            Flags to control details of reading; currently unused,
                                but present for consistency with
                                afw.table.BaseCatalog.readFits.

        See also getImage()
        """
        if not isinstance(fileName, MemFileManager) and not os.path.exists(fileName):
            raise RuntimeError("File not found: %s" % fileName)

        self = BackgroundList()

        while True:
            hdu += 1

            md = dafBase.PropertyList()
            try:
                img = afwImage.ImageF(fileName, hdu, md); hdu += 1
            except pexExcept.LsstCppException, e:
                if isinstance(e.args[0], FitsError):
                    break
                raise

            msk = afwImage.MaskU( fileName, hdu);     hdu += 1
            var = afwImage.ImageF(fileName, hdu)

            statsImage = afwImage.makeMaskedImage(img, msk, var)

            x0 = md.get("BKGD_X0")
            y0 = md.get("BKGD_Y0")
            width  = md.get("BKGD_WIDTH")
            height = md.get("BKGD_HEIGHT")
            imageBBox = afwGeom.BoxI(afwGeom.PointI(x0, y0), afwGeom.ExtentI(width, height))

            interpStyle =      md.get("INTERPSTYLE")
            undersampleStyle = md.get("UNDERSAMPLESTYLE")
            approxStyle = md.get("APPROXSTYLE") if "APPROXSTYLE" in md.names() \
                          else afwMath.ApproximateControl.UNKNOWN
            approxOrder = md.get("APPROXORDER") if "APPROXORDER" in md.names() else 1

            bkgd = afwMath.BackgroundMI(imageBBox, statsImage)
            actrl = afwMath.ApproximateControl(approxStyle, approxOrder)
            bkgd.getBackgroundControl().setApproximateControl(actrl)
            self.append((bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder))

        return self

    def getImage(self):
        """
        Compute and return a full-resolution image from our list of (Background, interpStyle, undersampleStyle)
        """

        bkgdImage = None
        for bkgd, interpStyle, undersampleStyle, approxStyle, approxOrder in self:
            if not bkgdImage:
                bkgdImage =  bkgd.getImageF(interpStyle, undersampleStyle)
            else:
                bkgdImage += bkgd.getImageF(interpStyle, undersampleStyle)

        return bkgdImage

    def __reduce__(self):
        return reduceToFits(self)
