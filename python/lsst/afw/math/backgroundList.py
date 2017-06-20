from __future__ import absolute_import, division, print_function
from builtins import object
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
import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.fits import FitsError, MemFileManager, reduceToFits
from . import mathLib as afwMath


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
            bkgd, interpStyle, undersampleStyle, approxStyle, \
                approxOrderX, approxOrderY, approxWeighting = val
            if interpStyle is None or undersampleStyle is None:
                interpStyle = bkgd.getAsUsedInterpStyle()
                undersampleStyle = bkgd.getAsUsedUndersampleStyle()
                actrl = bkgd.getBackgroundControl().getApproximateControl()
                approxStyle = actrl.getStyle()
                approxOrderX = actrl.getOrderX()
                approxOrderY = actrl.getOrderY()
                approxWeighting = actrl.getWeighting()
                self._backgrounds[i] = (bkgd, interpStyle, undersampleStyle,
                                        approxStyle, approxOrderX, approxOrderY, approxWeighting)
        #
        # And return what they wanted
        #
        return self._backgrounds.__getitem__(*args)

    def __len__(self, *args):
        return self._backgrounds.__len__(*args)

    def append(self, val):
        try:
            bkgd, interpStyle, undersampleStyle, approxStyle, \
                approxOrderX, approxOrderY, approxWeighting = val
        except TypeError:
            bkgd = val
            interpStyle = None
            undersampleStyle = None
            approxStyle = None
            approxOrderX = None
            approxOrderY = None
            approxWeighting = None

        bgInfo = (bkgd, interpStyle, undersampleStyle, approxStyle,
                  approxOrderX, approxOrderY, approxWeighting)
        self._backgrounds.append(bgInfo)

    def clone(self):
        """Return a shallow copy

        Shallow copies do not share backgrounds that are appended after copying,
        but do share changes to contained background objects.
        """
        return BackgroundList(*self)

    def writeFits(self, fileName, flags=0):
        """Save our list of Backgrounds to a file
        @param fileName         FITS file to write
        @param flags            Flags to control details of writing; currently unused,
                                but present for consistency with
                                afw.table.BaseCatalog.writeFits.
        """

        for i, bkgd in enumerate(self):
            (bkgd, interpStyle, undersampleStyle, approxStyle, approxOrderX, approxOrderY,
             approxWeighting) = bkgd

            statsImage = bkgd.getStatsImage()

            md = dafBase.PropertyList()
            md.set("INTERPSTYLE", int(interpStyle))
            md.set("UNDERSAMPLESTYLE", int(undersampleStyle))
            md.set("APPROXSTYLE", int(approxStyle))
            md.set("APPROXORDERX", approxOrderX)
            md.set("APPROXORDERY", approxOrderY)
            md.set("APPROXWEIGHTING", approxWeighting)
            bbox = bkgd.getImageBBox()
            md.set("BKGD_X0", bbox.getMinX())
            md.set("BKGD_Y0", bbox.getMinY())
            md.set("BKGD_WIDTH", bbox.getWidth())
            md.set("BKGD_HEIGHT", bbox.getHeight())

            statsImage.getImage().writeFits(fileName, md, "w" if i == 0 else "a")
            statsImage.getMask().writeFits(fileName, md, "a")
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

        INT_MIN = -(1 << 31)
        if hdu == INT_MIN:
            hdu = -1
        else:
            # we want to start at 0 (post RFC-304), but are about to increment
            hdu -= 1

        while True:
            hdu += 1

            md = dafBase.PropertyList()
            try:
                img = afwImage.ImageF(fileName, hdu, md)
                hdu += 1
            except FitsError:
                break

            msk = afwImage.Mask(fileName, hdu)
            hdu += 1
            var = afwImage.ImageF(fileName, hdu)

            statsImage = afwImage.makeMaskedImage(img, msk, var)

            x0 = md.get("BKGD_X0")
            y0 = md.get("BKGD_Y0")
            width = md.get("BKGD_WIDTH")
            height = md.get("BKGD_HEIGHT")
            imageBBox = afwGeom.BoxI(afwGeom.PointI(
                x0, y0), afwGeom.ExtentI(width, height))

            interpStyle = afwMath.Interpolate.Style(md.get("INTERPSTYLE"))
            undersampleStyle = afwMath.UndersampleStyle(
                md.get("UNDERSAMPLESTYLE"))

            # Older outputs won't have APPROX* settings.  Provide alternative defaults.
            # Note: Currently X- and Y-orders must be equal due to a limitation in
            #       math::Chebyshev1Function2.  Setting approxOrderY = -1 is equivalent
            #       to saying approxOrderY = approxOrderX.
            approxStyle = md.get("APPROXSTYLE") if "APPROXSTYLE" in md.names() \
                else afwMath.ApproximateControl.UNKNOWN
            approxStyle = afwMath.ApproximateControl.Style(approxStyle)
            approxOrderX = md.get(
                "APPROXORDERX") if "APPROXORDERX" in md.names() else 1
            approxOrderY = md.get(
                "APPROXORDERY") if "APPROXORDERY" in md.names() else -1
            approxWeighting = md.get(
                "APPROXWEIGHTING") if "APPROXWEIGHTING" in md.names() else True

            bkgd = afwMath.BackgroundMI(imageBBox, statsImage)
            bctrl = bkgd.getBackgroundControl()
            bctrl.setInterpStyle(interpStyle)
            bctrl.setUndersampleStyle(undersampleStyle)
            actrl = afwMath.ApproximateControl(
                approxStyle, approxOrderX, approxOrderY, approxWeighting)
            bctrl.setApproximateControl(actrl)
            bgInfo = (bkgd, interpStyle, undersampleStyle, approxStyle,
                      approxOrderX, approxOrderY, approxWeighting)
            self.append(bgInfo)

        return self

    def getImage(self):
        """
        Compute and return a full-resolution image from our list of
        (Background, interpStyle, undersampleStyle)
        """

        bkgdImage = None
        for (bkgd, interpStyle, undersampleStyle, approxStyle,
             approxOrderX, approxOrderY, approxWeighting) in self:
            if not bkgdImage:
                if approxStyle != afwMath.ApproximateControl.UNKNOWN:
                    bkgdImage = bkgd.getImageF()
                else:
                    bkgdImage = bkgd.getImageF(interpStyle, undersampleStyle)
            else:
                if approxStyle != afwMath.ApproximateControl.UNKNOWN:
                    bkgdImage += bkgd.getImageF()
                else:
                    bkgdImage += bkgd.getImageF(interpStyle, undersampleStyle)

        return bkgdImage

    def __reduce__(self):
        return reduceToFits(self)
