#!/usr/bin/env python

#
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
#

import os
import lsst.utils
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay

try:
    display
except NameError:
    display = not False

afwDisplay.setDefaultMaskTransparency(75)

################################################


def getImage():
    imagePath = os.path.join(lsst.utils.getPackageDir("afwdata"),
                             "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci.fits")
    return afwImage.MaskedImageF(imagePath)


def simpleBackground(image):
    binsize = 128
    nx = int(image.getWidth()/binsize) + 1
    ny = int(image.getHeight()/binsize) + 1
    bctrl = afwMath.BackgroundControl(nx, ny)

    bkgd = afwMath.makeBackground(image, bctrl)

    image -= bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE)

    return bkgd


def complexBackground(image):
    MaskPixel = afwImage.MaskPixel
    binsize = 128
    nx = int(image.getWidth()/binsize) + 1
    ny = int(image.getHeight()/binsize) + 1

    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(3)
    sctrl.setNumIter(4)
    sctrl.setAndMask(afwImage.Mask[MaskPixel].getPlaneBitMask(["INTRP",
                                                               "EDGE"]))
    sctrl.setNoGoodPixelsMask(afwImage.Mask[MaskPixel].getPlaneBitMask("BAD"))
    sctrl.setNanSafe(True)
    if False:
        sctrl.setWeighted(True)
        sctrl.setCalcErrorFromInputVariance(True)

    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

    bkgd = afwMath.makeBackground(image, bctrl)

    statsImage = bkgd.getStatsImage()
    afwDisplay.Display(frame=3).mtv(statsImage.getVariance(), title="statsImage Variance")

    return bkgd


def main():
    image = getImage()

    if display:
        afwDisplay.Display(frame=0).mtv(image, title="Image")

    bkgd = simpleBackground(image)
    image = getImage()
    bkgd = complexBackground(image)

    if display:
        afwDisplay.Display(frame=1).mtv(image, title="image")
        afwDisplay.Display(frame=2).mtv(bkgd.getStatsImage(), title="background")

    order = 2
    actrl = afwMath.ApproximateControl(
        afwMath.ApproximateControl.CHEBYSHEV, order, order)
    approx = bkgd.getApproximate(actrl)

    approx.getImage()
    approx.getMaskedImage()
    approx.getImage(order - 1)


#################################################
if __name__ == '__main__':
    main()
