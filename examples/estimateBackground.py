#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

import os
import lsst.utils
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9

try:
    display
except NameError:
    display = not False

################################################

def getImage():
    imagePath = os.path.join(lsst.utils.getPackageDir("afwdata"),
                             "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci_img.fits")
    return afwImage.MaskedImageF(imagePath)

def simpleBackground(image):
    binsize   = 128
    nx = int(image.getWidth()/binsize) + 1
    ny = int(image.getHeight()/binsize) + 1
    bctrl = afwMath.BackgroundControl(nx, ny)

    bkgd = afwMath.makeBackground(image, bctrl)

    statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
    
    image  -= bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE)

    return bkgd

def complexBackground(image):
    binsize   = 128
    nx = int(image.getWidth()/binsize) + 1
    ny = int(image.getHeight()/binsize) + 1

    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(3)
    sctrl.setNumIter(4)
    sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(["INTRP", "EDGE"]))
    sctrl.setNoGoodPixelsMask(afwImage.MaskU.getPlaneBitMask("BAD"))
    sctrl.setNanSafe(True)
    if False:
        sctrl.setWeighted(True)
        sctrl.setCalcErrorFromInputVariance(True)

    bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)

    bkgd = afwMath.makeBackground(image, bctrl)

    statsImage = afwMath.cast_BackgroundMI(bkgd).getStatsImage()
    ds9.mtv(statsImage.getVariance())

    bkdgImages = dict(SPLINE = bkgd.getImageF(afwMath.Interpolate.NATURAL_SPLINE),
                      LINEAR = bkgd.getImageF(afwMath.Interpolate.LINEAR))

    return bkgd

def main():
    image = getImage()

    if display:
        ds9.mtv(image, frame=0)

    bkgd = simpleBackground(image)
    image = getImage()
    bkgd = complexBackground(image)

    if display:
        ds9.mtv(image, frame=1)
        ds9.mtv(afwMath.cast_BackgroundMI(bkgd).getStatsImage(), frame=2)

    order = 2
    actrl = afwMath.ApproximateControl(afwMath.ApproximateControl.CHEBYSHEV, order, order)
    approx = bkgd.getApproximate(actrl)

    approx.getImage()
    approx.getMaskedImage()
    approx.getImage(order - 1)

#################################################
if __name__ == '__main__':
    main()
