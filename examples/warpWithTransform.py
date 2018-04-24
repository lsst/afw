#!/usr/bin/env python

import math

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath


def main():

    # a 31x31 postage stamp image
    nx, ny = 31, 31
    # move xy0 to simulate it being a shallow bbox'd sub-image
    xorig, yorig = 100, 300
    xy0 = afwGeom.Point2I(xorig, yorig)

    psfSigma = 3.0
    x0, y0 = xorig + nx/2, yorig + ny/2
    p0 = afwGeom.Point2D(x0, y0)

    img = afwImage.ImageF(nx, ny, 0)
    img.setXY0(xy0)
    for i in range(ny):
        for j in range(nx):
            ic = i - (y0 - yorig)
            jc = j - (x0 - xorig)
            r = math.sqrt(ic*ic + jc*jc)
            img.set(j, i, 1.0*math.exp(-r**2/(2.0*psfSigma**2)))

    # now warp it about the centroid using a linear transform

    linTran = afwGeom.LinearTransform().makeScaling(
        1.2)  # a simple scale-by-20%
    # extent a bit along x-dir
    linTran[0] *= 1.2

    wimg = afwImage.ImageF(nx, ny, 0)            # output 'warped' image
    wimg.setXY0(xy0)
    kernel = afwMath.LanczosWarpingKernel(5)     # warping kernel
    afwMath.warpCenteredImage(wimg, img, kernel, linTran, p0)

    img.writeFits("img.fits")
    wimg.writeFits("wimg.fits")


if __name__ == '__main__':
    main()
