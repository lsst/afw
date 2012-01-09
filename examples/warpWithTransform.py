#!/usr/bin/env python

import sys, math

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.cameraGeom as cameraGeom

def main(useDistort=True):

    nx, ny = 32, 32
    xorig, yorig = 100, 300
    xy0 = afwGeom.Point2I(xorig, yorig)
    wid = 31
    rad0 = 3.0
    x0, y0 = xorig+16, yorig+16
    p0 = afwGeom.Point2D(x0, y0)
    cx0, cy0 = 300.0, 500.0
    cp0 = afwGeom.Point2D(cx0, cy0)

    img = afwImage.ImageF(nx, ny, 0)
    img.setXY0(xy0)
    for i in range(ny):
        for j in range(nx):
            ic = i - (y0 - yorig)
            jc = j - (x0 - xorig)
            r = math.sqrt(ic*ic + jc*jc)
            img.set(j, i, 1.0*math.exp(-r**2/(2.0*rad0**2)))

    if useDistort: #False:
        # try the suprimecam numbers
        coeffs = [0.0, 1.0, 7.16417e-04, 3.03146e-10, 5.69338e-14, -6.61572e-18]
        dist = cameraGeom.RadialPolyDistortion(coeffs)
        linTran = dist.computeQuadrupoleTransform(p0, True)
        print linTran
        #dist = cameraGeom.Distortion()
        p2 = dist.distort(cp0)
        wimg = dist.distort(cp0, img, p0)
        uwimg = dist.undistort(cp0, wimg, p0)

        settings = {'scale': 'minmax', 'zoom':"to fit", 'mask':'transparency 80'}
        ds9.mtv(img, frame=1, title='img', settings=settings)
        ds9.mtv(wimg, frame=2, title='wimg', settings=settings)
        ds9.mtv(uwimg, frame=3, title='uwimg', settings=settings)
        
    else:
        wimg = afwImage.ImageF(nx, ny, 0)
        linTran = afwGeom.LinearTransform().makeScaling(1.2)
        linTran[0] *= 1.2
        #linTran[0] = 2.51975
        #linTran[1] = 1.51975
        #linTran[2] = 1.51975
        #linTran[3] = 2.51975
        print linTran
        kernel = afwMath.LanczosWarpingKernel(5)
        afwMath.warpCenteredImage(wimg, img, kernel, linTran, p0)

        
    img.writeFits("img.fits")
    wimg.writeFits("wimg.fits")

    
if __name__ == '__main__':
    useDistort = True
    if len(sys.argv) > 1:
        useDistort = int(sys.argv[1])
        if useDistort == 0:
            useDistort = False
    print useDistort
    main(useDistort)
    
