#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

# -*- python -*-
#
# simpleStacker.py
# Steve Bickerton
# An example executible which calls the example 'stack' code 
#
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

######################################
# main body of code
######################################
def main():

    nImg = 10
    nX, nY = 64, 64
    
    imgList = afwImage.vectorImageF()
    for iImg in range(nImg):
        imgList.push_back(afwImage.ImageF(afwGeom.Extent2I(nX, nY), iImg))

    imgStack = afwMath.statisticsStack(imgList, afwMath.MEAN)

    print imgStack.get(nX/2, nY/2)

    
#######################################
if __name__ == '__main__':
    main()
