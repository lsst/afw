#!/usr/bin/env python

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

#
from __future__ import absolute_import, division, print_function
from builtins import str
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.display.ds9 as ds9

# This code was submitted as a part of ticket #749 to demonstrate
# the failure of Statistics in dealing with NaN

disp = False


def main():

    gaussFunction = afwMath.GaussianFunction2D(3, 2, 0.5)
    gaussKernel = afwMath.AnalyticKernel(10, 10, gaussFunction)
    inImage = afwImage.ImageF(afwGeom.Extent2I(100, 100))
    inImage.set(1)
    if disp:
        ds9.mtv(inImage, frame = 0)

    # works
    outImage = afwImage.ImageF(afwGeom.Extent2I(100, 100))
    afwMath.convolve(outImage, inImage, gaussKernel, False, True)
    if disp:
        ds9.mtv(outImage, frame = 1)
    print("Should be a number: ", afwMath.makeStatistics(
        outImage, afwMath.MEAN).getValue())
    print("Should be a number: ", afwMath.makeStatistics(
        outImage, afwMath.STDEV).getValue())

    # not works ... now does work
    outImage = afwImage.ImageF(afwGeom.Extent2I(100, 100))
    afwMath.convolve(outImage, inImage, gaussKernel, False, False)
    if disp:
        ds9.mtv(outImage, frame = 2)
    print("Should be a number: ", afwMath.makeStatistics(
        outImage, afwMath.MEAN).getValue())
    print("Should be a number: ", afwMath.makeStatistics(
        outImage, afwMath.STDEV).getValue())

    # This will print nan
    sctrl = afwMath.StatisticsControl()
    sctrl.setNanSafe(False)
    print ("Should be a nan (nanSafe set to False): " +
           str(afwMath.makeStatistics(outImage, afwMath.MEAN, sctrl).getValue()))


if __name__ == '__main__':
    main()
