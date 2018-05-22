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
import lsst.geom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

# This code was written as a result of ticket #1090 to
# demonstrate how to call the Statistics Constructor directly.


def main():

    mimg = afwImage.MaskedImageF(lsst.geom.Extent2I(100, 100))
    mimValue = (2, 0x0, 1)
    mimg.set(mimValue)

    # call with the factory function ... should get stats on the image plane
    fmt = "%-40s %-16s %3.1f\n"
    print(fmt % ("Using makeStatistics:", "(should be " + str(mimValue[0]) + ")",
                 afwMath.makeStatistics(mimg, afwMath.MEAN).getValue()), end=' ')

    # call the constructor directly ... once for image plane, then for variance
    # - make sure we're not using weighted stats for this
    sctrl = afwMath.StatisticsControl()
    sctrl.setWeighted(False)
    print(fmt % ("Using Statistics on getImage():", "(should be " + str(mimValue[0]) + ")",
                 afwMath.StatisticsF(mimg.getImage(), mimg.getMask(), mimg.getVariance(),
                                     afwMath.MEAN, sctrl).getValue()), end=' ')
    print(fmt % ("Using Statistics on getVariance():", "(should be " + str(mimValue[2]) + ")",
                 afwMath.StatisticsF(mimg.getVariance(), mimg.getMask(), mimg.getVariance(),
                                     afwMath.MEAN, sctrl).getValue()), end=' ')

    # call makeStatistics as a front-end for the constructor
    print(fmt % ("Using makeStatistics on getImage():", "(should be " + str(mimValue[0]) + ")",
                 afwMath.makeStatistics(mimg.getImage(), mimg.getMask(), afwMath.MEAN).getValue()), end=' ')
    print(fmt % ("Using makeStatistics on getVariance():", "(should be " + str(mimValue[2]) + ")",
                 afwMath.makeStatistics(mimg.getVariance(), mimg.getMask(),
                                        afwMath.MEAN).getValue()), end=' ')


if __name__ == '__main__':
    main()
