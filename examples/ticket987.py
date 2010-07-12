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

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import os

################################################
def main():

    dataDir   = os.environ['AFWDATA_DIR']
    imagePath = os.path.join(dataDir, "DC3a-Sim", "sci", "v5-e0", "v5-e0-c011-a00.sci")
    mimg      = afwImage.MaskedImageF(imagePath)
    binsize   = 512
    bctrl     = afwMath.BackgroundControl("NATURAL_SPLINE")

    ###  Adding this line solves the problem  ###
    # note: by default undersampleStyle is THROW_EXCEPTION 
    bctrl.setUndersampleStyle(afwMath.REDUCE_INTERP_ORDER)
    ################################################

    nx = int(mimg.getWidth()/binsize) + 1
    ny = int(mimg.getHeight()/binsize) + 1
    
    #print 'Binning', nx, ny
    bctrl.setNxSample(nx)
    bctrl.setNySample(ny)
    image   = mimg.getImage()
    backobj = afwMath.makeBackground(image, bctrl)
    image  -= backobj.getImageF()
    del image
    del backobj

#################################################
if __name__ == '__main__':
    main()
