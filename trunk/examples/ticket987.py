#!/usr/bin/env python
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
