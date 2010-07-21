import os.path
from math import *

import eups
import lsst.afw.image as afwImg
import lsst.afw.geom as afwGeom

mypath = eups.productDir('afw')

# fails due to #1386
#fn = os.path.join(mypath, 'tests', 'testWcs-1.wcs')
fn = os.path.join(mypath, 'tests', 'testWcs-2.wcs')
fitsheader = afwImg.readMetadata(fn)
wcs = afwImg.makeWcs(fitsheader)

pa = wcs.pixArea(afwGeom.makePointD(0,0));
print 'pixel area:', pa
print 'sqrt', sqrt(pa)
print '3600 sqrt', 3600. * sqrt(pa)

pa = wcs.pixArea(afwGeom.makePointD(359.9823936, -3.9997742));
print 'pixel area:', pa
print 'sqrt:', sqrt(pa)
print '3600 x sqrt:', 3600. * sqrt(pa)


    


