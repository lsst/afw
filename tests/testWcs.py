import os.path

import eups
import lsst.afw.image as afwImg
import lsst.afw.geom as afwGeom

mypath = eups.productDir('afw')

fn = os.path.join(mypath, 'tests', 'testWcs-1.wcs')
fitsheader = afwImg.readMetadata(fn)
wcs = afwImg.makeWcs(fitsheader)

pa = wcs.pixArea(afwGeom.makePointD(0,0));
print 'pixel area:', pa
print 'sqrt', sqrt(pa)
print '3600 sqrt', 3600. * sqrt(pa)



    
