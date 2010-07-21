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

for x,y in [(0,0), (359.9823936, -3.9997742), (-1,0), (1,1), (0,-1), (-1,-1), (0, -11), (11,-11), (360, -4), (0,-4), (360,0), (100,0), (200,0),
            (300,0), (350,0), (355,0), (359,0), (357,0), (358,0)]:
    pa = wcs.pixArea(afwGeom.makePointD(x,y));
    print 'pixel area at (%g,%g): %g arcsec/pix' % (x, y, 3600. * sqrt(pa))


import matplotlib
matplotlib.use('Agg')
from pylab import *
from numpy import *

clf()
X=arange(0, 400, 0.1)
A = [3600.*sqrt(wcs.pixArea(afwGeom.makePointD(x,0))) for x in X]
plot(X,A,'r-')
xlabel('X')
ylabel('pixel scale ("/pix)')
savefig('pixscale.png')
