"""Create a Exposure object for testing with non-zero elements in most fields.
"""
import numpy as np

from lsst.afw.cameraGeom.testUtils import DetectorWrapper
import lsst.afw.image
from lsst.afw.image.utils import resetFilters, defineFilter
import lsst.afw.geom
import lsst.daf.base
import lsst.geom


nx = 10
ny = 10
exp = lsst.afw.image.ExposureF(nx, ny)

# Fill the maskedImage
exp.maskedImage.image.array = np.arange(nx*ny, dtype='f').reshape(nx, ny)
exp.maskedImage.variance.array = np.ones((nx, ny), dtype='f')
exp.maskedImage.mask.array[5, 5] = 5  # make one pixel non-zero

# Fill the detector
detectorWrapper = DetectorWrapper(bbox=exp.getBBox())
exp.setDetector(detectorWrapper.detector)

# Fill the filter
resetFilters()
defineFilter('ha', 656.28)
filt = lsst.afw.image.Filter('ha')
exp.setFilter(filt)

# Fill the Calib
calib = lsst.afw.image.Calib()
calib.setFluxMag0(1e6, 2e4)
exp.setCalib(calib)

# Fill the SkyWcs
ra = 30.0 * lsst.geom.degrees
dec = 40.0 * lsst.geom.degrees
cdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds, orientation=45*lsst.geom.degrees)
crpix = lsst.geom.Point2D(4, 4)
crval = lsst.geom.SpherePoint(ra, dec)
skyWcs = lsst.afw.geom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
exp.setWcs(skyWcs)

# Fill the VisitInfo
exposureId = 12345
exposureTime = 30.0
darkTime = 600.0
date = lsst.daf.base.DateTime(2020, 1, 20, 12, 34, 56, lsst.daf.base.DateTime.TAI)
lat = -12.345*lsst.geom.degrees
lon = 123.45*lsst.geom.degrees
observatory = lsst.afw.coord.Observatory(lat, lon, 2000)
# Skipping ut1, era, altAz, airmass, rotAngle, and weather.
# They can be added later if someone needs them.
visitInfo = lsst.afw.image.VisitInfo(exposureId=exposureId,
                                     exposureTime=exposureTime,
                                     darkTime=darkTime,
                                     date=date,
                                     boresightRaDec=crval,
                                     observatory=observatory)
exp.getInfo().setVisitInfo(visitInfo)

filename = 'exposure-version-XXXX.fits'
exp.writeFits(filename)

print("Wrote ExposureF to:", filename)
print("Please rename it to reflect the version of Exposure that it represents.")
