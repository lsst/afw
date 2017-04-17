#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import lsst.afw.cameraGeom.testUtils as testUtils
import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom

# Construct a mock LSST-like camera.
# Normally you would obtain a camera from a data butler using butler.get("camera")
# but when this example was written the software stack did not including a sample data repository.
camera = testUtils.CameraWrapper(isLsstLike=True).camera

# Get a detector from the camera by name (though you may specify an ID, if you prefer).
det = camera["R:1,0 S:1,1"]

# Convert a 2-d point from PIXELS to both FOCAL_PLANE and PUPIL coordinates.
detPoint = det.makeCameraPoint(afwGeom.Point2D(25, 43.2), cameraGeom.PIXELS) # position on detector in pixels
fpPoint = det.transform(detPoint, cameraGeom.FOCAL_PLANE) # position in focal plane in mm
pupilPoint = camera.transform(detPoint, cameraGeom.PUPIL) # position in pupil, in radians

# Find all detectors that overlap a specific point (in this case find the detector we already have)
detList = camera.findDetectors(fpPoint)
assert len(detList) == 1
assert detList[0].getName() == det.getName()

# Convert a point from PUPIL to PIXELS coordinates.
# For a detector-based coordinate system, such as PIXELS, may specify a particular detector
# or let the Camera find a detector:
# * To specify a particular detector, specify the target coordinate system as a CameraSys
#   with the specified detectorName filled in (e.g. use detector.makeCameraSys).
#   This is faster than finding a detector, and the resulting point is allowed to be off the detector.
# * To have the Camera find a detector, specify the target coordinate system as a CameraSysPrefix
#   (e.g. cameraGeom.PIXELS). Camera will search for a detector that overlaps the point.
#   If it finds exactly one detector then it will use that detector, and you can figure
#   out which detector it used from the detector name in the returned CameraPoint.
#   If it finds no detectors, or more than one detector, then it will raise an exception.
detPixelsSys = det.makeCameraSys(cameraGeom.PIXELS)
detPointOnSpecifiedDetector = camera.transform(pupilPoint, detPixelsSys)
detPointOnFoundDetector = camera.transform(pupilPoint, cameraGeom.PIXELS)
assert detPointOnFoundDetector.getCameraSys() == detPixelsSys # same detector
