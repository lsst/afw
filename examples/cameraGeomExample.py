#!/usr/bin/env python
import lsst.afw.cameraGeom.testUtils as testUtils
import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom

# Construct a mock LSST-like camera.
# Normally you would obtain a camera from a data butler using butler.get("camera")
# but when this example was written the software stack did not including a
# sample data repository.
camera = testUtils.CameraWrapper(isLsstLike=True).camera

# Get a detector from the camera by name (though you may specify an ID, if
# you prefer).
det = camera["R:1,0 S:1,1"]

# Convert a 2-d point from PIXELS to both FOCAL_PLANE and FIELD_ANGLE coordinates.
pixelPos = afwGeom.Point2D(25, 43.2)
# position in focal plane in mm
focalPlanePos = det.transform(pixelPos, cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
# position in as a field angle, in radians
fieldAnglePos = det.transform(pixelPos, cameraGeom.PIXELS, cameraGeom.FIELD_ANGLE)

# Find all detectors that overlap a specific point (in this case find the
# detector we already have)
detList = camera.findDetectors(focalPlanePos, cameraGeom.FOCAL_PLANE)
assert len(detList) == 1
assert detList[0].getName() == det.getName()
