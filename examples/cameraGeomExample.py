#!/usr/bin/env python2
import lsst.afw.cameraGeom.testUtils as testUtils
lsstCamWrapper = testUtils.CameraWrapper(isLsstLike=True)
camera = lsstCamWrapper.camera

import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.geom as afwGeom

# get a detector by name
det = camera["R:1,0 S:1,1"]
# convert from PIXELS to FOCAL_PLANE and PUPIL coordinates
detPoint = det.makeCameraPoint(afwGeom.Point2D(25, 43.2), cameraGeom.PIXELS) # position on detector in pixels
fpPoint = det.transform(detPoint, cameraGeom.FOCAL_PLANE) # position in focal plane in mm
pupilPoint = camera.transform(detPoint, cameraGeom.PUPIL) # position in pupil, in radians

# convert PUPIL to PIXELS. The target system (PIXELS) is detector-based, so you may specify a detector
# or let the Camera find a detector:
# * To specify a particular detector, specify the target coordinate system as a CameraSys
#   with the specified detectorName filled in (e.g. use detector.makeCameraSys).
#   This is faster than finding a detector, and the resulting point is allowed to be off the detector.
# * To have the Camera find a detector, specify the target coordinate system as a CameraSysPrefix
#   (e.g. cameraGeom.PIXELS). Camera will search for a detector that overlaps the point;
#   if it finds 0 or more than 1 then it will raise an exception.
detPixelsSys = det.makeCameraSys(cameraGeom.PIXELS)
detPointOnSpecifiedDetector = camera.transform(pupilPoint, detPixelsSys)
detPointOnFoundDetector = camera.transform(pupilPoint, cameraGeom.PIXELS)
assert detPointOnFoundDetector.getCameraSys() == detPixelsSys # same detector
# find a detector given a camera point (in this case find the detector we already have)
detList = camera.findDetectors(fpPoint)
assert len(detList) == 1
assert detList[0].getName() == det.getName()
