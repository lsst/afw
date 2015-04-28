#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
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
from __future__ import absolute_import, division
from .cameraGeomLib import CameraPoint, CameraSysPrefix, PIXELS
from .detectorCollection import DetectorCollection
import lsst.afw.geom as afwGeom

class Camera(DetectorCollection):
    """!A collection of Detectors that also supports coordinate transformation
    """
    def __init__(self, name, detectorList, transformMap):
        """!Construct a Camera

        @param[in] name  name of camera
        @param[in] detectorList  a sequence of detectors in index order
        @param[in] transformMap  a CameraTransformMap whose native system is FOCAL_PLANE
            and that at least supports PUPIL coordinates
        """
        self._name = name
        self._transformMap = transformMap
        self._nativeCameraSys = self._transformMap.getNativeCoordSys()
        super(Camera, self).__init__(detectorList)

    def getName(self):
        """!Return the camera name
        """
        return self._name

    def _transformFromNativeSys(self, nativePoint, toSys):
        """!Transform a point in the native coordinate system to another coordinate system.

        @param[in] nativePoint  CameraPoint in the native system for the camera
        @param[in] toSys  destination CameraSys
        @return CameraPoint in the destination CameraSys
        """
        if isinstance(toSys, CameraSysPrefix):
            # Must be in a detector.  Find the detector and transform it.
            detList = self.findDetectors(nativePoint)
            if len(detList) == 0:
                raise RuntimeError("No detectors found")
            elif len(detList) > 1:
                raise RuntimeError("More than one detector found")
            else:
                det = detList[0]
                return det.transform(nativePoint, toSys)
        else:
            return self._transformSingleSys(nativePoint, toSys)

    def _transformSingleSysArray(self, positionArray, fromSys, toSys):
        """!Transform an array of points from once CameraSys to another CameraSys
        @warning This method only handles a single jump, not a transform linked by a common native sys.
        
        @param[in] positionArray Array of Point2D objects, one per position
        @param[in] fromSys  Initial coordinate system
        @param[in] toSys  Destination coordinate system
        
        @returns an array of Point2D objects containing the transformed coordinates in the destination system.
        """
        if fromSys.hasDetectorName():
            det = self[fromSys.getDetectorname()]
            detTrans = det.getTransfromMap()
            return detTrans.transform(positionArray, fromSys, toSys)
        elif toSys.hasDetectorName():
            det = self[toSys.getDetectorName()]
            detTrans = det.getTransformMap()
            return detTrans.transform(positionArray, fromSys, toSys)
        elif toSys in self._transformMap:
            # use camera transform map
            return self._transformMap.transform(positionArray, fromSys, toSys)
        raise RuntimeError("Could not find mapping from %s to %s"%(fromSys, toSys))

    def _transformSingleSys(self, cameraPoint, toSys):
        """!Transform a CameraPoint with a CameraSys to another CameraSys.

        @warning This method only handles a single jump, not a transform linked by a common native sys.

        @param[in] cameraPoint  CameraPoint to transform
        @param[in] toSys  Destination coordinate system
        """
        fromSys = cameraPoint.getCameraSys()
        if fromSys.hasDetectorName():
            # use from detector to transform
            det = self[fromSys.getDetectorName()]
            return det.transform(cameraPoint, toSys)
        elif toSys.hasDetectorName():
            # use the to detector to transform
            det = self[toSys.getDetectorName()]
            return det.transform(cameraPoint, toSys)
        elif toSys in self._transformMap:
            # use camera transform map
            outPoint = self._transformMap.transform(cameraPoint.getPoint(), cameraPoint.getCameraSys(), toSys)
            return CameraPoint(outPoint, toSys)
        raise RuntimeError("Could not find mapping from %s to %s"%(cameraPoint.getCameraSys(), toSys))

    def findDetectors(self, cameraPoint):
        """!Find the detectors that cover a given cameraPoint, or empty list

        @param[in] cameraPoint  position to use in lookup
        @return a list of zero or more Detectors that overlap the specified point
        """
        # first convert to focalPlane since the point may be in another overlapping detector
        nativePoint = self._transformSingleSys(cameraPoint, self._nativeCameraSys)

        detectorList = []
        for detector in self:
            cameraSys = detector.makeCameraSys(PIXELS)
            detPoint = detector.transform(nativePoint, cameraSys)
            #This is safe because CameraPoint is not templated and getPoint() returns a Point2D.
            if afwGeom.Box2D(detector.getBBox()).contains(detPoint.getPoint()):
                detectorList.append(detector)
        return detectorList

    def findDetectorsList(self, cameraPointList, coordSys):
        """!Find the detectors that cover a list of points specified by x and y coordinates in any system

        @param[in] cameraPointList  a list of cameraPoints
        @param[in] coordSys  the camera coordinate system in which cameraPointList is defined
        @return a list of lists; each list contains the names of all detectors which contain the
        corresponding point
        """

        #transform the points to the native coordinate system
        nativePointList = self._transformSingleSysArray(cameraPointList, coordSys, self._nativeCameraSys)

        detectorList = []
        for i in range(len(cameraPointList)):
            detectorList.append([])

        for detector in self:
            coordMap = detector.getTransformMap()
            cameraSys = detector.makeCameraSys(PIXELS)
            detectorPointList = coordMap.transform(nativePointList, self._nativeCameraSys, cameraSys)
            box = afwGeom.Box2D(detector.getBBox())
            for i, pt in enumerate(detectorPointList):
                if box.contains(pt):
                    detectorList[i].append(detector)

        return detectorList

    def getTransformMap(self):
        """!Obtain a pointer to the transform registry.

        @return a TransformMap

        @note: TransformRegistries are immutable, so this should be safe.
        """
        return self._transformMap

    def transform(self, cameraPoint, toSys):
        """!Transform a CameraPoint to a different CameraSys

        @param[in] cameraPoint  CameraPoint to transform
        @param[in] toSys  Transform to this CameraSys
        @return a CameraPoint in the the specified CameraSys
        """
        # All transform maps should know about the native coordinate system
        nativePoint = self._transformSingleSys(cameraPoint, self._nativeCameraSys)
        return self._transformFromNativeSys(nativePoint, toSys)

    def makeCameraPoint(self, point, cameraSys):
        """!Make a CameraPoint from a Point2D and a CameraSys

        @param[in] point  an lsst.afw.geom.Point2D
        @param[in] cameraSys  a CameraSys
        @return the CameraPoint
        """
        if isinstance(cameraSys, CameraSysPrefix):
            raise TypeError("Use the detector method to make a camera point from a CameraSysPrefix.")
        if cameraSys in self._transformMap:
            return CameraPoint(point, cameraSys)
        if cameraSys.hasDetectorName():
            if cameraSys in self[cameraSys.getDetectorName()].getTransformMap():
                return CameraPoint(point, cameraSys)
        raise RuntimeError("Could not find %s in any transformMap"%(cameraSys))
