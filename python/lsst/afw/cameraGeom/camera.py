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
from .cameraGeomLib import CameraPoint, CameraSys, CameraSysPrefix, PIXELS
from .detectorCollection import DetectorCollection
import lsst.afw.geom as afwGeom

class Camera(DetectorCollection):
    """A collection of Detectors that also supports coordinate transformation
    """
    def __init__(self, name, detectorList, transformMap):
        """Construct a Camera

        @param[in] detectorList: a sequence of detectors in index order
        @param[in] transformMap: a CameraTransformMap whose native system is FOCAL_PLANE
            and that at least supports PUPIL coordinates
        """
        self._name = name
        self._transformMap = transformMap
        self._nativeCameraSys = self._transformMap.getNativeCoordSys()
        super(Camera, self).__init__(detectorList)
  
    def getName(self):
        return self._name

    def _getNativeSysPoint(self, cameraPoint):
        fromSys = cameraPoint.getCameraSys()
        if not isinstance(fromSys, CameraSys):
            raise TypeError("CameraSystem must be fully qualified in order to convert to native coordinates.")
        if fromSys in self._transformMap:
            # Use camera map
            p = self._transformMap.transform(cameraPoint.getPoint(), cameraPoint.getCameraSys(),
                                                self._nativeCameraSys)
            return CameraPoint(p, self._nativeCameraSys)
        else:
            for detector in self:
                if fromSys in detector.getTransformMap():
                    # Return first one that matches.
                    p = detector.getTransformMap().transform(cameraPoint.getPoint(), 
                                                                cameraPoint.getCameraSys(), self._nativeCameraSys)
                    return CameraPoint(p, self._nativeCameraSys)
        raise ValueError("Could not convert camera point from %s to %s"%(cameraPoint.getCameraSys(), self._nativeCameraSys))
        
    def findDetectors(self, cameraPoint):
        """Find the detectors that cover a given cameraPoint, or empty list
        
        @param[in] cameraPoint: position to use in lookup
        @return a list of zero or more Detectors that overlap the specified point
        """
        
        # first convert to focalPlane since the point may be in another overlapping detector
        nativeCoord = self._getNativeSysPoint(cameraPoint)
        
        detectorList = []
        for detector in self:
            cameraSys = detector.makeCameraSys(PIXELS)
            detPoint = detector.transform(nativeCoord, cameraSys)
            #This is safe because CameraPoint is not templated and getPoint() returns a Point2D.
            if afwGeom.Box2D(detector.getBBox()).contains(detPoint.getPoint()):
                detectorList.append(detector)
        return detectorList

    def getTransformMap(self):
        """Obtain a pointer to the transform registry.  

        @return a TransformMap

        @note: TransformRegistries are immutable, so this should be safe.
        """
        return self._transformMap

    def transform(self, cameraPoint, toSys):
        """Transform a CameraPoint to a different CameraSys
        @param[in] cameraPoint: CameraPoint to transform
        @param[in] toSys: Transform to this CameraSys
        @return a CameraPoint in the new CameraSys
        """
        transformMap = None
        # All transform maps should know about the native coordinate system
        nativeCoord = self._getNativeSysPoint(cameraPoint)
        if isinstance(toSys, CameraSysPrefix):
            # Must be in a detector.  Find the detector and transform it.
            detList = self.findDetectors(nativeCoord)
            if len(detList) > 0:
                for det in detList:
                    if det.makeCameraSys(toSys) in det.getTransformMap():
                        if transformMap is None:
                            transformMap = det.getTransformMap()
                            toSys = det.makeCameraSys(toSys)
                        else:
                            raise ValueError("Found more than one detector that contains this point.  "+
                                             "Cannot convert to more than one coordinate system.")
        elif toSys.getDetectorName():
            # use the detctor to transform
            det = self[toSys.getDetectorName()]
            transformMap = det.getTransformMap()
            
        elif toSys in self._transformMap:
            transformMap = self._transformMap
        else:
            pass

        if transformMap is None:
            raise ValueError("Could not find mapping between %s and %s"%(cameraPoint.getCameraSys(), toSys))

        p = transformMap.transform(nativeCoord.getPoint(), nativeCoord.getCameraSys(), toSys) 
        return CameraPoint(p, toSys)
    @staticmethod
    def makeCameraPoint(point, cameraSys):
        """Make a CameraPoint from a Point2D and a CameraSys

        @param[in] point: an lsst.afw.geom.Point2D
        @param[in] cameraSys: a CameraSys
        @return cameraPoint: a CameraPoint
        """
        return CameraPoint(point, cameraSys)

