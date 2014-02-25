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
from .cameraGeomLib import CameraPoint
from .detectorCollection import DetectorCollection

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
        super(Camera, self).__init__(detectorList)
        
    def findDetectors(self, cameraPoint):
        """Find the detectors that cover a given cameraPoint, or empty list
        
        @param[in] cameraPoint: position to use in lookup
        @return a list of zero or more Detectors that overlap the specified point
        """
        # first convert to focalPlane because it's faster to convert to pixel from focalPlane
        fpCoord = self.convert(cameraPoint, "focalPlane")
        detectorList = []
        for detector in self._detectorList:
            detPoint = detector.convert(fpCoord, "pixel")
            if detector.getBBox().contains(detPoint):
                detectorList.append(detector)
        return detectorList

    def getTransformMap(self):
        """Obtain a pointer to the transform registry.  

        @return a TransformMap

        @note: TransformRegistries are immutable, so this should be safe.
        """
        return self._transformMap

    def convert(self, cameraPoint, cameraSys):
        """Convert a CameraPoint to another camera coordinate system
        
        @param[in] cameraPoint: initial CameraPoint
        @param[in] cameraSys: desired camera coordinate system (a CameraSys)
        @return converted cameraPoint (a CameraPoint)
        """
        if cameraSys in self._transformMap:
            return self._transformMap.convert(cameraPoint, cameraSys)
        else:
            detList = self.findDetectors(cameraPoint)
            if len(detList) <= 0:
                raise ValueError("Could not find detector or valid Camera coordinate system. %s"%(cameraSys))
            elif len(detList) > 1:
                raise ValueError("Found more than one detector that contains this point.  Cannot convert to more than one coordinate system.")
            else:
                detList[0].convert(cameraPoint, cameraSys)

    @staticmethod
    def makeCameraPoint(point, cameraSys):
        """Make a CameraPoint from a Point2D and a CameraSys

        @param[in] point: an lsst.afw.geom.Point2D
        @param[in] cameraSys: a CameraSys
        @return cameraPoint: a CameraPoint
        """
        return CameraPoint(point, cameraSys)

