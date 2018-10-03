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
from lsst.pex.exceptions import InvalidParameterError
import lsst.geom
from .cameraGeomLib import FOCAL_PLANE, PIXELS
from .detectorCollection import DetectorCollection
from .pupil import PupilFactory


class Camera(DetectorCollection):
    """!A collection of Detectors plus additional coordinate system support

    Camera.transform transforms points from one camera coordinate system to another.

    Camera.getTransform returns a transform between camera coordinate systems.

    Camera.findDetectors finds all detectors overlapping a specified point.
    """

    def __init__(self, name, detectorList, transformMap, pupilFactoryClass=PupilFactory):
        """!Construct a Camera

        @param[in] name  name of camera
        @param[in] detectorList  a sequence of detectors in index order
        @param[in] transformMap  a TransformMap that at least supports
            FOCAL_PLANE and FIELD_ANGLE coordinates
        @param[in] pupilFactoryClass  a PupilFactory class for this camera
            [default: afw.cameraGeom.PupilFactory]
        """
        self._name = name
        self._transformMap = transformMap
        self._nativeCameraSys = FOCAL_PLANE
        self._pupilFactoryClass = pupilFactoryClass
        super(Camera, self).__init__(detectorList)

    def getName(self):
        """!Return the camera name
        """
        return self._name

    def getPupilFactory(self, visitInfo, pupilSize, npix, **kwargs):
        """!Construct a PupilFactory.

        @param[in] visitInfo  VisitInfo object for a particular exposure.
        @param[in] pupilSize  Size in meters of constructed Pupil array.
                              Note that this may be larger than the actual
                              diameter of the illuminated pupil to
                              accommodate zero-padding.
        @param[in] npix       Constructed Pupils will be npix x npix.
        @param[in] kwargs     Other keyword arguments for the pupil factory
        """
        return self._pupilFactoryClass(visitInfo, pupilSize, npix, **kwargs)

    @property
    def telescopeDiameter(self):
        return self._pupilFactoryClass.telescopeDiameter

    def _getTransformFromOneTransformMap(self, fromSys, toSys):
        """!Get a transform from one TransformMap

        `fromSys` and `toSys` must both be present in the same transform map,
        but that transform map may be from any detector or this camera object.

        @param[in] fromSys  Camera coordinate system of `position`
                        input points
        @param[in] toSys  Camera coordinate system of returned point(s)

        @return an lsst.afw.geom.TransformPoint2ToPoint2 that transforms from
            `fromSys` to `toSys` in the forward direction

        @throws lsst.pex.exceptions.InvalidParameterError if no transform is
        available. This includes the case that fromSys specifies a known
        detector and toSys specifies any other detector (known or unknown)
        @throws KeyError if an unknown detector is specified
        """
        if fromSys.hasDetectorName():
            det = self[fromSys.getDetectorName()]
            return det.getTransformMap().getTransform(fromSys, toSys)
        elif toSys.hasDetectorName():
            det = self[toSys.getDetectorName()]
            return det.getTransformMap().getTransform(fromSys, toSys)
        else:
            return self.getTransformMap().getTransform(fromSys, toSys)

    def findDetectors(self, point, cameraSys):
        """!Find the detectors that cover a point in any camera system

        @param[in] point  position to use in lookup (lsst.geom.Point2D)
        @param[in] cameraSys  camera coordinate system of `point`
        @return a list of zero or more Detectors that overlap the specified point
        """
        # convert `point` to the native coordinate system
        transform = self._getTransformFromOneTransformMap(cameraSys, self._nativeCameraSys)
        nativePoint = transform.applyForward(point)

        detectorList = []
        for detector in self:
            nativeToPixels = detector.getTransform(self._nativeCameraSys, PIXELS)
            pointPixels = nativeToPixels.applyForward(nativePoint)
            if lsst.geom.Box2D(detector.getBBox()).contains(pointPixels):
                detectorList.append(detector)
        return detectorList

    def findDetectorsList(self, pointList, cameraSys):
        """!Find the detectors that cover a list of points in any camera system

        @param[in] pointList  a list of points (lsst.geom.Point2D)
        @param[in] cameraSys  the camera coordinate system of the points in `pointList`
        @return a list of lists; each list contains the names of all detectors
        which contain the corresponding point
        """

        # transform the points to the native coordinate system
        transform = self._getTransformFromOneTransformMap(cameraSys, self._nativeCameraSys)
        nativePointList = transform.applyForward(pointList)

        detectorList = []
        for i in range(len(pointList)):
            detectorList.append([])

        for detector in self:
            pixelSys = detector.makeCameraSys(PIXELS)
            transform = detector.getTransformMap().getTransform(self._nativeCameraSys, pixelSys)
            detectorPointList = transform.applyForward(nativePointList)
            box = lsst.geom.Box2D(detector.getBBox())
            for i, pt in enumerate(detectorPointList):
                if box.contains(pt):
                    detectorList[i].append(detector)

        return detectorList

    def getTransform(self, fromSys, toSys):
        """!Get a transform from one CameraSys to another

        @param[in] fromSys  From CameraSys
        @param[in] toSys  To CameraSys
        @return an lsst.afw.geom.TransformPoint2ToPoint2 that transforms from
            `fromSys` to `toSys` in the forward direction

        @throws lsst.pex.exceptions.InvalidParameterError if no transform is
        available.
        @throws KeyError if an unknown detector is specified
        """
        # Cameras built via makeCameraFromConfig or makeCameraFromPath
        # should now have all coordinate systems available in their
        # transformMap.
        try:
            return self.getTransformMap().getTransform(fromSys, toSys)
        except InvalidParameterError:
            pass
        # Camera must have been constructed an in an unusual way (which we
        # still support for backwards compatibility).
        # All transform maps should know about the native coordinate system
        fromSysToNative = self._getTransformFromOneTransformMap(fromSys, self._nativeCameraSys)
        nativeToToSys = self._getTransformFromOneTransformMap(self._nativeCameraSys, toSys)
        return fromSysToNative.then(nativeToToSys)

    def getTransformMap(self):
        """!Obtain the transform registry.

        @return a TransformMap

        @note: TransformRegistries are immutable, so this should be safe.
        """
        return self._transformMap

    def transform(self, points, fromSys, toSys):
        """!Transform a point or list of points from one camera coordinate system
        to another

        @param[in] points  an lsst.geom.Point2D or list of Point2D
        @param[in] fromSys  Transform from this CameraSys
        @param[in] toSys  Transform to this CameraSys
        @return `points` transformed to `toSys` (an lsst.geom.Point2D
        or list of Point2D)
        """
        transform = self.getTransform(fromSys, toSys)
        return transform.applyForward(points)
