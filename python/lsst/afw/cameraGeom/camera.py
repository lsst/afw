from lsst.afw.cameraGeom import CameraPoint, DetectorCollection, CameraSys, FOCAL_PLANE, PIXELS
import lsst.afw.geom as afwGeom

class Camera(DetectorCollection):
    """A collection of Detectors that also supports coordinate transformation
    """
    def __init__(self, name, detectorList, transformMap):
        """Construct a Camera

        @param[in] detectorList: a sequence of detectors in index order
        @param[in] transformMap: a coordinate transform registry, a TransformMap
        """
        self._name = name
        self._transformMap = transformMap
        super(Camera, self).__init__(detectorList)
        
    def findDetectors(self, cameraPoint):
        """Find the detectors that cover a given cameraPoint, or empty list
        
        @param[in] cameraPoint: position to use in lookup
        """
        # first convert to focalPlane because it's faster to convert to pixel from focalPlane
        fpCoord = self.convert(cameraPoint, FOCAL_PLANE)
        detectorList = []
        for detector in self._detectorList:
            cameraSys = detector.makeCameraSys(PIXELS)
            detPoint = detector.transform(fpCoord, cameraSys)
            #This is safe because CameraPoint is not templated and getPoint() returns a Point2D.
            if afwGeom.Box2D(detector.getBBox()).contains(detPoint.getPoint()):
                detectorList.append(detector)
        return detectorList

    def getTransformMap(self):
        """Obtain a pointer to the transform registry.  
           Since TransformRegistries are immutable, this should
           be safe.
        """
        return self._transformMap

    def transform(self, cameraPoint, toSys):
        """Convert a CameraPoint to another coordinate system
        
        @param[in] cameraPoint: CameraPoint to convert
        @param[in] toSys: desired CameraSystem
        """
        if coordSys in self._transformMap.getCoordSysList():
            p = self._transformMap.transform(cameraPoint.getPoint(), cameraPoint.getCameraSys(), toSys)
            return CameraPoint(p, toSys)
        else:
            detList = self.findDetectors(cameraPoint)
            if len(detList) <= 0:
                raise ValueError("Could not find detector or valid Camera coordinate system. %s"%(toSys))
            elif len(detList) > 1:
                raise ValueError("Found more than one detector that contains this point.  Cannot convert to more than one coordinate system.")
            else:
                detList[0].transform(cameraPoint, toSys)

    @staticmethod
    def makeCameraPoint(point, coordSysName):
        return CameraPoint(point, CameraSys(coordSysName))

