from lsst.afw.cameraGeom import CameraPoint, DetectorCollection

class Camera(DetectorCollection):
    """A collection of Detectors that also supports coordinate transformation
    """
    def __init__(self, name, detectorList, transformRegistry):
        """Construct a Camera

        @param[in] detectorList: a sequence of detectors in index order
        @param[in] transformRegistry: a coordinate transform registry, a TransformRegistry
        """
        self._name = name
        self._transformRegistry = transformRegistry
        super(Camera, self).__init__(detectorList)
        
    def findDetectors(self, cameraPoint):
        """Find the detectors that cover a given cameraPoint, or empty list
        
        @param[in] cameraPoint: position to use in lookup
        """
        # first convert to focalPlane because it's faster to convert to pixel from focalPlane
        fpCoord = self.convert(cameraPoint, "focalPlane")
        detectorList = []
        for detector in self._detectorList:
            detPoint = detector.convert(fpCoord, "pixel")
            if detector.getBBox().contains(detPoint):
                detectorList.append(detector)
        return detectorList

    def getTransformRegistry(self):
        """Obtain a pointer to the transform registry.  
           Since TransformRegistries are immutable, this should
           be safe.
        """
        return self._transformRegistry

    def convert(self, cameraPoint, coordSys):
        """Convert a CameraPoint to another coordinate system
        
        @param[in] cameraPoint: CameraPoint to convert
        @param[in] coordSys: desired coordinate system name
        """
        if coordSys in self._transformRegistry:
            return self._transformRegistry.convert(cameraPoint, coordSys)
        else:
            detList = self.findDetectors(cameraPoint)
            if len(detList) <= 0:
                raise ValueError("Could not find detector or valid Camera coordinate system. %s"%(coordSys))
            elif len(detList) > 1:
                raise ValueError("Found more than one detector that contains this point.  Cannot convert to more than one coordinate system.")
            else:
                detList[0].convert(cameraPoint, coordSys)

    @classmethod
    def makeCameraPoint(point, coordSys):
        return CameraPoint(point, CameraSys(coordSys))

