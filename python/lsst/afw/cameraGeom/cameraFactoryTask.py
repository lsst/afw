import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import FOCAL_PLANE, PUPIL, PIXELS, ACTUAL_PIXELS, CameraConfig, \
                                Camera, Detector, Orientation, CameraTransformMap, CameraSys

class CameraFactoryTask(object):
    ConfigClass = CameraConfig
    coordSysList = [PUPIL, FOCAL_PLANE, PIXELS, ACTUAL_PIXELS]
    coordSysMap = dict([(sys.getSysName(), sys) for sys in coordSysList])
    def __init__(self, config, ampInfoCatDict):
        '''
        Construct a CameraFactoryTask
        @param config -- CameraConfig object for this camera
        @param ampInfoCatDict -- Dictionary of afwTable.AmpInfoCatalog objects keyed on detector.name
        '''
        self.config = config
        self.ampInfo = ampInfoCatDict

    def run(self):
        '''
        Construct a camera given a camera config
        '''
        detectorList = []
        for i in xrange(len(self.config.detectorList)):
            ampInfoCat = self.ampInfo[self.config.detectorList[i].name]
            detectorList.append(self.makeDetector(self.config.detectorList[i], ampInfoCat))
        nativeSys = self.coordSysMap[self.config.transformDict.nativeSys]
        transformDict = self.makeTransformDict(self.config.transformDict.transforms)
        transformMap = CameraTransformMap(nativeSys, transformDict)
        return Camera(self.config.name, detectorList, transformMap)

    def makeDetector(self, config, ampInfoCatalog):
        """
        Make a detector object:
        @param config -- The config for this detector
        @param ampInfoCatalog -- The ampInfoCatalog for the amps in this detector
        """
        orientation = self.makeOrientation(config)
        pixelSize = afwGeom.Extent2D(config.pixelSize_x, config.pixelSize_y)
        transformDict = {FOCAL_PLANE:orientation.makePixelFpTransform(pixelSize)}
        transforms = self.makeTransformDict(config.transformDict.transforms, defaultMap=transformDict)
        llPoint = afwGeom.Point2I(config.bbox_x0, config.bbox_y0)
        urPoint = afwGeom.Point2I(config.bbox_x1, config.bbox_y1)
        bbox = afwGeom.Box2I(llPoint, urPoint)
        return Detector(config.name, config.detectorType, config.serial, bbox, ampInfoCatalog, 
                             orientation, pixelSize, transforms)

    def makeOrientation(self, config):
        """
        Make an instance of an Orientation class
        @param config -- config containing the necessary information
        """
        offset = afwGeom.Point2D(config.offset_x, config.offset_y)
        refPos = afwGeom.Point2D(config.refpos_x, config.refpos_y)
        yaw = afwGeom.Angle(config.yawDeg, afwGeom.degrees)
        pitch = afwGeom.Angle(config.pitchDeg, afwGeom.degrees)
        roll = afwGeom.Angle(config.rollDeg, afwGeom.degrees)
        return Orientation(offset, refPos, yaw, pitch, roll)
        
    def makeTransformDict(self, transformConfigDict, defaultMap={}):
        """
        Make a dictionary of CameraSys and transforms.  Optionally provide default transforms.
        @param transformConfigDict -- A dictionary of transforms
        @param defaultMap -- A dictionary of default transforms
        """
        if transformConfigDict is not None:
            for key in transformConfigDict:
                defaultMap[self.coordSysMap[key]] = transformConfigDict[key].transform.apply()
        return defaultMap
