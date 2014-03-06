import os.path
import lsst.afw.geom as afwGeom
from lsst.afw.table import AmpInfoCatalog
from .cameraGeomLib import FOCAL_PLANE, PUPIL, PIXELS, TAN_PIXELS, ACTUAL_PIXELS, CameraSys, \
    Detector, Orientation, CameraTransformMap
from .camera import Camera
from .makePixelToTanPixel import makePixelToTanPixel

__all__ = ["CameraFactoryTask"]

class CameraFactoryTask(object):
    """Make a camera

    Eventually we hope that camera data will be unpersisted using a butler,
    which is why this is written to look something like a Task.
    """
    cameraSysList = [PUPIL, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS]
    cameraSysMap = dict((sys.getSysName(), sys) for sys in cameraSysList)

    def __init__(self):
        """Construct a CameraFactoryTask
        """

    def run(self, cameraConfig, ampInfoPath, shortNameFunc):
        """Construct a camera (lsst.afw.cameraGeom Camera)

        @param[in] cameraConfig: an instance of CameraConfig
        @param[in] ampInfoPath: path to ampInfo data files
        @param[in] shortNameFunc: a function that converts a long detector name to a short one
        @return camera (an lsst.afw.cameraGeom.Camera)
        """
        ampInfoCatDict = dict()
        for detectorConfig in cameraConfig.detectorList.itervalues():
            shortName = shortNameFunc(detectorConfig.name)
            ampCatPath = os.path.join(ampInfoPath, shortName + ".fits")
            ampInfoCatalog = AmpInfoCatalog.readFits(ampCatPath)
            ampInfoCatDict[detectorConfig.name] = ampInfoCatalog

        return self.runCatDict(cameraConfig, ampInfoCatDict)

    def runCatDict(self, cameraConfig, ampInfoCatDict):
        """Construct a camera (lsst.afw.cameraGeom Camera)

        @param[in] cameraConfig: an instance of CameraConfig
        @param[in] ampInfoCatDict: a dictionary keyed on the detector name of AmpInfoCatalog objects
        @return camera (an lsst.afw.cameraGeom.Camera)
        """
        nativeSys = self.cameraSysMap[cameraConfig.transformDict.nativeSys]
        transformDict = self.makeTransformDict(cameraConfig.transformDict.transforms)
        focalPlaneToPupil = transformDict[PUPIL]
        transformMap = CameraTransformMap(nativeSys, transformDict)

        detectorList = []
        for detectorConfig in cameraConfig.detectorList.itervalues():
            ampInfoCatalog = ampInfoCatDict[detectorConfig.name]

            detectorList.append(self.makeDetector(
                detectorConfig = detectorConfig,
                ampInfoCatalog = ampInfoCatalog,
                focalPlaneToPupil = focalPlaneToPupil,
                plateScale = cameraConfig.plateScale,
            ))

        return Camera(cameraConfig.name, detectorList, transformMap)

    def makeDetector(self, detectorConfig, ampInfoCatalog, focalPlaneToPupil, plateScale):
        """Make a detector object:

        @param detectorConfig -- config for this detector (an lsst.pex.config.Config)
        @param ampInfoCatalog -- amplifier information for this detector (an lsst.afw.table.AmpInfoCatalog)
        @param focalPlaneToPupil -- FOCAL_PLANE to PUPIL XYTransform
        @param plateScale -- nominal plate scale (arcsec/mm)
        @return detector (an lsst.afw.cameraGeom.Detector)
        """
        orientation = self.makeOrientation(detectorConfig)
        pixelSizeMm = afwGeom.Extent2D(detectorConfig.pixelSize_x, detectorConfig.pixelSize_y)
        transforms = self.makeTransformDict(detectorConfig.transformDict.transforms)
        transforms[FOCAL_PLANE] = orientation.makePixelFpTransform(pixelSizeMm)

        llPoint = afwGeom.Point2I(detectorConfig.bbox_x0, detectorConfig.bbox_y0)
        urPoint = afwGeom.Point2I(detectorConfig.bbox_x1, detectorConfig.bbox_y1)
        bbox = afwGeom.Box2I(llPoint, urPoint)

        tanPixSys = CameraSys(TAN_PIXELS, detectorConfig.name)
        transforms[tanPixSys] = makePixelToTanPixel(
            bbox = bbox,
            orientation = orientation,
            focalPlaneToPupil = focalPlaneToPupil,
            pixelSizeMm = pixelSizeMm,
            plateScale = plateScale,
        )

        return Detector(
            detectorConfig.name,
            detectorConfig.detectorType,
            detectorConfig.serial,
            bbox,
            ampInfoCatalog,
            orientation,
            pixelSizeMm,
            transforms,
        )

    def makeOrientation(self, detectorConfig):
        """Make an instance of an Orientation class

        @param detectorConfig -- config for this detector (an lsst.pex.config.Config)
        @return orientation (an lsst.afw.cameraGeom.Orientation)
        """
        offset = afwGeom.Point2D(detectorConfig.offset_x, detectorConfig.offset_y)
        refPos = afwGeom.Point2D(detectorConfig.refpos_x, detectorConfig.refpos_y)
        yaw = afwGeom.Angle(detectorConfig.yawDeg, afwGeom.degrees)
        pitch = afwGeom.Angle(detectorConfig.pitchDeg, afwGeom.degrees)
        roll = afwGeom.Angle(detectorConfig.rollDeg, afwGeom.degrees)
        return Orientation(offset, refPos, yaw, pitch, roll)
        
    def makeTransformDict(self, transformConfigDict):
        """Make a dictionary of CameraSys: XYTransform.

        @param transformConfigDict -- an lsst.pex.config.ConfigDictField from an XYTransform registry;
            keys are camera system names.
        @return a dict of CameraSys or CameraSysPrefix: XYTransform
        """
        resMap = dict()
        if transformConfigDict is not None:
            for key in transformConfigDict:
                #TODO This needs to be handled by someone else.
                if key == "Pupil":
                    transform = afwGeom.InvertedXYTransform(transformConfigDict[key].transform.apply())
                else:
                    transform = transformConfigDict[key].transform.apply()
                resMap[CameraSys(key)] =  transform
        return resMap
