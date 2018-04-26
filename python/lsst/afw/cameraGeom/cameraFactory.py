import os.path
import lsst.afw.geom as afwGeom
from lsst.afw.table import AmpInfoCatalog
from .cameraGeomLib import FOCAL_PLANE, FIELD_ANGLE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS, CameraSys, \
    Detector, DetectorType, Orientation, TransformMap
from .camera import Camera
from .makePixelToTanPixel import makePixelToTanPixel
from .pupil import PupilFactory

__all__ = ["makeCameraFromPath", "makeCameraFromCatalogs",
           "makeDetector", "copyDetector"]

cameraSysList = [FIELD_ANGLE, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS]
cameraSysMap = dict((sys.getSysName(), sys) for sys in cameraSysList)


def makeDetector(detectorConfig, ampInfoCatalog, focalPlaneToField):
    """!Make a Detector instance from a detector config and amp info catalog

    @param detectorConfig  config for this detector (an lsst.pex.config.Config)
    @param ampInfoCatalog  amplifier information for this detector (an lsst.afw.table.AmpInfoCatalog)
    @param focalPlaneToField  FOCAL_PLANE to FIELD_ANGLE Transform
    @return detector (an lsst.afw.cameraGeom.Detector)
    """
    orientation = makeOrientation(detectorConfig)
    pixelSizeMm = afwGeom.Extent2D(
        detectorConfig.pixelSize_x, detectorConfig.pixelSize_y)
    transforms = makeTransformDict(detectorConfig.transformDict.transforms)
    transforms[FOCAL_PLANE] = orientation.makePixelFpTransform(pixelSizeMm)

    llPoint = afwGeom.Point2I(detectorConfig.bbox_x0, detectorConfig.bbox_y0)
    urPoint = afwGeom.Point2I(detectorConfig.bbox_x1, detectorConfig.bbox_y1)
    bbox = afwGeom.Box2I(llPoint, urPoint)

    tanPixSys = CameraSys(TAN_PIXELS, detectorConfig.name)
    transforms[tanPixSys] = makePixelToTanPixel(
        bbox=bbox,
        orientation=orientation,
        focalPlaneToField=focalPlaneToField,
        pixelSizeMm=pixelSizeMm,
    )

    args = [
        detectorConfig.name,
        detectorConfig.id,
        DetectorType(detectorConfig.detectorType),
        detectorConfig.serial,
        bbox,
        ampInfoCatalog,
        orientation,
        pixelSizeMm,
        transforms,
    ]
    crosstalk = detectorConfig.getCrosstalk(len(ampInfoCatalog))
    if crosstalk:
        args.append(crosstalk)
    return Detector(*args)


def copyDetector(detector, ampInfoCatalog=None):
    """!Return a copy of a Detector

    @param detector        The Detector to clone
    @param ampInfoCatalog  The ampInfoCatalog to use; default use original

    N.b. No deep copies are made;  the input transformDict is used unmodified
    """
    if ampInfoCatalog is None:
        ampInfoCatalog = detector.getAmpInfoCatalog()

    tm = detector.getTransformMap()
    native = detector.getNativeCoordSys()
    transformDict = dict()
    for cs in tm:
        if cs != native:
            transformDict[cs] = tm.getTransform(native, cs)

    return Detector(detector.getName(), detector.getId(), detector.getType(),
                    detector.getSerial(), detector.getBBox(),
                    ampInfoCatalog, detector.getOrientation(), detector.getPixelSize(),
                    transformDict, detector.getCrosstalk())


def makeOrientation(detectorConfig):
    """!Make an Orientation instance from a detector config

    @param detectorConfig  config for this detector (an lsst.pex.config.Config)
    @return orientation (an lsst.afw.cameraGeom.Orientation)
    """
    offset = afwGeom.Point2D(detectorConfig.offset_x, detectorConfig.offset_y)
    refPos = afwGeom.Point2D(detectorConfig.refpos_x, detectorConfig.refpos_y)
    yaw = afwGeom.Angle(detectorConfig.yawDeg, afwGeom.degrees)
    pitch = afwGeom.Angle(detectorConfig.pitchDeg, afwGeom.degrees)
    roll = afwGeom.Angle(detectorConfig.rollDeg, afwGeom.degrees)
    return Orientation(offset, refPos, yaw, pitch, roll)


def makeTransformDict(transformConfigDict):
    """!Make a dictionary of CameraSys: lsst.afw.geom.Transform from a config dict.

    @param transformConfigDict  an lsst.pex.config.ConfigDictField from an lsst.afw.geom.Transform
        registry; keys are camera system names.
    @return a dict of CameraSys or CameraSysPrefix: lsst.afw.geom.Transform
    """
    resMap = dict()
    if transformConfigDict is not None:
        for key in transformConfigDict:
            transform = transformConfigDict[key].transform.apply()
            resMap[CameraSys(key)] = transform
    return resMap


def makeCameraFromPath(cameraConfig, ampInfoPath, shortNameFunc,
                       pupilFactoryClass=PupilFactory):
    """!Make a Camera instance from a directory of ampInfo files

    The directory must contain one ampInfo fits file for each detector in cameraConfig.detectorList.
    The name of each ampInfo file must be shortNameFunc(fullDetectorName) + ".fits".

    @param[in] cameraConfig  an instance of CameraConfig
    @param[in] ampInfoPath  path to ampInfo data files
    @param[in] shortNameFunc  a function that converts a long detector name to a short one
    @param[in] pupilFactoryClass class to attach to camera; default afw.cameraGeom.PupilFactory
    @return camera (an lsst.afw.cameraGeom.Camera)
    """
    ampInfoCatDict = dict()
    for detectorConfig in cameraConfig.detectorList.values():
        shortName = shortNameFunc(detectorConfig.name)
        ampCatPath = os.path.join(ampInfoPath, shortName + ".fits")
        ampInfoCatalog = AmpInfoCatalog.readFits(ampCatPath)
        ampInfoCatDict[detectorConfig.name] = ampInfoCatalog

    return makeCameraFromCatalogs(cameraConfig, ampInfoCatDict, pupilFactoryClass)


def makeCameraFromCatalogs(cameraConfig, ampInfoCatDict,
                           pupilFactoryClass=PupilFactory):
    """!Construct a Camera instance from a dictionary of detector name: AmpInfoCatalog

    @param[in] cameraConfig  an instance of CameraConfig
    @param[in] ampInfoCatDict  a dictionary of detector name: AmpInfoCatalog
    @param[in] pupilFactoryClass class to attach to camera; default afw.cameraGeom.PupilFactory
    @return camera (an lsst.afw.cameraGeom.Camera)
    """
    nativeSys = cameraSysMap[cameraConfig.transformDict.nativeSys]
    transformDict = makeTransformDict(cameraConfig.transformDict.transforms)
    focalPlaneToField = transformDict[FIELD_ANGLE]
    transformMap = TransformMap(nativeSys, transformDict)

    detectorList = []
    for detectorConfig in cameraConfig.detectorList.values():
        ampInfoCatalog = ampInfoCatDict[detectorConfig.name]

        detectorList.append(makeDetector(
            detectorConfig=detectorConfig,
            ampInfoCatalog=ampInfoCatalog,
            focalPlaneToField=focalPlaneToField,
        ))

    return Camera(cameraConfig.name, detectorList, transformMap, pupilFactoryClass)
