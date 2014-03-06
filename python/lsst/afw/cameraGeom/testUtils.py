import os
import numpy
import eups
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from .cameraGeomLib import PIXELS, PUPIL, FOCAL_PLANE, SCIENCE, ACTUAL_PIXELS,\
                           CameraSys, Detector, Orientation
from .camera import Camera
from .cameraConfig import DetectorConfig, CameraConfig
from .cameraFactoryTask import CameraFactoryTask

__all__ = ["DetectorWrapper", "CameraWrapper"]

class DetectorWrapper(object):
    """Construct a simple detector

    Intended for use with unit tests, thus saves a copy of all input parameters.
    Does not support setting details of amplifiers.

    @param[in] name: detector name
    @param[in] detType: detector type
    @param[in] serial: serial "number" (a string)
    @param[in] bbox: bounding box; defaults to something sensible
    @param[in] numAmps: number of amplifiers
    @param[in] pixelSize: pixel size (mm)
    @param[in] ampExtent: dimensions of amplifier image bbox
    @param[in] orientation: orientation of CCC in focal plane (lsst.afw.cameraGeom.Orientation)
    @param[in] tryDuplicateAmpNames: create 2 amps with the same name (should result in an error)
    @param[in] tryBadCameraSys: add a transform for an unsupported coord. system (should result in an error)
    """
    def __init__(self,
        name = "detector 1",
        detType = SCIENCE,
        serial = "xkcd722",
        bbox = None,    # do not use mutable objects as defaults
        numAmps = 3,
        pixelSize = afwGeom.Extent2D(0.02),
        ampExtent = afwGeom.Extent2I(5,6), 
        orientation = Orientation(),
        tryDuplicateAmpNames = False,
        tryBadCameraSys = False,
    ):
        # note that (0., 0.) for the reference position is the center of the first pixel
        self.name = name
        self.type = detType
        self.serial = serial
        if bbox is None:
            bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1024, 1048))
        self.bbox = bbox
        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        self.ampInfo = afwTable.AmpInfoCatalog(schema)
        for i in range(numAmps):
            record = self.ampInfo.addNew()
            ampName = "amp %d" % (i + 1,)
            if i == 1 and tryDuplicateAmpNames:
                ampName = self.ampInfo[0].getName()
            record.setName(ampName)
            record.setBBox(afwGeom.Box2I(afwGeom.Point2I(-1, 1), ampExtent))
            record.setGain(1.71234e3)
            record.setReadNoise(0.521237e2)
            record.setReadoutCorner(afwTable.LL)
            record.setHasRawInfo(False)
        self.orientation = orientation
        self.pixelSize = pixelSize
        self.transMap = {
            FOCAL_PLANE: self.orientation.makePixelFpTransform(self.pixelSize),
            CameraSys(ACTUAL_PIXELS, self.name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        if tryBadCameraSys:
            self.transMap[CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform()
        self.detector = Detector(
            self.name,
            self.type,
            self.serial,
            self.bbox,
            self.ampInfo,
            self.orientation,
            self.pixelSize,
            self.transMap,
        )

class CameraWrapper(object):
    """Construct a simple camera

    Intended for use with unit tests, thus saves some interesting information.

    @param[in] isLsstLike: make repository products with one raw image per amplifier (True)
        or with one raw image per detector (False)
    """
    def makeDetectorConfigs(self, detFile):
        detectors = []
        self.detectorNames = []
        with open(detFile) as fh:
            names = fh.readline().rstrip().lstrip("#").split("|")
            for l in fh:
                els = l.rstrip().split("|")
                detectorProps = dict([(name, el) for name, el in zip(names, els)])
                detectors.append(detectorProps)
                self.detectorNames.append(detectorProps['name'])
        detectorConfigs = []
        self.nDetectors = 0
        for detector in detectors:
            self.nDetectors += 1
            detConfig = DetectorConfig()
            detConfig.name = detector['name']
            detConfig.bbox_x0 = 0
            detConfig.bbox_y0 = 0
            detConfig.bbox_x1 = int(detector['npix_x']) - 1
            detConfig.bbox_y1 = int(detector['npix_y']) - 1
            detConfig.serial = str(detector['serial'])
            detConfig.detectorType = int(detector['detectorType'])
            detConfig.offset_x = float(detector['x'])
            detConfig.offset_y = float(detector['y'])
            detConfig.refpos_x = float(detector['refPixPos_x'])
            detConfig.refpos_y = float(detector['refPixPos_y'])
            detConfig.yawDeg = float(detector['yaw'])
            detConfig.pitchDeg = float(detector['pitch'])
            detConfig.rollDeg = float(detector['roll'])
            detConfig.pixelSize_x = float(detector['pixelSize'])
            detConfig.pixelSize_y = float(detector['pixelSize'])
            detConfig.transposeDetector = False
            detConfig.transformDict.nativeSys = PIXELS.getSysName()
            detectorConfigs.append(detConfig)
        return detectorConfigs

    def makeAmpCatalogs(self, ampFile, isLsstLike=False):
        readoutMap = {'LL':0, 'LR':1, 'UR':2, 'UL':3}
        amps = []
        with open(ampFile) as fh:
            names = fh.readline().rstrip().lstrip("#").split("|")
            for l in fh:
                els = l.rstrip().split("|")
                ampProps = dict([(name, el) for name, el in zip(names, els)])
                amps.append(ampProps)
        ampTablesDict = {}
        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        linThreshKey = schema.addField('linearityThreshold', type=float)
        linMaxKey = schema.addField('linearityMaximum', type=float)
        linUnitsKey = schema.addField('linearityUnits', type=str, size=9)
        self.ampInfo = {}
        for amp in amps:
            if amp['ccd_name'] in ampTablesDict:
                ampCatalog = ampTablesDict[amp['ccd_name']]
                self.ampInfo[amp['ccd_name']]['namps'] += 1
            else:
                ampCatalog = afwTable.AmpInfoCatalog(schema)
                ampTablesDict[amp['ccd_name']] = ampCatalog
                self.ampInfo[amp['ccd_name']] = {'namps':1, 'linInfo':{}}
            record = ampCatalog.addNew()
            bbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['trimmed_xmin']), int(amp['trimmed_ymin'])),
                             afwGeom.Point2I(int(amp['trimmed_xmax']), int(amp['trimmed_ymax'])))
            rawBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['raw_xmin']), int(amp['raw_ymin'])),
                             afwGeom.Point2I(int(amp['raw_xmax']), int(amp['raw_ymax'])))
            rawDataBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['raw_data_xmin']), int(amp['raw_data_ymin'])),
                             afwGeom.Point2I(int(amp['raw_data_xmax']), int(amp['raw_data_ymax'])))
            rawHOverscanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['hoscan_xmin']), int(amp['hoscan_ymin'])),
                             afwGeom.Point2I(int(amp['hoscan_xmax']), int(amp['hoscan_ymax'])))
            rawVOverscanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['voscan_xmin']), int(amp['voscan_ymin'])),
                             afwGeom.Point2I(int(amp['voscan_xmax']), int(amp['voscan_ymax'])))
            rawPrescanBbox = afwGeom.Box2I(afwGeom.Point2I(int(amp['pscan_xmin']), int(amp['pscan_ymin'])),
                             afwGeom.Point2I(int(amp['pscan_xmax']), int(amp['pscan_ymax'])))
            xoffset = int(amp['x_offset'])
            yoffset = int(amp['y_offset'])
            flipx = bool(int(amp['flipx']))
            flipy = bool(int(amp['flipy']))
            readcorner = 'LL'
            if not isLsstLike:
                offext = afwGeom.Extent2I(xoffset, yoffset)
                if flipx:
                    xExt = rawBbox.getDimensions().getX()
                    rawBbox.flipLR(xExt)
                    rawDataBbox.flipLR(xExt)
                    rawHOverscanBbox.flipLR(xExt)
                    rawVOverscanBbox.flipLR(xExt)
                    rawPrescanBbox.flipLR(xExt)
                if flipy:
                    yExt = rawBbox.getDimensions().getY()
                    rawBbox.flipTB(yExt)
                    rawDataBbox.flipTB(yExt)
                    rawHOverscanBbox.flipTB(yExt)
                    rawVOverscanBbox.flipTB(yExt)
                    rawPrescanBbox.flipTB(yExt)
                if not flipx and not flipy:
                    readcorner = 'LL'
                elif flipx and not flipy:
                    readcorner = 'LR'
                elif flipx and flipy:
                    readcorner = 'UR'
                elif not flipx and flipy:
                    readcorner = 'UL'
                else:
                    raise RuntimeError("Couldn't find read corner")

                flipx = False
                flipy = False
                rawBbox.shift(offext)
                rawDataBbox.shift(offext)
                rawHOverscanBbox.shift(offext)
                rawVOverscanBbox.shift(offext)
                rawPrescanBbox.shift(offext)
                xoffset = 0
                yoffset = 0
            offset = afwGeom.Extent2I(xoffset, yoffset)
            record.setBBox(bbox)
            record.setRawXYOffset(offset)
            record.setName(str(amp['name']))
            record.setReadoutCorner(readoutMap[readcorner])
            record.setGain(float(amp['gain']))
            record.setReadNoise(float(amp['readnoise']))
            record.setLinearityCoeffs([float(amp['lin_coeffs']),])
            record.setLinearityType(str(amp['lin_type']))
            record.setHasRawInfo(True)
            record.setRawFlipX(flipx)
            record.setRawFlipY(flipy)
            record.setRawBBox(rawBbox)
            record.setRawDataBBox(rawDataBbox)
            record.setRawHorizontalOverscanBBox(rawHOverscanBbox)
            record.setRawVerticalOverscanBBox(rawVOverscanBbox)
            record.setRawPrescanBBox(rawPrescanBbox)
            record.set(linThreshKey, float(amp['lin_thresh']))
            record.set(linMaxKey, float(amp['lin_max']))
            record.set(linUnitsKey, str(amp['lin_units']))
            #The current schema assumes third order coefficients
            saveCoeffs = (float(amp['lin_coeffs']),)
            saveCoeffs += (numpy.nan, numpy.nan, numpy.nan)
            self.ampInfo[amp['ccd_name']]['linInfo'][amp['name']] = \
            {'lincoeffs':saveCoeffs, 'lintype':str(amp['lin_type']),
             'linthresh':float(amp['lin_thresh']), 'linmax':float(amp['lin_max']),
             'linunits':str(amp['lin_units'])}
        return ampTablesDict

    def makeTestRepositoryItems(self, isLsstLike=False):
        detFile = os.path.join(eups.productDir("afw"), "tests", "testCameraDetectors.dat")
        detectorConfigs = self.makeDetectorConfigs(detFile)
        ampFile = os.path.join(eups.productDir("afw"), "tests", "testCameraAmps.dat")
        ampCatalogDict = self.makeAmpCatalogs(ampFile, isLsstLike=isLsstLike)
        camConfig = CameraConfig()
        camConfig.name = "testCamera%s"%('LSST' if isLsstLike else 'SC')
        camConfig.detectorList = dict([(i,detectorConfigs[i]) for i in xrange(len(detectorConfigs))])
        plateScale = 20. #arcsec/mm
        camConfig.plateScale = plateScale
        pScaleRad = afwGeom.arcsecToRad(plateScale)
        #This matches what Dave M. has measured for an LSST like system.
        radialDistortCoeffs = [0.0, 1.0/pScaleRad, 0., 0.925/pScaleRad]
        tConfig = afwGeom.TransformConfig()
        tConfig.transform.name = 'radial'
        tConfig.transform.active.coeffs = radialDistortCoeffs
        tmc = afwGeom.TransformMapConfig()
        tmc.nativeSys = FOCAL_PLANE.getSysName()
        tmc.transforms = {PUPIL.getSysName():tConfig}
        camConfig.transformDict = tmc
        return camConfig, ampCatalogDict 

    def __init__(self, isLsstLike):
        self.camConfig, self.ampCatalogDict = self.makeTestRepositoryItems(isLsstLike)
        cameraTask = CameraFactoryTask()
        self.camera = cameraTask.runCatDict(self.camConfig, self.ampCatalogDict)
