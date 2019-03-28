# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["DetectorWrapper", "CameraWrapper"]

import os

import numpy as np

import lsst.utils
import lsst.geom
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.utils.tests import inTestCase
from .cameraGeomLib import PIXELS, TAN_PIXELS, FIELD_ANGLE, FOCAL_PLANE, SCIENCE, ACTUAL_PIXELS, \
    CameraSys, Detector, Orientation, Amplifier, ReadoutCorner
from .cameraConfig import DetectorConfig, CameraConfig
from .cameraFactory import makeCameraFromCatalogs
from .makePixelToTanPixel import makePixelToTanPixel
from .transformConfig import TransformMapConfig


class DetectorWrapper:
    """A Detector and the data used to construct it

    Intended for use with unit tests, thus saves a copy of all input parameters.
    Does not support setting details of amplifiers.

    Parameters
    ----------
    name : `str` (optional)
        Detector name.
    id : `int` (optional)
        Detector ID.
    detType : `lsst.afw.cameraGeom.DetectorType` (optional)
        Detector type.
    serial : `str` (optional)
        Serial "number".
    bbox : `lsst.geom.Box2I` (optional)
        Bounding box; defaults to (0, 0), (1024x1024).
    numAmps : `int` (optional)
        Number of amplifiers.
    pixelSize : `lsst.geom.Point2D` (optional)
        Pixel size (mm).
    ampExtent : `lsst.geom.Extent2I` (optional)
        Dimensions of amplifier image bbox.
    orientation : `lsst.afw.cameraGeom.Orientation` (optional)
        Orientation of CCC in focal plane.
    plateScale : `float` (optional)
        Plate scale in arcsec/mm; 20.0 is for LSST.
    radialDistortion : `float` (optional)
        Radial distortion, in mm/rad^2.
        The r^3 coefficient of the radial distortion polynomial
        that converts FIELD_ANGLE in radians to FOCAL_PLANE in mm;
        0.925 is the value Dave Monet measured for lsstSim data
    crosstalk : `iterable` (optional)
        Crosstalk coefficient matrix. If None, then no crosstalk correction
        can be performed.
    modFunc : `callable` (optional)
        A function that can modify attributes just before constructing the
        detector; modFunc receives one argument: a DetectorWrapper with all
        attributes except detector set.
    physicalType : `str` (optional)
        The physical type of the device, e.g. CCD, E2V, HgCdTe
    """

    def __init__(self,
                 name="detector 1",
                 id=1,
                 detType=SCIENCE,
                 serial="xkcd722",
                 bbox=None,    # do not use mutable objects as defaults
                 numAmps=3,
                 pixelSize=(0.02, 0.02),
                 ampExtent=(5, 6),
                 orientation=Orientation(),
                 plateScale=20.0,
                 radialDistortion=0.925,
                 crosstalk=None,
                 modFunc=None,
                 physicalType="CCD",
                 ):
        # note that (0., 0.) for the reference position is the center of the
        # first pixel
        self.name = name
        self.id = int(id)
        self.type = detType
        self.serial = serial
        if bbox is None:
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(1024, 1048))
        self.bbox = bbox
        self.pixelSize = lsst.geom.Extent2D(*pixelSize)
        self.ampExtent = lsst.geom.Extent2I(*ampExtent)
        self.plateScale = float(plateScale)
        self.radialDistortion = float(radialDistortion)
        self.ampInfo = []
        for i in range(numAmps):
            amp = Amplifier()
            ampName = "amp %d" % (i + 1,)
            amp.setName(ampName)
            amp.setBBox(lsst.geom.Box2I(lsst.geom.Point2I(-1, 1), self.ampExtent))
            amp.setGain(1.71234e3)
            amp.setReadNoise(0.521237e2)
            amp.setReadoutCorner(ReadoutCorner.LL)
            self.ampInfo.append(amp)
        self.orientation = orientation

        # compute TAN_PIXELS transform
        pScaleRad = lsst.geom.arcsecToRad(self.plateScale)
        radialDistortCoeffs = [0.0, 1.0/pScaleRad,
                               0.0, self.radialDistortion/pScaleRad]
        focalPlaneToField = afwGeom.makeRadialTransform(radialDistortCoeffs)
        pixelToTanPixel = makePixelToTanPixel(
            bbox=self.bbox,
            orientation=self.orientation,
            focalPlaneToField=focalPlaneToField,
            pixelSizeMm=self.pixelSize,
        )

        self.transMap = {
            FOCAL_PLANE: self.orientation.makePixelFpTransform(self.pixelSize),
            CameraSys(TAN_PIXELS, self.name): pixelToTanPixel,
            CameraSys(ACTUAL_PIXELS, self.name): afwGeom.makeRadialTransform([0, 0.95, 0.01]),
        }
        if crosstalk is None:
            crosstalk = [[0.0 for _ in range(numAmps)] for _ in range(numAmps)]
        self.crosstalk = crosstalk
        self.physicalType = physicalType
        if modFunc:
            modFunc(self)
        self.detector = Detector(
            self.name,
            self.id,
            self.type,
            self.serial,
            self.bbox,
            self.ampInfo,
            self.orientation,
            self.pixelSize,
            self.transMap,
            np.array(self.crosstalk, dtype=np.float32),
            self.physicalType,
        )


class CameraWrapper:
    """A simple Camera and the data used to construct it

    Intended for use with unit tests, thus saves some interesting information.

    Parameters
    ----------
    plateScale : `float`
        Plate scale in arcsec/mm; 20.0 is for LSST.
    radialDistortion : `float`
        Radial distortion, in mm/rad^2.
        The r^3 coefficient of the radial distortion polynomial
        that converts FIELD_ANGLE in radians to FOCAL_PLANE in mm;
        0.925 is the value Dave Monet measured for lsstSim data.
    isLsstLike : `bool`.
        Make repository products with one raw image per amplifier (True)
        or with one raw image per detector (False).
    """

    def __init__(self, plateScale=20.0, radialDistortion=0.925, isLsstLike=False):
        afwDir = lsst.utils.getPackageDir("afw")
        self._afwTestDataDir = os.path.join(afwDir, "python", "lsst", "afw",
                                            "cameraGeom", "testData")

        # Info to store for unit tests
        self.plateScale = float(plateScale)
        self.radialDistortion = float(radialDistortion)
        self.detectorNameList = []
        self.detectorIdList = []
        self.ampInfoDict = {}

        self.camConfig, self.ampCatalogDict = self.makeTestRepositoryItems(
            isLsstLike)
        self.camera = makeCameraFromCatalogs(
            self.camConfig, self.ampCatalogDict)

    @property
    def nDetectors(self):
        """Return the number of detectors"""
        return len(self.detectorNameList)

    def makeDetectorConfigs(self, detFile):
        """Construct a list of DetectorConfig, one per detector
        """
        detectors = []
        self.detectorNameList = []
        self.detectorIdList = []
        with open(detFile) as fh:
            names = fh.readline().rstrip().lstrip("#").split("|")
            for l in fh:
                els = l.rstrip().split("|")
                detectorProps = dict([(name, el)
                                      for name, el in zip(names, els)])
                detectors.append(detectorProps)
        detectorConfigs = []
        for i, detector in enumerate(detectors):
            detectorId = (i + 1) * 10  # to avoid simple 0, 1, 2...
            detectorName = detector['name']
            detConfig = DetectorConfig()
            detConfig.name = detectorName
            detConfig.id = detectorId
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
            self.detectorNameList.append(detectorName)
            self.detectorIdList.append(detectorId)
        return detectorConfigs

    def makeAmpCatalogs(self, ampFile, isLsstLike=False):
        """Construct a dict of AmpInfoCatalog, one per detector.

        Parameters
        ----------
        ampFile : `str`
            Path to amplifier data file.
        isLsstLike : `bool`
            If True then there is one raw image per amplifier;
            if False then there is one raw image per detector.
        """
        readoutMap = {
            'LL': afwTable.ReadoutCorner.LL,
            'LR': afwTable.ReadoutCorner.LR,
            'UR': afwTable.ReadoutCorner.UR,
            'UL': afwTable.ReadoutCorner.UL,
        }
        amps = []
        with open(ampFile) as fh:
            names = fh.readline().rstrip().lstrip("#").split("|")
            for l in fh:
                els = l.rstrip().split("|")
                ampProps = dict([(name, el) for name, el in zip(names, els)])
                amps.append(ampProps)
        ampTablesDict = {}
        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        linThreshKey = schema.addField('linearityThreshold', type=np.float64)
        linMaxKey = schema.addField('linearityMaximum', type=np.float64)
        linUnitsKey = schema.addField('linearityUnits', type=str, size=9)
        self.ampInfoDict = {}
        for amp in amps:
            if amp['ccd_name'] in ampTablesDict:
                ampCatalog = ampTablesDict[amp['ccd_name']]
                self.ampInfoDict[amp['ccd_name']]['namps'] += 1
            else:
                ampCatalog = afwTable.AmpInfoCatalog(schema)
                ampTablesDict[amp['ccd_name']] = ampCatalog
                self.ampInfoDict[amp['ccd_name']] = {'namps': 1, 'linInfo': {}}
            record = ampCatalog.addNew()
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(int(amp['trimmed_xmin']),
                                                     int(amp['trimmed_ymin'])),
                                   lsst.geom.Point2I(int(amp['trimmed_xmax']),
                                                     int(amp['trimmed_ymax'])))
            rawBbox = lsst.geom.Box2I(lsst.geom.Point2I(int(amp['raw_xmin']),
                                                        int(amp['raw_ymin'])),
                                      lsst.geom.Point2I(int(amp['raw_xmax']),
                                                        int(amp['raw_ymax'])))
            rawDataBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(amp['raw_data_xmin']),
                                  int(amp['raw_data_ymin'])),
                lsst.geom.Point2I(int(amp['raw_data_xmax']),
                                  int(amp['raw_data_ymax'])))
            rawHOverscanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(amp['hoscan_xmin']),
                                  int(amp['hoscan_ymin'])),
                lsst.geom.Point2I(int(amp['hoscan_xmax']),
                                  int(amp['hoscan_ymax'])))
            rawVOverscanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(amp['voscan_xmin']),
                                  int(amp['voscan_ymin'])),
                lsst.geom.Point2I(int(amp['voscan_xmax']),
                                  int(amp['voscan_ymax'])))
            rawPrescanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(amp['pscan_xmin']),
                                  int(amp['pscan_ymin'])),
                lsst.geom.Point2I(int(amp['pscan_xmax']),
                                  int(amp['pscan_ymax'])))
            xoffset = int(amp['x_offset'])
            yoffset = int(amp['y_offset'])
            flipx = bool(int(amp['flipx']))
            flipy = bool(int(amp['flipy']))
            readcorner = 'LL'
            if not isLsstLike:
                offext = lsst.geom.Extent2I(xoffset, yoffset)
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
            offset = lsst.geom.Extent2I(xoffset, yoffset)
            record.setBBox(bbox)
            record.setRawXYOffset(offset)
            record.setName(str(amp['name']))
            record.setReadoutCorner(readoutMap[readcorner])
            record.setGain(float(amp['gain']))
            record.setReadNoise(float(amp['readnoise']))
            record.setLinearityCoeffs([float(amp['lin_coeffs']), ])
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
            # The current schema assumes third order coefficients
            saveCoeffs = (float(amp['lin_coeffs']),)
            saveCoeffs += (np.nan, np.nan, np.nan)
            self.ampInfoDict[amp['ccd_name']]['linInfo'][amp['name']] = \
                {'lincoeffs': saveCoeffs, 'lintype': str(amp['lin_type']),
                 'linthresh': float(amp['lin_thresh']), 'linmax': float(amp['lin_max']),
                 'linunits': str(amp['lin_units'])}
        return ampTablesDict

    def makeTestRepositoryItems(self, isLsstLike=False):
        """Make camera config and amp catalog dictionary, using default
        detector and amp files.

        Parameters
        ----------
        isLsstLike : `bool`
            If True then there is one raw image per amplifier;
            if False then there is one raw image per detector.
        """
        detFile = os.path.join(self._afwTestDataDir, "testCameraDetectors.dat")
        detectorConfigs = self.makeDetectorConfigs(detFile)
        ampFile = os.path.join(self._afwTestDataDir, "testCameraAmps.dat")
        ampCatalogDict = self.makeAmpCatalogs(ampFile, isLsstLike=isLsstLike)
        camConfig = CameraConfig()
        camConfig.name = "testCamera%s"%('LSST' if isLsstLike else 'SC')
        camConfig.detectorList = dict((i, detConfig)
                                      for i, detConfig in enumerate(detectorConfigs))
        camConfig.plateScale = self.plateScale
        pScaleRad = lsst.geom.arcsecToRad(self.plateScale)
        radialDistortCoeffs = [0.0, 1.0/pScaleRad,
                               0.0, self.radialDistortion/pScaleRad]
        tConfig = afwGeom.TransformConfig()
        tConfig.transform.name = 'inverted'
        radialClass = afwGeom.transformRegistry['radial']
        tConfig.transform.active.transform.retarget(radialClass)
        tConfig.transform.active.transform.coeffs = radialDistortCoeffs
        tmc = TransformMapConfig()
        tmc.nativeSys = FOCAL_PLANE.getSysName()
        tmc.transforms = {FIELD_ANGLE.getSysName(): tConfig}
        camConfig.transformDict = tmc
        return camConfig, ampCatalogDict


@inTestCase
def compare2DFunctions(self, func1, func2, minVal=-10, maxVal=None, nVal=5):
    """Compare two Point2D(Point2D) functions by evaluating them over a
    range of values.
    """
    if maxVal is None:
        maxVal = -minVal
    dVal = (maxVal - minVal) / (nVal - 1)
    for xInd in range(nVal):
        x = minVal + (xInd * dVal)
        for yInd in range(nVal):
            y = minVal + (yInd * dVal)
            fromPoint = lsst.geom.Point2D(x, y)
            res1 = func1(fromPoint)
            res2 = func2(fromPoint)
            self.assertPairsAlmostEqual(res1, res2)


@inTestCase
def assertTransformMapsEqual(self, map1, map2, **kwds):
    """Compare two TransformMaps.
    """
    self.assertEqual(list(map1), list(map2))  # compares the sets of CameraSys
    for sysFrom in map1:
        for sysTo in map1:
            with self.subTest(sysFrom=sysFrom, sysTo=sysTo):
                transform1 = map1.getTransform(sysFrom, sysTo)
                transform2 = map2.getTransform(sysFrom, sysTo)
                self.compare2DFunctions(transform1.applyForward, transform2.applyForward, **kwds)
                self.compare2DFunctions(transform1.applyInverse, transform2.applyInverse, **kwds)


@inTestCase
def assertDetectorsEqual(self, detector1, detector2, **kwds):
    """Compare two Detectors.
    """
    self.assertEqual(detector1.getName(), detector2.getName())
    self.assertEqual(detector1.getId(), detector2.getId())
    self.assertEqual(detector1.getSerial(), detector2.getSerial())
    self.assertEqual(detector1.getPhysicalType(), detector2.getPhysicalType())
    self.assertEqual(detector1.getBBox(), detector2.getBBox())
    self.assertEqual(detector1.getPixelSize(), detector2.getPixelSize())
    orientationIn = detector1.getOrientation()
    orientationOut = detector2.getOrientation()
    self.assertEqual(orientationIn.getFpPosition(), orientationOut.getFpPosition())
    self.assertEqual(orientationIn.getReferencePoint(), orientationOut.getReferencePoint())
    self.assertEqual(orientationIn.getYaw(), orientationOut.getYaw())
    self.assertEqual(orientationIn.getPitch(), orientationOut.getPitch())
    self.assertEqual(orientationIn.getRoll(), orientationOut.getRoll())
    self.assertFloatsEqual(detector1.getCrosstalk(), detector2.getCrosstalk())
    self.assertTransformMapsEqual(detector1.getTransformMap(), detector2.getTransformMap(), **kwds)


@inTestCase
def assertDetectorCollectionsEqual(self, collection1, collection2, **kwds):
    """Compare two DetectorCollections.
    """
    self.assertCountEqual(list(collection1.getNameIter()), list(collection2.getNameIter()))
    for k in collection1.getNameIter():
        self.assertDetectorsEqual(collection1[k], collection2[k], **kwds)


@inTestCase
def assertCamerasEqual(self, camera1, camera2, **kwds):
    """Compare two Camers.
    """
    self.assertDetectorCollectionsEqual(camera1, camera2, **kwds)
    self.assertTransformMapsEqual(camera1.getTransformMap(), camera2.getTransformMap())
    self.assertEqual(camera1.getName(), camera2.getName())
    self.assertEqual(camera1.getPupilFactoryName(), camera2.getPupilFactoryName())
