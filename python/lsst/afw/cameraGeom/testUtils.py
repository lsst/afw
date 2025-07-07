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
from lsst.utils.tests import inTestCase
from ._cameraGeom import CameraSys, PIXELS, TAN_PIXELS, FIELD_ANGLE, FOCAL_PLANE, ACTUAL_PIXELS, Orientation
from ._cameraGeom import Amplifier, ReadoutCorner
from ._camera import Camera
from ._cameraGeom import DetectorType
from .cameraConfig import DetectorConfig, CameraConfig
from ._cameraFactory import makeCameraFromAmpLists
from ._makePixelToTanPixel import makePixelToTanPixel
from ._transformConfig import TransformMapConfig


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
                 detType=DetectorType.SCIENCE,
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
                 cameraBuilder=None
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
        self.orientation = orientation
        self.radialDistortion = float(radialDistortion)

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
        tanPixelSys = CameraSys(TAN_PIXELS, self.name)
        actualPixelSys = CameraSys(ACTUAL_PIXELS, self.name)
        self.transMap = {
            FOCAL_PLANE: self.orientation.makePixelFpTransform(self.pixelSize),
            tanPixelSys: pixelToTanPixel,
            actualPixelSys: afwGeom.makeRadialTransform([0, 0.95, 0.01]),
        }
        if crosstalk is None:
            crosstalk = [[0.0 for _ in range(numAmps)] for _ in range(numAmps)]
        self.crosstalk = crosstalk
        self.physicalType = physicalType
        if cameraBuilder is None:
            cameraBuilder = Camera.Builder("CameraForDetectorWrapper")
        self.ampList = []
        for i in range(numAmps):
            ampBuilder = Amplifier.Builder()
            ampName = f"amp {i + 1}"
            ampBuilder.setName(ampName)
            ampBuilder.setBBox(lsst.geom.Box2I(lsst.geom.Point2I(-1, 1), self.ampExtent))
            ampBuilder.setGain(1.71234e3)
            ampBuilder.setReadNoise(0.521237e2)
            ampBuilder.setReadoutCorner(ReadoutCorner.LL)
            self.ampList.append(ampBuilder)
        if modFunc:
            modFunc(self)
        detectorBuilder = cameraBuilder.add(self.name, self.id)
        detectorBuilder.setType(self.type)
        detectorBuilder.setSerial(self.serial)
        detectorBuilder.setPhysicalType(self.physicalType)
        detectorBuilder.setBBox(self.bbox)
        detectorBuilder.setOrientation(self.orientation)
        detectorBuilder.setPixelSize(self.pixelSize)
        detectorBuilder.setTransformFromPixelsTo(tanPixelSys, self.transMap[tanPixelSys])
        detectorBuilder.setTransformFromPixelsTo(actualPixelSys, self.transMap[actualPixelSys])
        detectorBuilder.setCrosstalk(np.array(self.crosstalk, dtype=np.float32))
        for ampBuilder in self.ampList:
            detectorBuilder.append(ampBuilder)
        camera = cameraBuilder.finish()
        self.detector = camera[self.name]


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
    focalPlaneParity : `bool`
        If `True`, the X axis is flipped between the FOCAL_PLANE and
        FIELD_ANGLE coordinate systems.
    """

    def __init__(self, plateScale=20.0, radialDistortion=0.925, isLsstLike=False, focalPlaneParity=False):
        afwDir = lsst.utils.getPackageDir("afw")
        self._afwTestDataDir = os.path.join(afwDir, "python", "lsst", "afw",
                                            "cameraGeom", "testData")

        # Info to store for unit tests
        self.plateScale = float(plateScale)
        self.radialDistortion = float(radialDistortion)
        self.detectorNameList = []
        self.detectorIdList = []
        self.ampDataDict = {}   # ampData[Dict]: raw dictionaries of test data fields

        # ampList[Dict]: actual cameraGeom.Amplifier objects
        self.camConfig, self.ampListDict = self.makeTestRepositoryItems(
            isLsstLike, focalPlaneParity=focalPlaneParity)
        self.camera = makeCameraFromAmpLists(
            self.camConfig, self.ampListDict)

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
            for line in fh:
                els = line.rstrip().split("|")
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
            detConfig.offset_z = float(detector['z'])
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

    def makeAmpLists(self, ampFile, isLsstLike=False):
        """Construct a dict of list of Amplifer, one list per detector.

        Parameters
        ----------
        ampFile : `str`
            Path to amplifier data file.
        isLsstLike : `bool`
            If True then there is one raw image per amplifier;
            if False then there is one raw image per detector.
        """
        readoutMap = {
            'LL': ReadoutCorner.LL,
            'LR': ReadoutCorner.LR,
            'UR': ReadoutCorner.UR,
            'UL': ReadoutCorner.UL,
        }
        ampDataList = []
        with open(ampFile) as fh:
            names = fh.readline().rstrip().lstrip("#").split("|")
            for line in fh:
                els = line.rstrip().split("|")
                ampProps = dict([(name, el) for name, el in zip(names, els)])
                ampDataList.append(ampProps)
        ampListDict = {}
        self.ampDataDict = {}
        for ampData in ampDataList:
            if ampData['ccd_name'] in ampListDict:
                ampList = ampListDict[ampData['ccd_name']]
                self.ampDataDict[ampData['ccd_name']]['namps'] += 1
            else:
                ampList = []
                ampListDict[ampData['ccd_name']] = ampList
                self.ampDataDict[ampData['ccd_name']] = {'namps': 1, 'linInfo': {}}
            builder = Amplifier.Builder()
            bbox = lsst.geom.Box2I(lsst.geom.Point2I(int(ampData['trimmed_xmin']),
                                                     int(ampData['trimmed_ymin'])),
                                   lsst.geom.Point2I(int(ampData['trimmed_xmax']),
                                                     int(ampData['trimmed_ymax'])))
            rawBbox = lsst.geom.Box2I(lsst.geom.Point2I(int(ampData['raw_xmin']),
                                                        int(ampData['raw_ymin'])),
                                      lsst.geom.Point2I(int(ampData['raw_xmax']),
                                                        int(ampData['raw_ymax'])))
            rawDataBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(ampData['raw_data_xmin']),
                                  int(ampData['raw_data_ymin'])),
                lsst.geom.Point2I(int(ampData['raw_data_xmax']),
                                  int(ampData['raw_data_ymax'])))
            rawHOverscanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(ampData['hoscan_xmin']),
                                  int(ampData['hoscan_ymin'])),
                lsst.geom.Point2I(int(ampData['hoscan_xmax']),
                                  int(ampData['hoscan_ymax'])))
            rawVOverscanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(ampData['voscan_xmin']),
                                  int(ampData['voscan_ymin'])),
                lsst.geom.Point2I(int(ampData['voscan_xmax']),
                                  int(ampData['voscan_ymax'])))
            rawPrescanBbox = lsst.geom.Box2I(
                lsst.geom.Point2I(int(ampData['pscan_xmin']),
                                  int(ampData['pscan_ymin'])),
                lsst.geom.Point2I(int(ampData['pscan_xmax']),
                                  int(ampData['pscan_ymax'])))
            xoffset = int(ampData['x_offset'])
            yoffset = int(ampData['y_offset'])
            flipx = bool(int(ampData['flipx']))
            flipy = bool(int(ampData['flipy']))
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
            builder.setBBox(bbox)
            builder.setRawXYOffset(offset)
            builder.setName(str(ampData['name']))
            builder.setReadoutCorner(readoutMap[readcorner])
            builder.setGain(float(ampData['gain']))
            builder.setReadNoise(float(ampData['readnoise']))
            linCoeffs = np.array([float(ampData['lin_coeffs']), ], dtype=float)
            builder.setLinearityCoeffs(linCoeffs)
            builder.setLinearityType(str(ampData['lin_type']))
            builder.setRawFlipX(flipx)
            builder.setRawFlipY(flipy)
            builder.setRawBBox(rawBbox)
            builder.setRawDataBBox(rawDataBbox)
            builder.setRawHorizontalOverscanBBox(rawHOverscanBbox)
            builder.setRawVerticalOverscanBBox(rawVOverscanBbox)
            builder.setRawPrescanBBox(rawPrescanBbox)
            builder.setLinearityThreshold(float(ampData['lin_thresh']))
            builder.setLinearityMaximum(float(ampData['lin_max']))
            builder.setLinearityUnits(str(ampData['lin_units']))
            self.ampDataDict[ampData['ccd_name']]['linInfo'][ampData['name']] = \
                {'lincoeffs': linCoeffs, 'lintype': str(ampData['lin_type']),
                 'linthresh': float(ampData['lin_thresh']), 'linmax': float(ampData['lin_max']),
                 'linunits': str(ampData['lin_units'])}
            ampList.append(builder)
        return ampListDict

    def makeTestRepositoryItems(self, isLsstLike=False, focalPlaneParity=False):
        """Make camera config and amp catalog dictionary, using default
        detector and amp files.

        Parameters
        ----------
        isLsstLike : `bool`
            If True then there is one raw image per amplifier;
            if False then there is one raw image per detector.
        focalPlaneParity : `bool`
            If `True`, the X axis is flipped between the FOCAL_PLANE and
            FIELD_ANGLE coordinate systems.
        """
        detFile = os.path.join(self._afwTestDataDir, "testCameraDetectors.dat")
        detectorConfigs = self.makeDetectorConfigs(detFile)
        ampFile = os.path.join(self._afwTestDataDir, "testCameraAmps.dat")
        ampListDict = self.makeAmpLists(ampFile, isLsstLike=isLsstLike)
        camConfig = CameraConfig()
        camConfig.name = "testCamera%s"%('LSST' if isLsstLike else 'SC')
        camConfig.detectorList = dict((i, detConfig)
                                      for i, detConfig in enumerate(detectorConfigs))
        camConfig.plateScale = self.plateScale
        camConfig.focalPlaneParity = focalPlaneParity
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
        return camConfig, ampListDict


@inTestCase
def compare2DFunctions(self, func1, func2, minVal=-10, maxVal=None, nVal=5):
    """Compare two Point2D(list(Point2D)) functions by evaluating them over a
    range of values.

    Notes
    -----
    Assumes the functions can be called with ``list[Point2D]`` and return
    ``list[Point2D]``.
    """
    if maxVal is None:
        maxVal = -minVal
    dVal = (maxVal - minVal) / (nVal - 1)
    points = []
    for xInd in range(nVal):
        x = minVal + (xInd * dVal)
        for yInd in range(nVal):
            y = minVal + (yInd * dVal)
            fromPoint = lsst.geom.Point2D(x, y)
            points.append(fromPoint)

    vres1 = func1(points)
    vres2 = func2(points)
    for res1, res2 in zip(vres1, vres2):
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
def assertAmplifiersEqual(self, amp1, amp2):
    self.assertEqual(amp1.getName(), amp2.getName())
    self.assertEqual(amp1.getBBox(), amp2.getBBox())
    self.assertFloatsEqual(amp1.getGain(), amp2.getGain(), ignoreNaNs=True)
    self.assertFloatsEqual(amp1.getReadNoise(), amp2.getReadNoise(), ignoreNaNs=True)
    self.assertFloatsEqual(amp1.getSaturation(), amp2.getSaturation(), ignoreNaNs=True)
    self.assertEqual(amp1.getReadoutCorner(), amp2.getReadoutCorner())
    self.assertFloatsEqual(amp1.getSuspectLevel(), amp2.getSuspectLevel(), ignoreNaNs=True)
    self.assertEqual(amp1.getLinearityCoeffs().shape, amp2.getLinearityCoeffs().shape)
    self.assertFloatsEqual(amp1.getLinearityCoeffs(), amp2.getLinearityCoeffs(), ignoreNaNs=True)
    self.assertEqual(amp1.getLinearityType(), amp2.getLinearityType())
    self.assertFloatsEqual(amp1.getLinearityThreshold(), amp2.getLinearityThreshold(), ignoreNaNs=True)
    self.assertFloatsEqual(amp1.getLinearityMaximum(), amp2.getLinearityMaximum(), ignoreNaNs=True)
    self.assertEqual(amp1.getLinearityUnits(), amp2.getLinearityUnits())
    self.assertEqual(amp1.getRawBBox(), amp2.getRawBBox())
    self.assertEqual(amp1.getRawDataBBox(), amp2.getRawDataBBox())
    self.assertEqual(amp1.getRawFlipX(), amp2.getRawFlipX())
    self.assertEqual(amp1.getRawFlipY(), amp2.getRawFlipY())
    self.assertEqual(amp1.getRawHorizontalOverscanBBox(), amp2.getRawHorizontalOverscanBBox())
    self.assertEqual(amp1.getRawVerticalOverscanBBox(), amp2.getRawVerticalOverscanBBox())
    self.assertEqual(amp1.getRawPrescanBBox(), amp2.getRawPrescanBBox())


@inTestCase
def assertDetectorsEqual(self, detector1, detector2, *, compareTransforms=True, **kwds):
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
    if compareTransforms:
        self.assertTransformMapsEqual(detector1.getTransformMap(), detector2.getTransformMap(), **kwds)
    self.assertEqual(len(detector1.getAmplifiers()), len(detector2.getAmplifiers()))
    for amp1, amp2 in zip(detector1.getAmplifiers(), detector2.getAmplifiers()):
        self.assertAmplifiersEqual(amp1, amp2)


@inTestCase
def assertDetectorCollectionsEqual(self, collection1, collection2, **kwds):
    """Compare two DetectorCollections.
    """
    self.assertCountEqual(list(collection1.getNameIter()), list(collection2.getNameIter()))
    for k in collection1.getNameIter():
        self.assertDetectorsEqual(collection1[k], collection2[k], **kwds)


@inTestCase
def assertCamerasEqual(self, camera1, camera2, **kwds):
    """Compare two Cameras.
    """
    self.assertDetectorCollectionsEqual(camera1, camera2, **kwds)
    self.assertTransformMapsEqual(camera1.getTransformMap(), camera2.getTransformMap())
    self.assertEqual(camera1.getName(), camera2.getName())
    self.assertEqual(camera1.getPupilFactoryName(), camera2.getPupilFactoryName())
    self.assertEqual(camera1.getFocalPlaneParity(), camera2.getFocalPlaneParity())
