import unittest

import lsst.utils.tests
from lsst.afw.cameraGeom.testUtils import DetectorWrapper
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.base


class ExposureInfoTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        crpix = lsst.geom.Point2D(100, 100)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        self.wcs = afwGeom.makeSkyWcs(crpix, crval, afwGeom.makeCdMatrix(90.0*lsst.geom.degrees))

        self.psf = afwDetect.GaussianPsf(1, 2, 3)
        self.detector = DetectorWrapper().detector
        self.calib = afwImage.Calib(10)
        self.apCorrMap = afwImage.ApCorrMap()
        self.coaddInputs = afwImage.CoaddInputs()
        box = lsst.geom.Box2D(lsst.geom.Point2D(0, 0), lsst.geom.Point2D(10, 10))
        self.polygon = afwGeom.Polygon(box)
        self.transmissionCurve = afwImage.TransmissionCurve.makeIdentity()
        self.visitInfo = afwImage.VisitInfo()

    def testDefaultConstructor(self):
        """Test that the default constructor sets things to None."""
        exposureInfo = afwImage.ExposureInfo()
        self.assertIsNone(exposureInfo.getWcs())
        self.assertIsNone(exposureInfo.getPsf())
        self.assertIsNone(exposureInfo.getDetector())
        self.assertIsNone(exposureInfo.getValidPolygon())
        self.assertIsNone(exposureInfo.getCoaddInputs())
        self.assertIsNone(exposureInfo.getApCorrMap())
        self.assertIsNone(exposureInfo.getVisitInfo())
        self.assertIsNone(exposureInfo.getTransmissionCurve())

        self.assertEqual(exposureInfo.getCalib(), lsst.afw.image.Calib())

        self.assertEqual(exposureInfo.getFilter().getName(), "_unknown_")

        self.assertEqual(exposureInfo.getMetadata(), lsst.daf.base.PropertyList())

        self.assertFalse(exposureInfo.isSurfaceBrightness)
        self.assertFalse(exposureInfo.isFluence)
        self.assertEqual(exposureInfo.getImagePhotometricCalibrationType(),
                         afwImage.ImagePhotometricCalibrationType.NOTAPPLICABLE)

    def testExposureConstructor(self):
        exposureInfo = afwImage.ExposureInfo(wcs=self.wcs,
                                             psf=self.psf,
                                             calib=self.calib,
                                             detector=self.detector,
                                             polygon=self.polygon,
                                             coaddInputs=self.coaddInputs,
                                             transmissionCurve=self.transmissionCurve
                                             )
        self.assertEqual(exposureInfo.getDetector(), self.detector)
        self.assertEqual(exposureInfo.getWcs(), self.wcs)
        self.assertEqual(exposureInfo.getPsf(), self.psf)
        self.assertEqual(exposureInfo.getCalib(), self.calib)
        self.assertEqual(exposureInfo.getValidPolygon(), self.polygon)
        self.assertEqual(exposureInfo.getCoaddInputs(), self.coaddInputs)
        self.assertEqual(exposureInfo.getTransmissionCurve(), self.transmissionCurve)

    def testExposureInfoSetNone(self):
        """Test that set*(None) works for shared_ptr members."""
        exposureInfo = afwImage.ExposureInfo(wcs=self.wcs,
                                             psf=self.psf,
                                             calib=self.calib,
                                             detector=self.detector,
                                             apCorrMap=self.apCorrMap,
                                             coaddInputs=self.coaddInputs,
                                             polygon=self.polygon,
                                             visitInfo=self.visitInfo,
                                             )

        exposureInfo.setWcs(None)
        self.assertIsNone(exposureInfo.getWcs())

        exposureInfo.setPsf(None)
        self.assertIsNone(exposureInfo.getPsf())

        exposureInfo.setDetector(None)
        self.assertIsNone(exposureInfo.getDetector())

        exposureInfo.setCalib(None)
        self.assertIsNone(exposureInfo.getCalib())

        exposureInfo.setValidPolygon(None)
        self.assertIsNone(exposureInfo.getValidPolygon())

        exposureInfo.setCoaddInputs(None)
        self.assertIsNone(exposureInfo.getCoaddInputs())

        exposureInfo.setVisitInfo(None)
        self.assertIsNone(exposureInfo.getVisitInfo())

        exposureInfo.setApCorrMap(None)
        self.assertIsNone(exposureInfo.getApCorrMap())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
