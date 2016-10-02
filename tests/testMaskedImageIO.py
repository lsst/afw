#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from builtins import object
#pybind11#
#pybind11##
#pybind11## LSST Data Management System
#pybind11## Copyright 2008, 2009, 2010 LSST Corporation.
#pybind11##
#pybind11## This product includes software developed by the
#pybind11## LSST Project (http://www.lsst.org/).
#pybind11##
#pybind11## This program is free software: you can redistribute it and/or modify
#pybind11## it under the terms of the GNU General Public License as published by
#pybind11## the Free Software Foundation, either version 3 of the License, or
#pybind11## (at your option) any later version.
#pybind11##
#pybind11## This program is distributed in the hope that it will be useful,
#pybind11## but WITHOUT ANY WARRANTY; without even the implied warranty of
#pybind11## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#pybind11## GNU General Public License for more details.
#pybind11##
#pybind11## You should have received a copy of the LSST License Statement and
#pybind11## the GNU General Public License along with this program.  If not,
#pybind11## see <http://www.lsstcorp.org/LegalNotices/>.
#pybind11##
#pybind11#
#pybind11#"""
#pybind11#Tests for MaskedImages
#pybind11#
#pybind11#Run with:
#pybind11#   python MaskedImageIO.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import MaskedImageIO; MaskedImageIO.run()
#pybind11#"""
#pybind11#
#pybind11#
#pybind11#import contextlib
#pybind11#import os.path
#pybind11#import unittest
#pybind11#import shutil
#pybind11#import tempfile
#pybind11#
#pybind11#import numpy
#pybind11#import pyfits
#pybind11#
#pybind11#import lsst.utils
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.math as afwMath
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#import lsst.pex.exceptions as pexEx
#pybind11#
#pybind11#try:
#pybind11#    dataDir = lsst.utils.getPackageDir("afwdata")
#pybind11#except pexEx.NotFoundError:
#pybind11#    dataDir = None
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class MaskedImageTestCase(unittest.TestCase):
#pybind11#    """A test case for MaskedImage"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        # Set a (non-standard) initial Mask plane definition
#pybind11#        #
#pybind11#        # Ideally we'd use the standard dictionary and a non-standard file, but
#pybind11#        # a standard file's what we have
#pybind11#        #
#pybind11#        self.Mask = afwImage.MaskU
#pybind11#        mask = self.Mask()
#pybind11#
#pybind11#        # Store the default mask planes for later use
#pybind11#        maskPlaneDict = self.Mask().getMaskPlaneDict()
#pybind11#        self.defaultMaskPlanes = sorted(maskPlaneDict, key=maskPlaneDict.__getitem__)
#pybind11#
#pybind11#        # reset so tests will be deterministic
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in ("ZERO", "BAD", "SAT", "INTRP", "CR", "EDGE"):
#pybind11#            mask.addMaskPlane(p)
#pybind11#
#pybind11#        if dataDir is not None:
#pybind11#            if False:
#pybind11#                self.fileName = os.path.join(dataDir, "Small_MI.fits")
#pybind11#            else:
#pybind11#                self.fileName = os.path.join(dataDir, "CFHT", "D4", "cal-53535-i-797722_1.fits")
#pybind11#            self.mi = afwImage.MaskedImageF(self.fileName)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        if dataDir is not None:
#pybind11#            del self.mi
#pybind11#        # Reset the mask plane to the default
#pybind11#        self.Mask.clearMaskPlaneDict()
#pybind11#        for p in self.defaultMaskPlanes:
#pybind11#            self.Mask.addMaskPlane(p)
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFitsRead(self):
#pybind11#        """Check if we read MaskedImages"""
#pybind11#
#pybind11#        image = self.mi.getImage()
#pybind11#        mask = self.mi.getMask()
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.mi)
#pybind11#
#pybind11#        self.assertEqual(image.get(32, 1), 3728)
#pybind11#        self.assertEqual(mask.get(0, 0), 2)  # == BAD
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFitsReadImage(self):
#pybind11#        """Check if we can read a single-HDU image as a MaskedImage, setting the mask and variance
#pybind11#        planes to zero."""
#pybind11#        filename = os.path.join(dataDir, "data", "small_img.fits")
#pybind11#        image = afwImage.ImageF(filename)
#pybind11#        maskedImage = afwImage.MaskedImageF(filename)
#pybind11#        exposure = afwImage.ExposureF(filename)
#pybind11#        self.assertEqual(image.get(0, 0), maskedImage.getImage().get(0, 0))
#pybind11#        self.assertEqual(image.get(0, 0), exposure.getMaskedImage().getImage().get(0, 0))
#pybind11#        self.assertTrue(numpy.all(maskedImage.getMask().getArray() == 0))
#pybind11#        self.assertTrue(numpy.all(exposure.getMaskedImage().getMask().getArray() == 0))
#pybind11#        self.assertTrue(numpy.all(maskedImage.getVariance().getArray() == 0.0))
#pybind11#        self.assertTrue(numpy.all(exposure.getMaskedImage().getVariance().getArray() == 0.0))
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFitsReadConform(self):
#pybind11#        """Check if we read MaskedImages and make them replace Mask's plane dictionary"""
#pybind11#
#pybind11#        metadata, bbox, conformMasks = None, afwGeom.Box2I(), True
#pybind11#        self.mi = afwImage.MaskedImageF(self.fileName, metadata, bbox, afwImage.LOCAL, conformMasks)
#pybind11#
#pybind11#        image = self.mi.getImage()
#pybind11#        mask = self.mi.getMask()
#pybind11#
#pybind11#        self.assertEqual(image.get(32, 1), 3728)
#pybind11#        self.assertEqual(mask.get(0, 0), 1)  # i.e. not shifted 1 place to the right
#pybind11#
#pybind11#        self.assertEqual(mask.getMaskPlane("CR"), 3, "Plane CR has value specified in FITS file")
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFitsReadNoConform2(self):
#pybind11#        """Check that reading a mask doesn't invalidate the plane dictionary"""
#pybind11#
#pybind11#        testMask = afwImage.MaskU(self.fileName, 3)
#pybind11#
#pybind11#        mask = self.mi.getMask()
#pybind11#        mask |= testMask
#pybind11#
#pybind11#    @unittest.skipIf(dataDir is None, "afwdata not setup")
#pybind11#    def testFitsReadConform2(self):
#pybind11#        """Check that conforming a mask invalidates the plane dictionary"""
#pybind11#
#pybind11#        hdu, metadata, bbox, conformMasks = 3, None, afwGeom.Box2I(), True
#pybind11#        testMask = afwImage.MaskU(self.fileName,
#pybind11#                                  hdu, metadata, bbox, afwImage.LOCAL, conformMasks)
#pybind11#
#pybind11#        mask = self.mi.getMask()
#pybind11#
#pybind11#        def tst(mask=mask):
#pybind11#            mask |= testMask
#pybind11#
#pybind11#        self.assertRaises(pexEx.RuntimeError, tst)
#pybind11#
#pybind11#    def testTicket617(self):
#pybind11#        """Test reading an F64 image and converting it to a MaskedImage"""
#pybind11#        im = afwImage.ImageD(afwGeom.Extent2I(100, 100))
#pybind11#        im.set(666)
#pybind11#        afwImage.MaskedImageD(im)
#pybind11#
#pybind11#    def testReadWriteXY0(self):
#pybind11#        """Test that we read and write (X0, Y0) correctly"""
#pybind11#        im = afwImage.MaskedImageF(afwGeom.Extent2I(10, 20))
#pybind11#
#pybind11#        x0, y0 = 1, 2
#pybind11#        im.setXY0(x0, y0)
#pybind11#        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
#pybind11#            im.writeFits(tmpFile)
#pybind11#
#pybind11#            im2 = im.Factory(tmpFile)
#pybind11#            self.assertEqual(im2.getX0(), x0)
#pybind11#            self.assertEqual(im2.getY0(), y0)
#pybind11#
#pybind11#            self.assertEqual(im2.getImage().getX0(), x0)
#pybind11#            self.assertEqual(im2.getImage().getY0(), y0)
#pybind11#
#pybind11#            self.assertEqual(im2.getMask().getX0(), x0)
#pybind11#            self.assertEqual(im2.getMask().getY0(), y0)
#pybind11#
#pybind11#            self.assertEqual(im2.getVariance().getX0(), x0)
#pybind11#            self.assertEqual(im2.getVariance().getY0(), y0)
#pybind11#
#pybind11#
#pybind11#@contextlib.contextmanager
#pybind11#def tmpFits(*hdus):
#pybind11#    # Given a list of numpy arrays, create a temporary FITS file that
#pybind11#    # contains them as consecutive HDUs. Yield it, then remove it.
#pybind11#    hdus = [pyfits.PrimaryHDU(hdus[0])] + [pyfits.ImageHDU(hdu) for hdu in hdus[1:]]
#pybind11#    hdulist = pyfits.HDUList(hdus)
#pybind11#    tempdir = tempfile.mkdtemp()
#pybind11#    try:
#pybind11#        filename = os.path.join(tempdir, 'test.fits')
#pybind11#        hdulist.writeto(filename)
#pybind11#        yield filename
#pybind11#    finally:
#pybind11#        shutil.rmtree(tempdir)
#pybind11#
#pybind11#
#pybind11#class MultiExtensionTestCase(object):
#pybind11#    """Base class for testing that we correctly read multi-extension FITS files.
#pybind11#
#pybind11#    MEF files may be read to either MaskedImage or Exposure objects. We apply
#pybind11#    the same set of tests to each by subclassing and defining _constructImage
#pybind11#    and _checkImage.
#pybind11#    """
#pybind11#    # When persisting a MaskedImage (or derivative, e.g. Exposure) to FITS, we impose a data
#pybind11#    # model which the combination of the limits of the FITS structure and the desire to maintain
#pybind11#    # backwards compatibility make it hard to express. We attempt to make this as safe as
#pybind11#    # possible by handling the following situations and logging appropriate warnings:
#pybind11#    #
#pybind11#    # Note that Exposures always set needAllHdus to False.
#pybind11#    #
#pybind11#    # 1. If needAllHdus is true:
#pybind11#    #    1.1 If the user has specified a non-default HDU, we throw.
#pybind11#    #    1.2 If the user has not specified an HDU (or has specified one equal to the default):
#pybind11#    #        1.2.1 If any of the image, mask or variance is unreadable (eg because they don't
#pybind11#    #              exist, or they have the wrong data type), we throw.
#pybind11#    #        1.2.2 Otherwise, we return the MaskedImage with image/mask/variance set as
#pybind11#    #              expected.
#pybind11#    # 2. If needAllHdus is false:
#pybind11#    #    2.1 If the user has specified a non-default HDU:
#pybind11#    #        2.1.1 If the user specified HDU is unreadable, we throw.
#pybind11#    #        2.1.2 Otherwise, we return the contents of that HDU as the image and default
#pybind11#    #              (=empty) mask & variance.
#pybind11#    #    2.2 If the user has not specified an HDU, or has specified one equal to the default:
#pybind11#    #        2.2.1 If the default HDU is unreadable, we throw.
#pybind11#    #        2.2.2 Otherwise, we attempt to read both mask and variance from the FITS file,
#pybind11#    #              and return them together with the image. If one or both are unreadable,
#pybind11#    #              we fall back to an empty default for the missing data and return the
#pybind11#    #              remainder..
#pybind11#    #
#pybind11#    # See also the discussion at DM-2599.
#pybind11#
#pybind11#    def _checkMaskedImage(self, mim, width, height, val1, val2, val3):
#pybind11#        # Check that the input image has dimensions width & height and that the image, mask and
#pybind11#        # variance have mean val1, val2 & val3 respectively.
#pybind11#        self.assertEqual(mim.getWidth(), width)
#pybind11#        self.assertEqual(mim.getHeight(), width)
#pybind11#        self.assertEqual(afwMath.makeStatistics(mim.getImage(), afwMath.MEAN).getValue(), val1)
#pybind11#        s = afwMath.makeStatistics(mim.getMask(), afwMath.SUM | afwMath.NPOINT)
#pybind11#        self.assertEqual(float(s.getValue(afwMath.SUM)) / s.getValue(afwMath.NPOINT), val2)
#pybind11#        self.assertEqual(afwMath.makeStatistics(mim.getVariance(), afwMath.MEAN).getValue(), val3)
#pybind11#
#pybind11#    def testUnreadableExtensionAsImage(self):
#pybind11#        # Test for case 2.1.1 above.
#pybind11#        with tmpFits(None, numpy.array([[1]]), numpy.array([[2]], dtype=numpy.int16), None) as fitsfile:
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile, 3)
#pybind11#
#pybind11#    def testReadableExtensionAsImage(self):
#pybind11#        # Test for case 2.1.2 above.
#pybind11#        with tmpFits(None, numpy.array([[1]]), numpy.array([[2]], dtype=numpy.int16),
#pybind11#                     numpy.array([[3]])) as fitsfile:
#pybind11#            self._checkImage(self._constructImage(fitsfile, 3), 1, 1, 3, 0, 0)
#pybind11#
#pybind11#    def testUnreadbleDefaultAsImage(self):
#pybind11#        # Test for case 2.2.1 above.
#pybind11#        with tmpFits(None, None, numpy.array([[2]], dtype=numpy.int16), numpy.array([[3]])) as fitsfile:
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile)
#pybind11#
#pybind11#    def testUnreadbleOptionalExtensions(self):
#pybind11#        # Test for case 2.2.2 above.
#pybind11#        # Unreadable mask.
#pybind11#        with tmpFits(None, numpy.array([[1]]), None, numpy.array([[3]])) as fitsfile:
#pybind11#            self._checkImage(self._constructImage(fitsfile), 1, 1, 1, 0, 3)
#pybind11#        # Unreadable variance.
#pybind11#        with tmpFits(None, numpy.array([[1]]), numpy.array([[2]], dtype=numpy.int16), None) as fitsfile:
#pybind11#            self._checkImage(self._constructImage(fitsfile, needAllHdus=False), 1, 1, 1, 2, 0)
#pybind11#
#pybind11#
#pybind11#class MaskedMultiExtensionTestCase(MultiExtensionTestCase, lsst.utils.tests.TestCase):
#pybind11#    """Derived version of MultiExtensionTestCase for MaskedImages."""
#pybind11#
#pybind11#    def _constructImage(self, filename, hdu=None, needAllHdus=False):
#pybind11#        # Construct an instance of MaskedImageF by loading from filename. If hdu
#pybind11#        # is specified, load that HDU specifically. Pass through needAllHdus
#pybind11#        # to the MaskedImageF constructor.  This function exists only to stub
#pybind11#        # default arguments into the constructor for parameters which we are
#pybind11#        # not exercising in this test.
#pybind11#        if hdu:
#pybind11#            filename = "%s[%d]" % (filename, hdu)
#pybind11#        return afwImage.MaskedImageF(filename, None, afwGeom.Box2I(), afwImage.PARENT, False, needAllHdus)
#pybind11#
#pybind11#    def _checkImage(self, *args, **kwargs):
#pybind11#        self._checkMaskedImage(*args, **kwargs)
#pybind11#
#pybind11#    def testNeedAllHdus(self):
#pybind11#        # Tests for cases 1.1 & 1.2.2 above.
#pybind11#        # We'll regard it as ok for the user to specify any of:
#pybind11#        # * No HDU;
#pybind11#        # * The "zeroeth" (primary) HDU;
#pybind11#        # * The first (first extension) HDU.
#pybind11#        # Any others should raise when needAllHdus is true
#pybind11#        with tmpFits(None, numpy.array([[1]]), numpy.array([[2]], dtype=numpy.int16),
#pybind11#                     numpy.array([[3]])) as fitsfile:
#pybind11#            # No HDU specified -> ok.
#pybind11#            self._checkImage(self._constructImage(fitsfile, needAllHdus=True), 1, 1, 1, 2, 3)
#pybind11#            # First HDU -> ok.
#pybind11#            self._checkImage(self._constructImage(fitsfile, 0, needAllHdus=True), 1, 1, 1, 2, 3)
#pybind11#            # First HDU -> ok.
#pybind11#            self._checkImage(self._constructImage(fitsfile, 1, needAllHdus=True), 1, 1, 1, 2, 3)
#pybind11#            # Second HDU -> raises.
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile, 2, needAllHdus=True)
#pybind11#
#pybind11#    def testUnreadableImage(self):
#pybind11#        # Test for case 1.2.1 above.
#pybind11#        with tmpFits(None, None, numpy.array([[2]], dtype=numpy.int16), numpy.array([[3]])) as fitsfile:
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile, None, needAllHdus=True)
#pybind11#
#pybind11#    def testUnreadableMask(self):
#pybind11#        # Test for case 1.2.1 above.
#pybind11#        with tmpFits(None, numpy.array([[1]]), None, numpy.array([[3]])) as fitsfile:
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile, None, needAllHdus=True)
#pybind11#
#pybind11#    def testUnreadableVariance(self):
#pybind11#        # Test for case 1.2.1 above.
#pybind11#        with tmpFits(None, numpy.array([[1]]), numpy.array([[2]], dtype=numpy.int16), None) as fitsfile:
#pybind11#            self.assertRaises(Exception, self._constructImage, fitsfile, None, needAllHdus=True)
#pybind11#
#pybind11#
#pybind11#class ExposureMultiExtensionTestCase(MultiExtensionTestCase, lsst.utils.tests.TestCase):
#pybind11#    """Derived version of MultiExtensionTestCase for Exposures."""
#pybind11#
#pybind11#    def _constructImage(self, filename, hdu=None, needAllHdus=False):
#pybind11#        # Construct an instance of ExposureF by loading from filename. If hdu
#pybind11#        # is specified, load that HDU specifically. needAllHdus exists for API
#pybind11#        # compatibility, but should always be False. This function exists only
#pybind11#        # to stub default arguments into the constructor for parameters which
#pybind11#        # we are not exercising in this test.
#pybind11#        if hdu:
#pybind11#            filename = "%s[%d]" % (filename, hdu)
#pybind11#        if needAllHdus:
#pybind11#            raise Exception("Cannot needAllHdus with Exposure")
#pybind11#        return afwImage.ExposureF(filename)
#pybind11#
#pybind11#    def _checkImage(self, im, width, height, val1, val2, val3):
#pybind11#        self._checkMaskedImage(im.getMaskedImage(), width, height, val1, val2, val3)
#pybind11#
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#class TestMemory(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
