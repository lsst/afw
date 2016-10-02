#pybind11##!/usr/bin/env python
#pybind11#from __future__ import absolute_import, division
#pybind11#from __future__ import print_function
#pybind11#from past.builtins import cmp
#pybind11#from builtins import zip
#pybind11#from builtins import range
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
#pybind11#Tests for Footprints, and FootprintSets
#pybind11#
#pybind11#Run with:
#pybind11#   footprint2.py
#pybind11#or
#pybind11#   python
#pybind11#   >>> import footprint2; footprint2.run()
#pybind11#"""
#pybind11#
#pybind11#import unittest
#pybind11#import lsst.utils.tests
#pybind11#import lsst.afw.image as afwImage
#pybind11#import lsst.afw.geom as afwGeom
#pybind11#import lsst.afw.detection as afwDetect
#pybind11#import lsst.afw.display.ds9 as ds9
#pybind11#
#pybind11#try:
#pybind11#    type(display)
#pybind11#except NameError:
#pybind11#    display = False
#pybind11#
#pybind11#
#pybind11#def toString(*args):
#pybind11#    """toString written in python"""
#pybind11#    if len(args) == 1:
#pybind11#        args = args[0]
#pybind11#
#pybind11#    y, x0, x1 = args
#pybind11#    return "%d: %d..%d" % (y, x0, x1)
#pybind11#
#pybind11#
#pybind11#def peakFromImage(im, pos):
#pybind11#    """Function to extract the sort key of peak height. Sort by decreasing peak height."""
#pybind11#    val = im.get(pos[0], pos[1])[0]
#pybind11#    return -1.0 * val
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class Object(object):
#pybind11#
#pybind11#    def __init__(self, val, spans):
#pybind11#        self.val = val
#pybind11#        self.spans = spans
#pybind11#
#pybind11#    def insert(self, im):
#pybind11#        """Insert self into an image"""
#pybind11#        for sp in self.spans:
#pybind11#            y, x0, x1 = sp
#pybind11#            for x in range(x0, x1+1):
#pybind11#                im.set(x, y, self.val)
#pybind11#
#pybind11#    def __eq__(self, other):
#pybind11#        for osp, sp in zip(other.getSpans(), self.spans):
#pybind11#            if osp.toString() != toString(sp):
#pybind11#                return False
#pybind11#
#pybind11#        return True
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class FootprintSetTestCase(unittest.TestCase):
#pybind11#    """A test case for FootprintSet"""
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.im = afwImage.ImageU(afwGeom.Extent2I(12, 8))
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects = []
#pybind11#        self.objects += [Object(10, [(1, 4, 4), (2, 3, 5), (3, 4, 4)])]
#pybind11#        self.objects += [Object(20, [(5, 7, 8), (5, 10, 10), (6, 8, 9)])]
#pybind11#        self.objects += [Object(20, [(6, 3, 3)])]
#pybind11#
#pybind11#        self.im.set(0)                       # clear image
#pybind11#        for obj in self.objects:
#pybind11#            obj.insert(self.im)
#pybind11#
#pybind11#        if False and display:
#pybind11#            ds9.mtv(self.im, frame=0)
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.im
#pybind11#
#pybind11#    def testGC(self):
#pybind11#        """Check that FootprintSets are automatically garbage collected (when MemoryTestCase runs)"""
#pybind11#
#pybind11#        afwDetect.FootprintSet(afwImage.ImageU(afwGeom.Extent2I(10, 20)), afwDetect.Threshold(10))
#pybind11#
#pybind11#    def testFootprints(self):
#pybind11#        """Check that we found the correct number of objects and that they are correct"""
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testFootprints2(self):
#pybind11#        """Check that we found the correct number of objects using FootprintSet"""
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testFootprintsImageId(self):
#pybind11#        """Check that we can insert footprints into an Image"""
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        idImage = afwImage.ImageU(self.im.getDimensions())
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        for foot in objects:
#pybind11#            foot.insertIntoImage(idImage, foot.getId())
#pybind11#
#pybind11#        if False:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#        for i in range(len(objects)):
#pybind11#            for sp in objects[i].getSpans():
#pybind11#                for x in range(sp.getX0(), sp.getX1() + 1):
#pybind11#                    self.assertEqual(idImage.get(x, sp.getY()), objects[i].getId())
#pybind11#
#pybind11#    def testFootprintSetImageId(self):
#pybind11#        """Check that we can insert a FootprintSet into an Image, setting relative IDs"""
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        idImage = ds.insertIntoImage(True)
#pybind11#        if display:
#pybind11#            ds9.mtv(idImage, frame=2)
#pybind11#
#pybind11#        for i in range(len(objects)):
#pybind11#            for sp in objects[i].getSpans():
#pybind11#                for x in range(sp.getX0(), sp.getX1() + 1):
#pybind11#                    self.assertEqual(idImage.get(x, sp.getY()), i + 1)
#pybind11#
#pybind11#    def testFootprintsImage(self):
#pybind11#        """Check that we can search Images as well as MaskedImages"""
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#
#pybind11#        self.assertEqual(len(objects), len(self.objects))
#pybind11#        for i in range(len(objects)):
#pybind11#            self.assertEqual(objects[i], self.objects[i])
#pybind11#
#pybind11#    def testGrow2(self):
#pybind11#        """Grow some more interesting shaped Footprints.  Informative with display, but no numerical tests"""
#pybind11#        # Can't set mask plane as the image is not a masked image.
#pybind11#        ds = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#
#pybind11#        idImage = afwImage.ImageU(self.im.getDimensions())
#pybind11#        idImage.set(0)
#pybind11#
#pybind11#        i = 1
#pybind11#        for foot in ds.getFootprints()[0:1]:
#pybind11#            gfoot = afwDetect.growFootprint(foot, 3, False)
#pybind11#            gfoot.insertIntoImage(idImage, i)
#pybind11#            i += 1
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.im, frame=0)
#pybind11#            ds9.mtv(idImage, frame=1)
#pybind11#
#pybind11#    def testGrow(self):
#pybind11#        """Grow footprints using the FootprintSet constructor"""
#pybind11#        fs = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10))
#pybind11#        self.assertEqual(len(fs.getFootprints()), len(self.objects))
#pybind11#        for isotropic in (True, False, afwDetect.FootprintControl(True),):
#pybind11#            grown = afwDetect.FootprintSet(fs, 1, isotropic)
#pybind11#            self.assertEqual(len(fs.getFootprints()), len(self.objects))
#pybind11#
#pybind11#            self.assertGreater(len(grown.getFootprints()), 0)
#pybind11#            self.assertLessEqual(len(grown.getFootprints()), len(fs.getFootprints()))
#pybind11#
#pybind11#    def testFootprintControl(self):
#pybind11#        """Test the FootprintControl constructor"""
#pybind11#        fctrl = afwDetect.FootprintControl()
#pybind11#        self.assertFalse(fctrl.isCircular()[0])  # not set
#pybind11#        self.assertFalse(fctrl.isIsotropic()[0])  # not set
#pybind11#
#pybind11#        fctrl.growIsotropic(False)
#pybind11#        self.assertTrue(fctrl.isCircular()[0])
#pybind11#        self.assertTrue(fctrl.isIsotropic()[0])
#pybind11#        self.assertTrue(fctrl.isCircular()[1])
#pybind11#        self.assertFalse(fctrl.isIsotropic()[1])
#pybind11#
#pybind11#        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#        fctrl = afwDetect.FootprintControl()
#pybind11#        fctrl.growLeft(False)
#pybind11#        self.assertTrue(fctrl.isLeft()[0])  # it's now set
#pybind11#        self.assertFalse(fctrl.isLeft()[1])  # ... but False
#pybind11#
#pybind11#        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#        fctrl = afwDetect.FootprintControl(True, False, False, False)
#pybind11#        self.assertTrue(fctrl.isLeft()[0])
#pybind11#        self.assertTrue(fctrl.isRight()[0])
#pybind11#        self.assertTrue(fctrl.isUp()[0])
#pybind11#        self.assertTrue(fctrl.isDown()[0])
#pybind11#
#pybind11#        self.assertTrue(fctrl.isLeft()[1])
#pybind11#        self.assertFalse(fctrl.isRight()[1])
#pybind11#
#pybind11#    def testGrowCircular(self):
#pybind11#        """Grow footprints in all 4 directions using the FootprintSet/FootprintControl constructor """
#pybind11#        im = afwImage.MaskedImageF(11, 11)
#pybind11#        im.set(5, 5, (10,))
#pybind11#        fs = afwDetect.FootprintSet(im, afwDetect.Threshold(10))
#pybind11#        self.assertEqual(len(fs.getFootprints()), 1)
#pybind11#
#pybind11#        radius = 3                      # How much to grow by
#pybind11#        for fctrl in (afwDetect.FootprintControl(),
#pybind11#                      afwDetect.FootprintControl(True),
#pybind11#                      afwDetect.FootprintControl(True, True),
#pybind11#                      ):
#pybind11#            grown = afwDetect.FootprintSet(fs, radius, fctrl)
#pybind11#            afwDetect.setMaskFromFootprintList(im.getMask(), grown.getFootprints(), 0x10)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(im)
#pybind11#
#pybind11#            foot = grown.getFootprints()[0]
#pybind11#
#pybind11#            if not fctrl.isCircular()[0]:
#pybind11#                self.assertEqual(foot.getNpix(), 1)
#pybind11#            elif fctrl.isCircular()[0]:
#pybind11#                assert radius == 3
#pybind11#                if fctrl.isIsotropic()[1]:
#pybind11#                    self.assertEqual(foot.getNpix(), 29)
#pybind11#                else:
#pybind11#                    self.assertEqual(foot.getNpix(), 25)
#pybind11#
#pybind11#    def testGrowLRUD(self):
#pybind11#        """Grow footprints in various directions using the FootprintSet/FootprintControl constructor """
#pybind11#        im = afwImage.MaskedImageF(11, 11)
#pybind11#        x0, y0, ny = 5, 5, 3
#pybind11#        for y in range(y0 - ny//2, y0 + ny//2 + 1):
#pybind11#            im.set(x0, y, (10,))
#pybind11#        fs = afwDetect.FootprintSet(im, afwDetect.Threshold(10))
#pybind11#        self.assertEqual(len(fs.getFootprints()), 1)
#pybind11#
#pybind11#        ngrow = 2                       # How much to grow by
#pybind11#        #
#pybind11#        # Test growing to the left and/or right
#pybind11#        #
#pybind11#        for fctrl in (
#pybind11#            afwDetect.FootprintControl(False, True, False, False),
#pybind11#            afwDetect.FootprintControl(True, False, False, False),
#pybind11#            afwDetect.FootprintControl(True, True, False, False),
#pybind11#        ):
#pybind11#            grown = afwDetect.FootprintSet(fs, ngrow, fctrl)
#pybind11#            im.getMask().set(0)
#pybind11#            afwDetect.setMaskFromFootprintList(im.getMask(), grown.getFootprints(), 0x10)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(im)
#pybind11#
#pybind11#            foot = grown.getFootprints()[0]
#pybind11#            nextra = 0
#pybind11#            if fctrl.isLeft()[1]:
#pybind11#                nextra += ngrow
#pybind11#                for y in range(y0 - ny//2, y0 + ny//2 + 1):
#pybind11#                    self.assertNotEqual(im.getMask().get(x0 - 1, y), 0)
#pybind11#
#pybind11#            if fctrl.isRight()[1]:
#pybind11#                nextra += ngrow
#pybind11#                for y in range(y0 - ny//2, y0 + ny//2 + 1):
#pybind11#                    self.assertNotEqual(im.getMask().get(x0 + 1, y), 0)
#pybind11#
#pybind11#            self.assertEqual(foot.getNpix(), (1 + nextra)*ny)
#pybind11#        #
#pybind11#        # Test growing to up and/or down
#pybind11#        #
#pybind11#        for fctrl in (
#pybind11#            afwDetect.FootprintControl(False, False, True, False),
#pybind11#            afwDetect.FootprintControl(False, False, False, True),
#pybind11#            afwDetect.FootprintControl(False, False, True, True),
#pybind11#        ):
#pybind11#            grown = afwDetect.FootprintSet(fs, ngrow, fctrl)
#pybind11#            im.getMask().set(0)
#pybind11#            afwDetect.setMaskFromFootprintList(im.getMask(), grown.getFootprints(), 0x10)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(im)
#pybind11#
#pybind11#            foot = grown.getFootprints()[0]
#pybind11#            nextra = 0
#pybind11#            if fctrl.isUp()[1]:
#pybind11#                nextra += ngrow
#pybind11#                for y in range(y0 + ny//2 + 1, y0 + ny//2 + ngrow + 1):
#pybind11#                    self.assertNotEqual(im.getMask().get(x0, y), 0)
#pybind11#
#pybind11#            if fctrl.isDown()[1]:
#pybind11#                nextra += ngrow
#pybind11#                for y in range(y0 - ny//2 - 1, y0 - ny//2 - ngrow - 1):
#pybind11#                    self.assertNotEqual(im.getMask().get(x0, y), 0)
#pybind11#
#pybind11#            self.assertEqual(foot.getNpix(), ny + nextra)
#pybind11#
#pybind11#    def testGrowLRUD2(self):
#pybind11#        """Grow footprints in various directions using the FootprintSet/FootprintControl constructor
#pybind11#
#pybind11#        Check that overlapping grown Footprints give the expected answers
#pybind11#        """
#pybind11#        ngrow = 3                       # How much to grow by
#pybind11#        for fctrl, xy in [
#pybind11#            (afwDetect.FootprintControl(True, True, False, False), [(4, 5), (5, 6), (6, 5)]),
#pybind11#            (afwDetect.FootprintControl(False, False, True, True), [(5, 4), (6, 5), (5, 6)]),
#pybind11#        ]:
#pybind11#            im = afwImage.MaskedImageF(11, 11)
#pybind11#            for x, y in xy:
#pybind11#                im.set(x, y, (10,))
#pybind11#            fs = afwDetect.FootprintSet(im, afwDetect.Threshold(10))
#pybind11#            self.assertEqual(len(fs.getFootprints()), 1)
#pybind11#
#pybind11#            grown = afwDetect.FootprintSet(fs, ngrow, fctrl)
#pybind11#            im.getMask().set(0)
#pybind11#            afwDetect.setMaskFromFootprintList(im.getMask(), grown.getFootprints(), 0x10)
#pybind11#
#pybind11#            if display:
#pybind11#                ds9.mtv(im)
#pybind11#
#pybind11#            self.assertEqual(len(grown.getFootprints()), 1)
#pybind11#            foot = grown.getFootprints()[0]
#pybind11#
#pybind11#            npix = 1 + 2*ngrow
#pybind11#            npix += 3 + 2*ngrow         # 3: distance between pair of set pixels 000X0X000
#pybind11#            self.assertEqual(foot.getNpix(), npix)
#pybind11#
#pybind11#    def testInf(self):
#pybind11#        """Test detection for images with Infs"""
#pybind11#
#pybind11#        im = afwImage.MaskedImageF(afwGeom.Extent2I(10, 20))
#pybind11#        im.set(0)
#pybind11#
#pybind11#        import numpy
#pybind11#        for x in range(im.getWidth()):
#pybind11#            im.set(x, im.getHeight() - 1, (numpy.Inf, 0x0, 0))
#pybind11#
#pybind11#        ds = afwDetect.FootprintSet(im, afwDetect.createThreshold(100))
#pybind11#
#pybind11#        objects = ds.getFootprints()
#pybind11#        afwDetect.setMaskFromFootprintList(im.getMask(), objects, 0x10)
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(im)
#pybind11#
#pybind11#        self.assertEqual(len(objects), 1)
#pybind11#
#pybind11##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#pybind11#
#pybind11#
#pybind11#class PeaksInFootprintsTestCase(unittest.TestCase):
#pybind11#    """A test case for detecting Peaks within Footprints"""
#pybind11#
#pybind11#    def doSetUp(self, dwidth=0, dheight=0, x0=0, y0=0):
#pybind11#        width, height = 14 + x0 + dwidth, 10 + y0 + dheight
#pybind11#        self.im = afwImage.MaskedImageF(afwGeom.Extent2I(width, height))
#pybind11#        #
#pybind11#        # Objects that we should detect
#pybind11#        #
#pybind11#        self.objects, self.peaks = [], []
#pybind11#        self.objects.append([[4, 1, 10], [3, 2, 10], [4, 2, 20], [5, 2, 10], [4, 3, 10], ])
#pybind11#        self.peaks.append([[4, 2]])
#pybind11#        self.objects.append([[9, 7, 30], [10, 7, 29], [12, 7, 25], [10, 8, 27], [11, 8, 26], ])
#pybind11#        self.peaks.append([[9, 7]])
#pybind11#        self.objects.append([[3, 8, 10], [4, 8, 10], ])
#pybind11#        self.peaks.append([[3, 8], [4, 8], ])
#pybind11#
#pybind11#        for pp in self.peaks:           # allow for x0, y0
#pybind11#            for p in pp:
#pybind11#                p[0] += x0
#pybind11#                p[1] += y0
#pybind11#        for oo in self.objects:
#pybind11#            for o in oo:
#pybind11#                o[0] += x0
#pybind11#                o[1] += y0
#pybind11#
#pybind11#        self.im.set((0, 0x0, 0))                       # clear image
#pybind11#        for obj in self.objects:
#pybind11#            for x, y, I in obj:
#pybind11#                self.im.getImage().set(x, y, I)
#pybind11#
#pybind11#        if False and display:
#pybind11#            ds9.mtv(self.im, frame=0)
#pybind11#
#pybind11#    def setUp(self):
#pybind11#        self.im, self.fs = None, None
#pybind11#
#pybind11#    def tearDown(self):
#pybind11#        del self.im
#pybind11#        del self.fs
#pybind11#
#pybind11#    def doTestPeaks(self, dwidth=0, dheight=0, x0=0, y0=0, threshold=10, callback=None, polarity=True, grow=0):
#pybind11#        """Worker routine for tests
#pybind11#        polarity:  True if should search for +ve pixels"""
#pybind11#
#pybind11#        self.doSetUp(dwidth, dheight, x0, y0)
#pybind11#        if not polarity:
#pybind11#            self.im *= -1
#pybind11#
#pybind11#        if callback:
#pybind11#            callback()
#pybind11#        #
#pybind11#        # Sort self.peaks in decreasing peak height to match Footprint.getPeaks()
#pybind11#        #
#pybind11#        def peakDescending(p):
#pybind11#            return p[2] * -1.0
#pybind11#        for i, peaks in enumerate(self.peaks):
#pybind11#            self.peaks[i] = sorted([(x, y, self.im.getImage().get(x, y)) for x, y in peaks],
#pybind11#                                   key=peakDescending)
#pybind11#
#pybind11#        threshold = afwDetect.Threshold(threshold, afwDetect.Threshold.VALUE, polarity)
#pybind11#        fs = afwDetect.FootprintSet(self.im, threshold, "BINNED1")
#pybind11#
#pybind11#        if grow:
#pybind11#            fs = afwDetect.FootprintSet(fs, grow, True)
#pybind11#            msk = self.im.getMask()
#pybind11#            afwDetect.setMaskFromFootprintList(msk, fs.getFootprints(), msk.getPlaneBitMask("DETECTED"))
#pybind11#            del msk
#pybind11#
#pybind11#        self.fs = fs
#pybind11#        self.checkPeaks(dwidth, dheight, frame=3)
#pybind11#
#pybind11#    def checkPeaks(self, dwidth=0, dheight=0, frame=3):
#pybind11#        """Check that we got the peaks right"""
#pybind11#        feet = self.fs.getFootprints()
#pybind11#        #
#pybind11#        # Check that we found all the peaks
#pybind11#        #
#pybind11#        self.assertEqual(sum([len(f.getPeaks()) for f in feet]), sum([len(f.getPeaks()) for f in feet]))
#pybind11#
#pybind11#        if display:
#pybind11#            ds9.mtv(self.im, frame=frame)
#pybind11#
#pybind11#            with ds9.Buffering():
#pybind11#                for i, foot in enumerate(feet):
#pybind11#                    for p in foot.getPeaks():
#pybind11#                        ds9.dot("+", p.getIx(), p.getIy(), size=0.4, frame=frame)
#pybind11#
#pybind11#                    if i < len(self.peaks):
#pybind11#                        for trueX, trueY, peakVal in self.peaks[i]:
#pybind11#                            ds9.dot("x", trueX, trueY, size=0.4, ctype=ds9.RED, frame=frame)
#pybind11#
#pybind11#        for i, foot in enumerate(feet):
#pybind11#            npeak = None
#pybind11#            #
#pybind11#            # Peaks that touch the edge are handled differently, as only the single highest/lowest pixel
#pybind11#            # is treated as a Peak
#pybind11#            #
#pybind11#            if (dwidth != 0 or dheight != 0):
#pybind11#                if (foot.getBBox().getMinX() == 0 or foot.getBBox().getMaxX() == self.im.getWidth() - 1 or
#pybind11#                        foot.getBBox().getMinY() == 0 or foot.getBBox().getMaxY() == self.im.getHeight() - 1):
#pybind11#                    npeak = 1
#pybind11#
#pybind11#            if npeak is None:
#pybind11#                npeak = len(self.peaks[i])
#pybind11#
#pybind11#            if npeak != len(foot.getPeaks()):
#pybind11#                print("RHL", foot.repr())
#pybind11#                # print "RHL", [(p.repr().split(":")[0], p.getIx(), p.getIy()) for p in foot.getPeaks()]
#pybind11#                print("RHL", [(p.getId(), p.getIx(), p.getIy()) for p in foot.getPeaks()])
#pybind11#                print("RHL", [p[0:2] for p in self.peaks[i]])
#pybind11#
#pybind11#            self.assertEqual(len(foot.getPeaks()), npeak)
#pybind11#
#pybind11#            for j, p in enumerate(foot.getPeaks()):
#pybind11#                trueX, trueY, peakVal = self.peaks[i][j]
#pybind11#                if (p.getIx(), p.getIy()) != (trueX, trueY):
#pybind11#                    print("RHL", [(pp.getId(), pp.getIx(), pp.getIy()) for pp in foot.getPeaks()])
#pybind11#                    print("RHL", [pp[0:2] for pp in self.peaks[i]])
#pybind11#
#pybind11#                self.assertEqual((p.getIx(), p.getIy()), (trueX, trueY))
#pybind11#
#pybind11#    def testSinglePeak(self):
#pybind11#        """Test that we can find single Peaks in Footprints"""
#pybind11#
#pybind11#        self.doTestPeaks()
#pybind11#
#pybind11#    def testSingleNegativePeak(self):
#pybind11#        """Test that we can find single Peaks in Footprints when looking for -ve detections"""
#pybind11#
#pybind11#        self.doTestPeaks(polarity=False)
#pybind11#
#pybind11#    def testSinglePeakAtEdge(self):
#pybind11#        """Test that we handle Peaks correctly at the edge"""
#pybind11#
#pybind11#        self.doTestPeaks(dheight=-1)
#pybind11#
#pybind11#    def testSingleNegativePeakAtEdge(self):
#pybind11#        """Test that we handle -ve Peaks correctly at the edge"""
#pybind11#
#pybind11#        self.doTestPeaks(dheight=-1, polarity=False)
#pybind11#
#pybind11#    def testMultiPeak(self):
#pybind11#        """Test that multiple peaks are handled correctly"""
#pybind11#        def callback():
#pybind11#            x, y = 12, 7
#pybind11#            self.im.getImage().set(x, y, 100)
#pybind11#            self.peaks[1].append((x, y))
#pybind11#
#pybind11#        self.doTestPeaks(callback=callback)
#pybind11#
#pybind11#    def testMultiNegativePeak(self):
#pybind11#        """Test that multiple negative peaks are handled correctly"""
#pybind11#        def callback():
#pybind11#            x, y = 12, 7
#pybind11#            self.im.getImage().set(x, y, -100)
#pybind11#            self.peaks[1].append((x, y))
#pybind11#
#pybind11#        self.doTestPeaks(polarity=False, callback=callback)
#pybind11#
#pybind11#    def testGrowFootprints(self):
#pybind11#        """Test that we can grow footprints, correctly merging those that now touch"""
#pybind11#        def callback():
#pybind11#            self.im.getImage().set(10, 4, 20)
#pybind11#            self.peaks[-2].append((10, 4,))
#pybind11#
#pybind11#        self.doTestPeaks(dwidth=1, dheight=1, callback=callback, grow=1)
#pybind11#
#pybind11#    def testGrowFootprints2(self):
#pybind11#        """Test that we can grow footprints, correctly merging those that now overlap
#pybind11#        N.b. this caused RHL's initial implementation to crash
#pybind11#        """
#pybind11#        def callback():
#pybind11#            self.im.getImage().set(10, 4, 20)
#pybind11#            self.peaks[-2].append((10, 4, ))
#pybind11#
#pybind11#            def peaksSortKey(p):
#pybind11#                return peakFromImage(self.im, p)
#pybind11#            self.peaks[0] = sorted(sum(self.peaks, []), key=peaksSortKey)
#pybind11#
#pybind11#        self.doTestPeaks(x0=0, y0=2, dwidth=2, dheight=2, callback=callback, grow=2)
#pybind11#
#pybind11#    def testGrowFootprints3(self):
#pybind11#        """Test that we can grow footprints, correctly merging those that now totally overwritten"""
#pybind11#
#pybind11#        self.im = afwImage.MaskedImageF(14, 11)
#pybind11#
#pybind11#        self.im.getImage().set(0)
#pybind11#        self.peaks = []
#pybind11#
#pybind11#        I = 11
#pybind11#        for x, y in [(4, 7), (5, 7), (6, 7), (7, 7), (8, 7),
#pybind11#                     (4, 6), (8, 6),
#pybind11#                     (4, 5), (8, 5),
#pybind11#                     (4, 4), (8, 4),
#pybind11#                     (4, 3), (8, 3),
#pybind11#                     ]:
#pybind11#            self.im.getImage().set(x, y, I)
#pybind11#            I -= 1e-3
#pybind11#
#pybind11#        self.im.getImage().set(4, 7, 15)
#pybind11#        self.peaks.append([(4, 7,), ])
#pybind11#
#pybind11#        self.im.getImage().set(6, 5, 30)
#pybind11#        self.peaks[0].append((6, 5,))
#pybind11#
#pybind11#        self.fs = afwDetect.FootprintSet(self.im, afwDetect.Threshold(10), "BINNED1")
#pybind11#        #
#pybind11#        # The disappearing Footprint special case only shows up if the outer Footprint is grown
#pybind11#        # _after_ the inner one.  So arrange the order properly
#pybind11#        feet = self.fs.getFootprints()
#pybind11#        feet[0], feet[1] = feet[1], feet[0]
#pybind11#
#pybind11#        msk = self.im.getMask()
#pybind11#
#pybind11#        grow = 2
#pybind11#        self.fs = afwDetect.FootprintSet(self.fs, grow, False)
#pybind11#        afwDetect.setMaskFromFootprintList(msk, self.fs.getFootprints(),
#pybind11#                                           msk.getPlaneBitMask("DETECTED_NEGATIVE"))
#pybind11#
#pybind11#        if display:
#pybind11#            frame = 0
#pybind11#
#pybind11#            ds9.mtv(self.im, frame=frame)
#pybind11#
#pybind11#            with ds9.Buffering():
#pybind11#                for i, foot in enumerate(self.fs.getFootprints()):
#pybind11#                    for p in foot.getPeaks():
#pybind11#                        ds9.dot("+", p.getIx(), p.getIy(), size=0.4, frame=frame)
#pybind11#
#pybind11#                    if i < len(self.peaks):
#pybind11#                        for trueX, trueY in self.peaks[i]:
#pybind11#                            ds9.dot("x", trueX, trueY, size=0.4, ctype=ds9.RED, frame=frame)
#pybind11#
#pybind11#        self.assertEqual(len(self.fs.getFootprints()), 1)
#pybind11#        self.assertEqual(len(self.fs.getFootprints()[0].getPeaks()), len(self.peaks[0]))
#pybind11#
#pybind11#    def testMergeFootprints(self):      # YYYY
#pybind11#        """Merge positive and negative Footprints"""
#pybind11#        x0, y0 = 5, 6
#pybind11#        dwidth, dheight = 6, 7
#pybind11#
#pybind11#        def callback():
#pybind11#            x, y, I = x0 + 10, y0 + 4, -20
#pybind11#            self.im.getImage().set(x, y, I)
#pybind11#            peaks2.append((x, y, I))
#pybind11#
#pybind11#        for grow1, grow2 in [(1, 1), (3, 3), (6, 6), ]:
#pybind11#            peaks2 = []
#pybind11#            self.doTestPeaks(threshold=10, callback=callback, grow=0,
#pybind11#                             x0=x0, y0=y0, dwidth=dwidth, dheight=dheight)
#pybind11#
#pybind11#            threshold = afwDetect.Threshold(10, afwDetect.Threshold.VALUE, False)
#pybind11#            fs2 = afwDetect.FootprintSet(self.im, threshold)
#pybind11#
#pybind11#            msk = self.im.getMask()
#pybind11#            afwDetect.setMaskFromFootprintList(
#pybind11#                msk, fs2.getFootprints(), msk.getPlaneBitMask("DETECTED_NEGATIVE"))
#pybind11#
#pybind11#            self.fs.merge(fs2, grow1, grow2)
#pybind11#            self.peaks[-2] += peaks2
#pybind11#
#pybind11#            if grow1 + grow2 > 2:                                                         # grow merged all peaks
#pybind11#                def peaksSortKey(p):
#pybind11#                    return peakFromImage(self.im, p)
#pybind11#                self.peaks[0] = sorted(sum(self.peaks, []), key=peaksSortKey)
#pybind11#
#pybind11#            afwDetect.setMaskFromFootprintList(msk, self.fs.getFootprints(), msk.getPlaneBitMask("EDGE"))
#pybind11#
#pybind11#            self.checkPeaks(frame=3)
#pybind11#
#pybind11#    def testMergeFootprintsEngulf(self):
#pybind11#        """Merge two Footprints when growing one Footprint totally replaces the other"""
#pybind11#        def callback():
#pybind11#            self.im.set(0)
#pybind11#            self.peaks, self.objects = [], []
#pybind11#
#pybind11#            for x, y, I in [[6, 4, 20], [6, 5, 10]]:
#pybind11#                self.im.getImage().set(x, y, I)
#pybind11#            self.peaks.append([[6, 4]])
#pybind11#
#pybind11#            x, y, I = 8, 4, -20
#pybind11#            self.im.getImage().set(x, y, I)
#pybind11#            peaks2.append((x, y, I))
#pybind11#
#pybind11#        grow1, grow2 = 0, 3
#pybind11#        peaks2 = []
#pybind11#        self.doTestPeaks(threshold=10, callback=callback, grow=0)
#pybind11#
#pybind11#        threshold = afwDetect.Threshold(10, afwDetect.Threshold.VALUE, False)
#pybind11#        fs2 = afwDetect.FootprintSet(self.im, threshold)
#pybind11#
#pybind11#        msk = self.im.getMask()
#pybind11#        afwDetect.setMaskFromFootprintList(msk, fs2.getFootprints(), msk.getPlaneBitMask("DETECTED_NEGATIVE"))
#pybind11#
#pybind11#        self.fs.merge(fs2, grow1, grow2)
#pybind11#        self.peaks[0] += peaks2
#pybind11#
#pybind11#        def peaksSortKey(p):
#pybind11#            return peakFromImage(self.im, p)
#pybind11#        self.peaks[0] = sorted(sum(self.peaks, []), key=peaksSortKey)
#pybind11#
#pybind11#        afwDetect.setMaskFromFootprintList(msk, self.fs.getFootprints(), msk.getPlaneBitMask("EDGE"))
#pybind11#
#pybind11#        self.checkPeaks(frame=3)
#pybind11#
#pybind11#
#pybind11#class MemoryTester(lsst.utils.tests.MemoryTestCase):
#pybind11#    pass
#pybind11#
#pybind11#
#pybind11#def setup_module(module):
#pybind11#    lsst.utils.tests.init()
#pybind11#
#pybind11#
#pybind11#if __name__ == "__main__":
#pybind11#    lsst.utils.tests.init()
#pybind11#    unittest.main()
