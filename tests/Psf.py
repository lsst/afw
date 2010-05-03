#!/usr/bin/env python
"""
Tests for bad pixel interpolation code

Run with:
   python Interp.py
or
   python
   >>> import Interp; Interp.run()
"""

import os, sys
from math import *
import unittest
import eups
import lsst.utils.tests as utilsTests
import lsst.pex.exceptions as pexExceptions
import lsst.pex.logging as logging
import lsst.pex.policy as policy
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.math as afwMath
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(verbose)
except NameError:
    verbose = 0
    logging.Trace_setVerbosity("algorithms.Interp", verbose)

try:
    type(display)
except NameError:
    display = False

class dgPsfTestCase(unittest.TestCase):
    """A test case for dgPsfs"""
    def setUp(self):
        FWHM = 5
        self.ksize = 25                      # size of desired kernel
        self.psf = afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize,
                                       FWHM/(2*sqrt(2*log(2))), 1, 0.1)

    def tearDown(self):
        del self.psf

    def testComputeImage(self):
        """Test the computation of the PSF's image at a point"""

        ccdXY = afwGeom.makePointD(0, 0)
        kIm = self.psf.computeImage(ccdXY)

        if False:
            ds9.mtv(kIm)        

        self.assertTrue(kIm.getWidth() == self.ksize)
        #
        # Check that the image is as expected. 
        #
        xcen, ycen = self.ksize/2, self.ksize/2
        I0 = kIm.get(xcen, ycen)
        self.assertAlmostEqual(kIm.get(xcen + 1, ycen + 1),
                               I0*self.psf.computeImage().get(xcen + 1, ycen + 1))
        #
        # Is image normalised to a peak value of unity?
        #
        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.MAX).getValue(), 1.0)
        #
        # Now create a normalised version
        #
        kIm = self.psf.computeImage(ccdXY, False)

        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.SUM).getValue(), 1.0)

    def testComputeImage2(self):
        """Test the computation of the PSF's image at a point"""

        color = afwImage.Color(1.0)
        ccdXY = afwGeom.makePointD(0, 0)
        dimen = afwGeom.makeExtentI(self.ksize, self.ksize)

        kIm = self.psf.computeImage(ccdXY)
        self.assertTrue(kIm.getWidth() == self.ksize)
        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.MAX).getValue(), 1.0)

        kIm = self.psf.computeImage(dimen)
        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.MAX).getValue(), 1.0)

        kIm = self.psf.computeImage(ccdXY, dimen)
        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.MAX).getValue(), 1.0)

        kIm = self.psf.computeImage(ccdXY, False)

        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.SUM).getValue(), 1.0)

    def testKernel(self):
        """Test the creation of the dgPsf's kernel"""

        kIm = afwImage.ImageD(self.psf.getKernel().getDimensions())
        self.psf.getKernel().computeImage(kIm, False)

        self.assertTrue(kIm.getWidth() == self.ksize)
        #
        # Check that the image is as expected. 
        #
        xcen, ycen = self.ksize/2, self.ksize/2
        I0 = kIm.get(xcen, ycen)
        self.assertAlmostEqual(kIm.get(xcen + 1, ycen + 1),
                               I0*self.psf.computeImage().get(xcen + 1, ycen + 1))
        #
        # Is image normalised?
        #
        self.assertAlmostEqual(afwMath.makeStatistics(kIm, afwMath.SUM).getValue(), 1.0)

        if False:
            ds9.mtv(kIm)        

    def testInvalidDgPsf(self):
        """Test parameters of dgPsfs, both valid and not"""
        sigma1, sigma2, b = 1, 0, 0                     # sigma2 may be 0 iff b == 0
        afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize, sigma1, sigma2, b)

        def badSigma1():
            sigma1 = 0
            afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize, sigma1, sigma2, b)

        utilsTests.assertRaisesLsstCpp(self, pexExceptions.DomainErrorException, badSigma1)

        def badSigma2():
            sigma2, b = 0, 1
            afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize, sigma1, sigma2, b)

        utilsTests.assertRaisesLsstCpp(self, pexExceptions.DomainErrorException, badSigma2)

    def testGetImage(self):
        """Test returning a realisation of the dgPsf"""

        xcen = self.psf.getKernel().getWidth()//2
        ycen = self.psf.getKernel().getHeight()//2

        stamps = []
        trueCenters = []
        for x, y in ([10, 10], [9.4999, 10.4999], [10.5001, 10.5001]):
            fx, fy = x - int(x), y - int(y)
            if fx >= 0.5:
                fx -= 1.0
            if fy >= 0.5:
                fy -= 1.0

            im = self.psf.computeImage(afwGeom.makePointD(x, y)).convertFloat()

            stamps.append(im.Factory(im, True))
            trueCenters.append([xcen + fx, ycen + fy])
            
        if display:
            mos = displayUtils.Mosaic()     # control mosaics
            ds9.mtv(mos.makeMosaic(stamps))

            for i in range(len(trueCenters)):
                bbox = mos.getBBox(i)

                ds9.dot("+",
                        bbox.getX0() + xcen, bbox.getY0() + ycen, ctype = ds9.RED, size = 1)
                ds9.dot("+",
                        bbox.getX0() + trueCenters[i][0], bbox.getY0() + trueCenters[i][1])

                ds9.dot("%.2f, %.2f" % (trueCenters[i][0], trueCenters[i][1]),
                        bbox.getX0() + xcen, bbox.getY0() + 2)

    def testKernelPsf(self):
        """Test creating a Psf from a Kernel"""

        x,y = 10.4999, 10.4999
        ksize = 15
        sigma1 = 1
        #
        # Make a PSF from that kernel
        #
        kPsf = afwDetect.createPsf("Kernel",
                                   afwMath.AnalyticKernel(ksize, ksize,
                                                          afwMath.GaussianFunction2D(sigma1, sigma1)))

        kIm = kPsf.computeImage(afwGeom.makePointD(x, y))
        #
        # And now via the dgPsf model
        #
        dgPsf = afwDetect.createPsf("DoubleGaussian", ksize, ksize, sigma1)
        dgIm = dgPsf.computeImage(afwGeom.makePointD(x, y))
        #
        # Check that they're the same
        #
        diff = type(kIm)(kIm, True); diff -= dgIm
        stats = afwMath.makeStatistics(diff, afwMath.MAX | afwMath.MIN)
        self.assertEqual(stats.getValue(afwMath.MAX), 0.0)
        self.assertEqual(stats.getValue(afwMath.MIN), 0.0)

        if display:
            mos = displayUtils.Mosaic()
            mos.setBackground(-0.1)
            ds9.mtv(mos.makeMosaic([kIm, dgIm, diff], mode="x"), frame=1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class SpatialModelPsfTestCase(unittest.TestCase):
    """A test case for SpatialModelPsf"""

    def setUp(self):
        width, height = 100, 301
        self.mi = afwImage.MaskedImageF(width, height)
        self.mi.set(0)
        sd = 3                          # standard deviation of image
        self.mi.getVariance().set(sd*sd)
        self.mi.getMask().addMaskPlane("DETECTED")

        self.FWHM = 5
        self.ksize = 25                      # size of desired kernel
        #self.psf = afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize,
        #                                self.FWHM/(2*sqrt(2*log(2))), 1, 0.1)

        rand = afwMath.Random()               # make these tests repeatable by setting seed

        im = self.mi.getImage()
        afwMath.randomGaussianImage(im, rand) # N(0, 1)
        im *= sd                              # N(0, sd^2)
        del im

        for x, y in [(20, 20), (60, 20),
                     (30, 35), (50, 50),
                     (50, 130), (70, 80),
                     (60, 210), (20, 210)]:
            source = afwDetect.Source()

            flux = 10000 - 0*x - 10*y

            sigma = 3 + 0.005*(y - self.mi.getHeight()/2)
            psf = afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize, sigma, 1, 0.1)
            im = psf.getImage(0, 0).convertFloat()
            im *= flux
            smi = self.mi.getImage().Factory(self.mi.getImage(),
                                             afwImage.BBox(afwImage.PointI(x - self.ksize/2,
                                                                           y - self.ksize/2),
                                                           self.ksize, self.ksize))

            dx = rand.uniform() - 0.5
            dy = rand.uniform() - 0.5
            im = afwMath.offsetImage(im, dx, dy)

            smi += im
            del psf; del im; del smi

        psf = afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize,
                                   self.FWHM/(2*sqrt(2*log(2))), 1, 0.1)

        self.cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), width, height), 100)
        ds = afwDetect.FootprintSetF(self.mi, afwDetect.Threshold(100), "DETECTED")
        objects = ds.getFootprints()

        if display:
            ds9.mtv(self.mi)
        #
        # Prepare to measure
        #
        moPolicy = policy.Policy()
        moPolicy.add("centroidAlgorithm", "SDSS")
        moPolicy.add("shapeAlgorithm", "SDSS")
        moPolicy.add("photometryAlgorithm", "NAIVE")
        moPolicy.add("apRadius", 3.0)

        measureSources = afwDetect.makeMeasureSources(afwImage.makeExposure(self.mi), moPolicy, psf)

        sourceList = afwDetect.SourceSet()
        for i in range(len(objects)):
            source = afwDetect.Source()
            sourceList.append(source)

            source.setId(i)
            source.setFlagForDetection(source.getFlagForDetection() | afwDetect.Flags.BINNED1);

            measureSources.apply(source, objects[i])

            self.cellSet.insertCandidate(afwDetect.makePsfCandidate(source, self.mi))

    def tearDown(self):
        del self.cellSet
        del self.mi

    def testGetPcaKernel(self):
        """Convert our cellSet to a LinearCombinationKernel"""

        nEigenComponents = 2
        spatialOrder  =    1
        kernelSize =      31
        nStarPerCell =     4
        nStarPerCellSpatialFit = 0
        tolerance =     1e-5
        reducedChi2ForPsfCandidates = 2.0
        nIterForPsf =      5

        width, height = kernelSize, kernelSize
        afwDetect.PsfCandidateF.setWidth(width); afwDetect.PsfCandidateF.setHeight(height);
        nu = width*height - 1           # number of degrees of freedom/star for chi^2

        reply = ""
        for iter in range(nIterForPsf):
            if display:
                ds9.mtv(self.mi, frame = 0)
                #
                # Show the candidates we're using
                #
                for cell in self.cellSet.getCellList():
                    #print "Cell", cell.getBBox()
                    i = 0
                    for cand in cell.begin(False): # don't skip BAD stars
                        i += 1
                        source = afwDetect.cast_PsfCandidateF(cand).getSource()

                        xc, yc = source.getXAstrom() - self.mi.getX0(), source.getYAstrom() - self.mi.getY0()
                        if cand.isBad():
                            ds9.dot("o", xc, yc, ctype = ds9.RED)
                        elif i <= nStarPerCell:
                            ds9.dot("o", xc, yc, ctype = ds9.GREEN)
                        else:
                            ds9.dot("o", xc, yc, ctype = ds9.YELLOW)

            pair = afwDetect.createKernelFromPsfCandidates(self.cellSet, nEigenComponents, spatialOrder,
                                                            kernelSize, nStarPerCell)

            kernel, eigenValues = pair[0], pair[1]; del pair

            print "lambda", " ".join(["%g" % l for l in eigenValues])

            pair = afwDetect.fitSpatialKernelFromPsfCandidates(kernel, self.cellSet,
                                                                nStarPerCellSpatialFit, tolerance)
            status, chi2 = pair[0], pair[1]; del pair
            print "Spatial fit: %s chi^2 = %.2g" % (status, chi2)

            psf = afwDetect.createPsf("PCA", kernel) # Hurrah!
            #
            # Label Psf candidate stars with bad chi^2 as BAD
            #
            nDiscard = 1
            for cell in self.cellSet.getCellList():
                worstId, worstChi2 = -1, -1
                for cand in cell.begin(True): # only not BAD candidates
                    cand = afwDetect.cast_PsfCandidateF(cand)

                    rchi2 = cand.getChi2()/nu

                    if rchi2 < reducedChi2ForPsfCandidates:
                        cand.setStatus(afwMath.SpatialCellCandidate.GOOD)
                        continue

                    if rchi2 > worstChi2:
                        worstId, worstChi2 = cand.getId(), rchi2
                        
                for cand in cell.begin(True): # only not BAD candidates
                    cand = afwDetect.cast_PsfCandidateF(cand)
                    if cand.getId() == worstId:
                        cand.setStatus(afwMath.SpatialCellCandidate.BAD)

            self.assertTrue(afwMath.cast_AnalyticKernel(psf.getKernel()) is None)
            self.assertTrue(afwMath.cast_LinearCombinationKernel(psf.getKernel()) is not None)
            #
            # OK, we're done for this iteration.  The rest is fluff
            #
            if not display:
                continue
            
            #print psf.getKernel().toString()

            eImages = []
            for k in afwMath.cast_LinearCombinationKernel(psf.getKernel()).getKernelList():
                im = afwImage.ImageD(k.getDimensions())
                k.computeImage(im, False)
                eImages.append(im)

            mos = displayUtils.Mosaic()
            frame = 3
            ds9.mtv(mos.makeMosaic(eImages), frame = frame)
            ds9.dot("Eigen Images", 0, 0, frame = frame)
            #
            # Make a mosaic of Psf candidates
            #
            stamps = []; stampInfo = []

            for cell in self.cellSet.getCellList():
                for cand in cell.begin(False):
                    #
                    # Swig doesn't know that we inherited from SpatialCellImageCandidate;  all
                    # it knows is that we have a SpatialCellCandidate, and SpatialCellCandidates
                    # don't know about getImage;  so cast the pointer to PsfCandidate
                    #
                    cand = afwDetect.cast_PsfCandidateF(cand)
                    s = cand.getSource()

                    im = cand.getImage()

                    stamps.append(im)
                    stampInfo.append("[%d 0x%x]" % (s.getId(), s.getFlagForDetection()))

            mos = displayUtils.Mosaic()
            frame = 1
            ds9.mtv(mos.makeMosaic(stamps), frame = frame, lowOrderBits = True)
            mos.drawLabels(stampInfo, frame = frame)
            #
            # Reconstruct the Psf as a function of position
            #
            psfImages = []; labels = []

            nx, ny = 3, 4
            for iy in range(ny):
                for ix in range(nx):
                    x = int((ix + 0.5)*self.mi.getWidth()/nx)
                    y = int((iy + 0.5)*self.mi.getHeight()/ny)

                    im = psf.getImage(x, y)
                    psfImages.append(im.Factory(im, True))
                    labels.append("Psf(%d,%d)" % (int(x), int(y)))

                    if not True:
                        print x, y, "Psf parameters:", psf.getKernel().getKernelParameters()

            frame = 2
            mos.makeMosaic(psfImages, frame = frame, mode = nx)
            mos.drawLabels(labels, frame = frame)

            stamps = []; stampInfo = []

            for cell in self.cellSet.getCellList():
                for cand in cell.begin(False): # include bad candidates
                    cand = afwDetect.cast_PsfCandidateF(cand)

                    infoStr = "%d X^2=%.1f" % (cand.getSource().getId(), cand.getChi2()/nu)

                    if cand.isBad():
                        if True:
                            infoStr += "B"
                        else:
                            continue

                    im = cand.getImage()
                    stamps.append(im)
                    stampInfo.append(infoStr)

            try:
                frame = 5
                mos.makeMosaic(stamps, frame = frame)
                mos.drawLabels(stampInfo, frame = frame)
                ds9.dot("PsfCandidates", 0, -3, frame = frame)
            except RuntimeError, e:
                print e

            residuals = self.mi.Factory(self.mi, True)
            for cell in self.cellSet.getCellList():
                for cand in cell.begin(False):
                    #
                    # Swig doesn't know that we inherited from SpatialCellImageCandidate;  all
                    # it knows is that we have a SpatialCellCandidate, and SpatialCellCandidates
                    # don't know about getImage;  so cast the pointer to PsfCandidate
                    #
                    cand = afwDetect.cast_PsfCandidateF(cand)
                    s = cand.getSource()

                    afwDetect.subtractPsf(psf, residuals, s.getXAstrom(), s.getYAstrom())

            ds9.mtv(residuals, frame = 4)

            if iter < nIterForPsf - 1 and reply != "c":
                while True:
                    reply = raw_input("Next iteration? [ync] ")
                    if reply in ("", "c", "n", "y"):
                        break
                    else:
                        print >> sys.stderr, "Unrecognised response: %s" % reply

                if reply == "n":
                    break
            
    def testCandidateList(self):
        if False and display:
            ds9.mtv(self.mi)

            for cell in self.cellSet.getCellList():
                x0, y0 = cell.getBBox().getX0(), cell.getBBox().getY0()
                x1, y1 = cell.getBBox().getX1(), cell.getBBox().getY1()
                
                print x0, y0, " ", x1, y1
                x0 -= 0.5; y0 -= 0.5
                x1 += 0.5; y1 += 0.5

                ds9.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], ctype = ds9.RED)

        self.assertFalse(self.cellSet.getCellList()[0].empty())
        self.assertFalse(self.cellSet.getCellList()[1].empty())
        self.assertFalse(self.cellSet.getCellList()[2].empty())
        self.assertTrue(self.cellSet.getCellList()[3].empty())

        stamps = []
        stampInfo = []
        for cell in self.cellSet.getCellList():
            for cand in cell:
                #
                # Swig doesn't know that we inherited from SpatialCellImageCandidate;  all
                # it knows is that we have a SpatialCellCandidate, and SpatialCellCandidates
                # don't know about getImage;  so cast the pointer to SpatialCellImageCandidate<Image<float> >
                # and all will be well
                #
                cand = afwMath.cast_SpatialCellImageCandidateMF(cell[0])
                width, height = 29, 25
                cand.setWidth(width); cand.setHeight(height);

                im = cand.getImage()
                stamps.append(im)

                self.assertEqual(im.getWidth(), width)
                self.assertEqual(im.getHeight(), height)
        
        if display:
            mos = displayUtils.Mosaic()
            mos.makeMosaic(stamps, frame = 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class RHLTestCase(unittest.TestCase):
    """A test case for SpatialModelPsf"""

    def calcDoubleGaussian(self, im, x, y, amp, sigma1, sigma2 = 1.0, b = 0):
        """Insert a DoubleGaussian into the image centered at (x, y)"""
        import math

        x = x - im.getX0(); y = y - im.getY0()

        for ix in range(im.getWidth()):
            for iy in range(im.getHeight()):
                r2 = math.pow(x - ix, 2) + math.pow(y - iy, 2)
                val = math.exp(-r2/(2.0*pow(sigma1, 2))) + b*math.exp(-r2/(2.0*pow(sigma2, 2)))
                im.set(ix, iy, amp/(1 + b)*val)

    def setUp(self):
        width, height = 300, 250
        self.mi = afwImage.MaskedImageF(width, height)
        self.mi.set(0)
        self.mi.getVariance().set(10)
        self.mi.getMask().addMaskPlane("DETECTED")

        self.ksize = 45                      # size of desired kernel

        for x, y in [(120, 120), (160, 120), ]:
            flux = 10000 # - 0*x - 10*(y - 10)

            sigma = 3
            dx, dy = 0.50, 0.50
            dx, dy = -0.50, -0.50
            #dx, dy = 0, 0

            smi = self.mi.getImage().Factory(self.mi.getImage(),
                                             afwImage.BBox(afwImage.PointI(x - self.ksize/2,
                                                                           y - self.ksize/2),
                                                           self.ksize, self.ksize))
            
            im = afwImage.ImageF(self.ksize, self.ksize)
            self.calcDoubleGaussian(im, self.ksize/2 + dx, self.ksize/2 + dy, 1.0, sigma, 1, 0.1)

            im /= afwMath.makeStatistics(im, afwMath.MAX).getValue()
            im *= flux

            smi += im
            del im; del smi

        self.FWHM = 5
        psf = afwDetect.createPsf("DoubleGaussian", self.ksize, self.ksize,
                                   self.FWHM/(2*sqrt(2*log(2))), 1, 0.1)

        self.cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), width, height), 100)
        ds = afwDetect.FootprintSetF(self.mi, afwDetect.Threshold(10), "DETECTED")
        objects = ds.getFootprints()
        #
        # Prepare to measure
        #
        moPolicy = policy.Policy()
        moPolicy.add("centroidAlgorithm", "SDSS")
        moPolicy.add("shapeAlgorithm", "SDSS")
        moPolicy.add("photometryAlgorithm", "NAIVE")
        moPolicy.add("apRadius", 3.0)
 
        measureSources = afwDetect.makeMeasureSources(afwImage.makeExposure(self.mi), moPolicy, psf)

        sourceList = afwDetect.SourceSet()
        for i in range(len(objects)):
            source = afwDetect.Source()
            sourceList.append(source)

            source.setId(i)

            measureSources.apply(source, objects[i])
            source.setFlagForDetection(source.getFlagForDetection() | afwDetect.Flags.BINNED1);

            source.setXAstrom(1e-2*int(100*source.getXAstrom() + 0.5)) # get exact centroids
            source.setYAstrom(1e-2*int(100*source.getYAstrom() + 0.5))

            if not False:
                print source.getXAstrom(), source.getYAstrom(), source.getPsfFlux(), \
                      measureSourceUtils.explainDetectionFlags(source.getFlagForDetection())

            self.cellSet.insertCandidate(afwDetect.makePsfCandidate(source, self.mi))
            
        frame = 1
        ds9.mtv(self.mi, frame = frame); ds9.dot("Double Gaussian", 140, 100, frame = frame)

    def tearDown(self):
        del self.cellSet
        del self.mi

    def testRHL(self):
        """Convert our cellSet to a LinearCombinationKernel"""

        nEigenComponents = 1
        spatialOrder  =    0
        kernelSize =      35
        nStarPerCell =     2

        width, height = 45, 45
        afwDetect.PsfCandidateF.setWidth(width); afwDetect.PsfCandidateF.setHeight(height);
        #
        # Show candidates
        #
        if not False:
            stamps = []
            for cell in self.cellSet.getCellList():
                for cand in cell:
                    cand = afwDetect.cast_PsfCandidateF(cand)
                    s = cand.getSource()

                    im = cand.getImage()

                    stamps.append(im)

            if False:
                mos = displayUtils.Mosaic()
                frame = 2
                im = mos.makeMosaic(stamps)
                
                imim = im.getImage()
                imim *= 10000/afwMath.makeStatistics(imim, afwMath.MAX).getValue()
                del imim
                
                ds9.mtv(im, frame = frame)

        pair = afwDetect.createKernelFromPsfCandidates(self.cellSet, nEigenComponents, spatialOrder,
                                                        kernelSize, nStarPerCell)

        kernel, eigenValues = pair[0], pair[1]; del pair
        
        psf = afwDetect.createPsf("PCA", kernel) # Hurrah!

        if display:
            xy = []
            showModel = not True
            if showModel:
                oim = self.mi.Factory(self.mi, True)
            for cell in self.cellSet.getCellList():
                for cand in cell:
                    source = afwDetect.cast_PsfCandidateF(cand).getSource()

                    delta = -0.5
                    delta =  0.0
                    afwDetect.subtractPsf(psf, self.mi,
                                           source.getXAstrom() + delta, source.getYAstrom() + delta)

                    xy.append([source.getXAstrom(), source.getYAstrom()])

            frame = 4
            ds9.mtv(self.mi, frame = frame)
            for xc, yc in xy:
                ds9.dot("x", xc - self.mi.getX0(), yc - self.mi.getY0(), ctype = ds9.GREEN, frame = frame)

            if showModel:
                self.mi -= oim
                self.mi *= -1
                ds9.mtv(self.mi, frame = frame + 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(dgPsfTestCase)
    #suites += unittest.makeSuite(SpatialModelPsfTestCase)
    #suites += unittest.makeSuite(RHLTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the utilsTests"""
    utilsTests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
