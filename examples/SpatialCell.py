#!/usr/bin/env python
"""
An example using SpatialCells

Run with:
   python SpatialCell.py
or
   python
   >>> import SpatialCell; SpatialCell.run()
"""

import os
import sys

import eups
import lsst.pex.exceptions as pexExcept
import lsst.afw.detection as afwDetection
import lsst.afw.image.imageLib as afwImage
import lsst.afw.math.mathLib as afwMath
import lsst.afw.display.ds9 as ds9

import testSpatialCellLib

try:
    type(display)
except NameError:
    display = False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def readImage(filename=None):
    """Read an image and background subtract it"""
    if not filename:
        dataDir = eups.productDir("afwdata")
        if not dataDir:
            raise RuntimeError("You must provide a filename or setup afwdata to run these examples")

        filename = os.path.join(eups.productDir("afwdata"), "CFHT", "D4", "cal-53535-i-797722_1")
        
        bbox = afwImage.BBox(afwImage.PointI(270, 2530), 512, 512)
    else:
        bbox = None
        
    mi = afwImage.MaskedImageF(filename, 0, None, bbox)
    mi.setXY0(afwImage.PointI(0, 0))
    #
    # Subtract the background.  We'd use a canned procedure, but that's in meas/utils/sourceDetection.py. We
    # can't fix those pesky cosmic rays either, as that's in a dependent product (meas/algorithms) too
    #
    bctrl = afwMath.BackgroundControl(afwMath.Interpolate.NATURAL_SPLINE);
    bctrl.setNxSample(int(mi.getWidth()/256) + 1);
    bctrl.setNySample(int(mi.getHeight()/256) + 1);
    bctrl.sctrl.setNumSigmaClip(3.0)  
    bctrl.sctrl.setNumIter(2)

    im = mi.getImage()
    try:
        backobj = afwMath.makeBackground(im, bctrl)
    except Exception, e:
        print >> sys.stderr, e,

        bctrl.setInterpStyle(afwMath.Interpolate.CONSTANT)
        backobj = afwMath.makeBackground(im, bctrl)

    im -= backobj.getImageF()
    #
    # Find sources
    #
    threshold = afwDetection.Threshold(5, afwDetection.Threshold.STDEV)
    npixMin = 5                         # we didn't smooth
    fs = afwDetection.makeFootprintSet(mi, threshold, "DETECTED", npixMin)
    grow, isotropic = 2, False
    fs = afwDetection.makeFootprintSet(fs, grow, isotropic)
    fs.setMask(mi.getMask(), "DETECTED")

    return mi, fs

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def SpatialCellTestCase(nCandidate=5):
    """A test case for SpatialCell"""
    candidateList = afwMath.SpatialCellCandidateList()
    
    def getFlux(x):
        return 1000 - 10*x
    
    for i in (0, 1, 4, 3, 2):       # must be all numbers in range(nCandidate)
        x, y = i, 5*i
        candidateList.append(testSpatialCellLib.TestCandidate(x, y, getFlux(x)))
        
    cell = afwMath.SpatialCell("Test", afwImage.BBox(), candidateList)

    # Check that we can retrieve candidates, and that they are sorted by ranking
    assert(cell[0].getXCenter(), 0)
    assert(cell[1].getXCenter(), 1)
    assert(cell[1].getYCenter(), 5)
    
    # Build a candidate list by inserting candidates

    cell = afwMath.SpatialCell("Test", afwImage.BBox())

    for x, y in ([5, 0], [1, 1], [2, 2], [0, 0], [4, 4], [3, 4]):
        cell.insertCandidate(testSpatialCellLib.TestCandidate(x, y, getFlux(x)))

    assert(cell[0].getXCenter(), 0)
    #
    # Exercise the SpatialCell iterators
    #
    # Count the candidates
    #
    assert(cell.size(), nCandidate)
    assert(cell.end() - cell.begin(), nCandidate)

    ptr = cell.begin(); ptr.__incr__()
    assert(cell.end() - ptr, nCandidate - 1)

    assert(ptr - cell.begin(), 1)
    #
    # Now label one candidate as bad
    #
    ptr = cell[2].setStatus(afwMath.SpatialCellCandidate.BAD)

    assert(cell.size(), nCandidate - 1)
    assert(cell.end() - cell.begin(), nCandidate - 1)

    cell.setIgnoreBad(False)
    assert(cell.size(), nCandidate)
    assert(cell.end() - cell.begin(), nCandidate)

    # Check that we can lookup candidates by ID
    id = cell[1].getId()
    assert(cell.getCandidateById(id).getId(), id)


    assert(cell.getCandidateById(-1, True), None)
    try:
        cell.getCandidateById(-1)           # non-existent ID; should throw an exception
    except Exception, e:
        pass
    else:
        print >> sys.stderr, "Expected an exception looking up a non-existant ID"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def SpatialCellSetTestCase(filename=None):
    """A test case for SpatialCellSet"""

    im, fs = readImage(filename)

    if display:
        ds9.mtv(im, frame=0, title="Input")
    #
    # Create an (empty) SpatialCellSet
    #
    cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), im.getWidth(), im.getHeight()),
                                     260, 200)

    if display:
        for i in range(len(cellSet.getCellList())):
            cell = cellSet.getCellList()[i]
            ds9.line([(cell.getBBox().getX0(), cell.getBBox().getY0()),
                      (cell.getBBox().getX0(), cell.getBBox().getY1()),
                      (cell.getBBox().getX1(), cell.getBBox().getY1()),
                      (cell.getBBox().getX1(), cell.getBBox().getY0()),
                      (cell.getBBox().getX0(), cell.getBBox().getY0()),
                      ], frame=0)
            ds9.dot(cell.getLabel(),
                    (cell.getBBox().getX0() + cell.getBBox().getX1())/2,
                    (cell.getBBox().getY0() + cell.getBBox().getY1())/2)
    #
    # Populate cellSet
    #
    for foot in fs.getFootprints():
        bbox = foot.getBBox()
        xc = (bbox.getX0() + bbox.getX1())/2.0
        yc = (bbox.getY0() + bbox.getY1())/2.0
        cellSet.insertCandidate(testSpatialCellLib.TestCandidate(xc, yc, 
                                                                 im.get(int(xc + 0.5), int(yc + 0.5))[0]))
    #
    # OK, the SpatialCellList is populated
    #
    visitor = testSpatialCellLib.TestCandidateVisitor()

    cellSet.visitCandidates(visitor)
    print "There are %d candidates" % (visitor.getN())
    
    ctypes = ["red", "yellow", "cyan",]
    for i in range(len(cellSet.getCellList())):
        cell = cellSet.getCellList()[i]
        cell.visitCandidates(visitor)

        print "%s nobj=%d N=%d" % (cell.getLabel(), cell.size(), visitor.getN())
        j = 1
        for cand in cell:
            ds9.dot(str(j), cand.getXCenter(), cand.getYCenter(), size=4, ctype=ctypes[i%len(ctypes)])
            j += 1

        #print [afwMath.cast_SpatialCellImageCandidateMF(cand) for cand in cell]

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def TestImageCandidateCase():
    """A test case for TestImageCandidate"""

    cellSet = afwMath.SpatialCellSet(afwImage.BBox(afwImage.PointI(0, 0), 501, 501), 2, 3)

    # Test that we can use SpatialCellImageCandidate

    flux = 10
    cellSet.insertCandidate(testSpatialCellLib.TestImageCandidate(0, 0, flux))

    cand = cellSet.getCellList()[0][0]
    #
    # Swig doesn't know that we're a SpatialCellImageCandidate;  all it knows is that we have
    # a SpatialCellCandidate, and SpatialCellCandidates don't know about getImage;  so cast the
    # pointer to SpatialCellImageCandidate<Image<float> > and all will be well;
    #
    # First check that we _can't_ cast to SpatialCellImageCandidate<MaskedImage<float> >
    #
    assert(afwMath.cast_SpatialCellImageCandidateMF(cand), None)

    cand = afwMath.cast_SpatialCellImageCandidateF(cand)

    width, height = 15, 21
    cand.setWidth(width); cand.setHeight(height);

    im = cand.getImage()
    if False and display:
        ds9.mtv(im, title="Candidate", frame=1)
    assert(im.get(0,0), flux) # This is how TestImageCandidate sets its pixels
    assert(im.getWidth(), width)
    assert(im.getHeight(), height)
        
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def run(exit=False):
    """Run the tests"""
    SpatialCellTestCase()
    SpatialCellSetTestCase()
    TestImageCandidateCase()

if __name__ == "__main__":
    run(True)
