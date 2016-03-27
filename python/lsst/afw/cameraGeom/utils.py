#!/usr/bin/env python

#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#

"""
Support for cameraGeom
"""
from __future__ import division
import math
import numpy
import itertools

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase

from .rotateBBoxBy90 import rotateBBoxBy90
from .assembleImage import assembleAmplifierImage, assembleAmplifierRawImage
from .cameraGeomLib import PUPIL, FOCAL_PLANE
from lsst.afw.display.utils import _getDisplayFromDisplayOrFrame

import lsst.afw.display as afwDisplay
import lsst.afw.display.utils as displayUtils

def prepareWcsData(wcs, amp, isTrimmed=True):
    """!Put Wcs from an Amp image into CCD coordinates

    @param[in, out] wcs  WCS object to modify in place
    @param[in] amp  Amp object to use
    @param[in] isTrimmed  Is the image to which the WCS refers trimmed of non-imaging pixels?
    """
    if not amp.getHasRawInfo():
        raise RuntimeError("Cannot modify wcs without raw amp information")
    if isTrimmed:
        ampBox = amp.getRawDataBBox()
    else:
        ampBox = amp.getRawBBox()
    wcs.flipImage(amp.getRawFlipX(), amp.getRawFlipY(), ampBox.getDimensions())
    #Shift WCS for trimming
    wcs.shiftReferencePixel(-ampBox.getMinX(), -ampBox.getMinY())
    #Account for shift of amp data in larger ccd matrix
    offset = amp.getRawXYOffset()
    wcs.shiftReferencePixel(offset.getX(), offset.getY())

def plotFocalPlane(camera, pupilSizeDeg_x=0, pupilSizeDeg_y=None, dx=0.1, dy=0.1, figsize=(10., 10.),
                   useIds=False, showFig=True, savePath=None):
    """!Make a plot of the focal plane along with a set points that sample the Pupil

    @param[in] camera  a camera object
    @param[in] pupilSizeDeg_x  Amount of the pupil to sample in x in degrees
    @param[in] pupilSizeDeg_y  Amount of the pupil to sample in y in degrees
    @param[in] dx  Spacing of sample points in x in degrees
    @param[in] dy  Spacing of sample points in y in degrees
    @param[in] figsize  matplotlib style tuple indicating the size of the figure in inches
    @param[in] useIds Label detectors by name, not id
    @param[in] showFig  Display the figure on the screen?
    @param[in] savePath  If not None, save a copy of the figure to this name
    """
    try:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Can't run plotFocalPlane: matplotlib has not been set up")

    if pupilSizeDeg_x:
        if pupilSizeDeg_y is None:
            pupilSizeDeg_y = pupilSizeDeg_x

        pupil_gridx, pupil_gridy = numpy.meshgrid(numpy.arange(0., pupilSizeDeg_x+dx, dx) - pupilSizeDeg_x/2., 
                                                  numpy.arange(0., pupilSizeDeg_y+dy, dy) -  pupilSizeDeg_y/2.)
        pupil_gridx, pupil_gridy = pupil_gridx.flatten(), pupil_gridy.flatten()
    else:
        pupil_gridx, pupil_gridy = [], []

    xs = []
    ys = []
    pcolors = []
    for pos in zip(pupil_gridx, pupil_gridy):
        posRad = afwGeom.Point2D(math.radians(pos[0]), math.radians(pos[1]))
        cp = camera.makeCameraPoint(posRad, PUPIL)
        ncp = camera.transform(cp, FOCAL_PLANE)
        xs.append(ncp.getPoint().getX())
        ys.append(ncp.getPoint().getY())
        dets = camera.findDetectors(cp)
        if len(dets) > 0:
            pcolors.append('w')
        else:
            pcolors.append('k')


    colorMap = {0:'b', 1:'y', 2:'g', 3:'r'}

    patches = []
    colors = []
    plt.figure(figsize=figsize)
    ax = plt.gca()
    xvals = []
    yvals = []
    for det in camera:
        corners = [(c.getX(), c.getY()) for c in det.getCorners(FOCAL_PLANE)]
        for corner in corners:
            xvals.append(corner[0])
            yvals.append(corner[1])
        colors.append(colorMap[det.getType()])
        patches.append(Polygon(corners, True))
        center = det.getOrientation().getFpPosition()
        ax.text(center.getX(), center.getY(), det.getId() if useIds else det.getName(),
                horizontalalignment='center', size=6)

    patchCollection = PatchCollection(patches, alpha=0.6, facecolor=colors)
    ax.add_collection(patchCollection)
    ax.scatter(xs, ys, s=10, alpha=.7, linewidths=0., c=pcolors)
    ax.set_xlim(min(xvals) - abs(0.1*min(xvals)), max(xvals) + abs(0.1*max(xvals)))
    ax.set_ylim(min(yvals) - abs(0.1*min(yvals)), max(yvals) + abs(0.1*max(yvals)))
    ax.set_xlabel('Focal Plane X (mm)')
    ax.set_ylabel('Focal Plane Y (mm)')
    if savePath is not None:
        plt.savefig(savePath)
    if showFig:
        plt.show()

def makeImageFromAmp(amp, imValue=None, imageFactory=afwImage.ImageU, markSize=10, markValue=0,
                     scaleGain = lambda gain: (gain*1000)//10):
    """!Make an image from an amp object

    Since images are integer images by default, the gain needs to be scaled to give enough dynamic range
    to see variation from amp to amp.  The scaling algorithm is assignable.

    @param[in] amp  Amp record to use for constructing the raw amp image
    @param[in] imValue  Value to assign to the constructed image scaleGain(gain) is used if not set
    @param[in] imageFactory  Type of image to construct
    @param[in] markSize  Size of mark at read corner in pixels
    @param[in] markValue  Value of pixels in the read corner mark
    @param[in] scaleGain  The function by which to scale the gain
    @return an untrimmed amp image
    """
    if not amp.getHasRawInfo():
        raise RuntimeError("Can't create a raw amp image without raw amp information")
    bbox = amp.getRawBBox()
    dbbox = amp.getRawDataBBox()
    img = imageFactory(bbox)
    if imValue is None:
        img.set(scaleGain(amp.getGain()))
    else:
        img.set(imValue)
    #Set the first pixel read to a different value
    markbbox = afwGeom.Box2I()
    if amp.getReadoutCorner() == 0:
        markbbox.include(dbbox.getMin())
        markbbox.include(dbbox.getMin()+afwGeom.Extent2I(markSize, markSize))
    elif amp.getReadoutCorner() == 1:
        cornerPoint = afwGeom.Point2I(dbbox.getMaxX(), dbbox.getMinY())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(-markSize, markSize))
    elif amp.getReadoutCorner() == 2:
        cornerPoint = afwGeom.Point2I(dbbox.getMax())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(-markSize, -markSize))
    elif amp.getReadoutCorner() == 3:
        cornerPoint = afwGeom.Point2I(dbbox.getMinX(), dbbox.getMaxY())
        markbbox.include(cornerPoint)
        markbbox.include(cornerPoint + afwGeom.Extent2I(markSize, -markSize))
    else:
        raise RuntimeError("Could not set readout corner")
    mimg = imageFactory(img, markbbox, False)
    mimg.set(markValue)
    return img

def calcRawCcdBBox(ccd):
    """!Calculate the raw ccd bounding box

    @param[in] ccd  Detector for with to calculate the un-trimmed bounding box
    @return Box2I of the un-trimmed Detector,
            or None if there is not enough information to calculate raw BBox
    """
    bbox = afwGeom.Box2I()
    for amp in ccd:
        if not amp.getHasRawInfo():
            return None
        tbbox = amp.getRawBBox()
        tbbox.shift(amp.getRawXYOffset())
        bbox.include(tbbox)
    return bbox

def makeImageFromCcd(ccd, isTrimmed=True, showAmpGain=True, imageFactory=afwImage.ImageU, rcMarkSize=10,
                     binSize=1):
    """!Make an Image of a Ccd

    @param[in] ccd  Detector to use in making the image
    @param[in] isTrimmed  Assemble a trimmed Detector image if True
    @param[in] showAmpGain  Use the per amp gain to color the pixels in the image
    @param[in] imageFactory  Image type to generate
    @param[in] rcMarkSize  Size of the mark to make in the amp images at the read corner
    @param[in] binSize  Bin the image by this factor in both dimensions
    @return Image of the Detector
    """
    ampImages = []
    index = 0
    if isTrimmed:
         bbox = ccd.getBBox()
    else:
         bbox = calcRawCcdBBox(ccd)
    for amp in ccd:
        if amp.getHasRawInfo():
            if showAmpGain:
                ampImages.append(makeImageFromAmp(amp, imageFactory=imageFactory, markSize=rcMarkSize))
            else:
                ampImages.append(makeImageFromAmp(amp, imValue=(index+1)*1000,
                                                  imageFactory=imageFactory, markSize=rcMarkSize))
            index += 1

    if len(ampImages) > 0:
        ccdImage = imageFactory(bbox)
        for ampImage, amp in itertools.izip(ampImages, ccd):
            if isTrimmed:
                assembleAmplifierImage(ccdImage, ampImage, amp)
            else:
                assembleAmplifierRawImage(ccdImage, ampImage, amp)
    else:
        if not isTrimmed:
            raise RuntimeError("Cannot create untrimmed CCD without amps with raw information")
        ccdImage = imageFactory(ccd.getBBox())
    ccdImage = afwMath.binImage(ccdImage, binSize)
    return ccdImage

class FakeImageDataSource(object):
    """A class to retrieve synthetic images for display by the show* methods"""
    def __init__(self, isTrimmed=True, verbose=False, background=numpy.nan, 
                 showAmpGain=True, markSize=10, markValue=0,
                 ampImValue=None, scaleGain=lambda gain: (gain*1000)//10):
        """!Construct a FakeImageDataSource

        @param[in] isTrimmed  Should amps be trimmed?
        @param[in] verbose  Be chatty
        @param[in] background  The value of any pixels that lie outside the CCDs
        @param[in] showAmpGain  color the amp segments with the gain of the amp
        @param[in] markSize  size of the side of the box used to mark the read corner
        @param[in] markValue  value to assing the read corner mark
        @param[in] ampImValue  Value to assing to amps.  scaleGain(gain) is used if None
        @param[in] scaleGain  function to scale the gain by
        """
        self.isTrimmed = isTrimmed
        self.verbose = verbose
        self.background = background
        self.showAmpGain = showAmpGain
        self.markSize = markSize
        self.markValue = markValue
        self.ampImValue = ampImValue
        self.scaleGain = scaleGain

    def getCcdImage(self, det, imageFactory, binSize):
        """!Return a CCD image for the detector

        @param[in] det: Detector to use for making the image
        @param[in] imageFactory: image constructor for making the image
        @param[in] binSize: number of pixels per bin axis
        """
        return makeImageFromCcd(det, isTrimmed=self.isTrimmed, showAmpGain=self.showAmpGain,
                                imageFactory=imageFactory, binSize=binSize)

    def getAmpImage(self, amp, imageFactory):
        """!Return an amp segment image

        @param[in] amp  AmpInfoTable for this amp
        @param[in] imageFactory  image constructor fo making the imag
        """
        ampImage = makeImageFromAmp(amp, imValue=self.ampImValue, imageFactory=imageFactory,
                                    markSize=self.markSize,
                markValue=self.markValue, scaleGain=self.scaleGain)
        if self.isTrimmed:
            ampImage = ampImage.Factory(ampImage, amp.getRawDataBBox(), False)
        return ampImage

class ButlerImage(FakeImageDataSource):
    """A class to return an Image of a given Ccd using the butler"""
    
    def __init__(self, butler=None, type="raw",
                 isTrimmed=True, verbose=False, background=numpy.nan, gravity=None, *args, **kwargs):
        """!Create an object that knows how to prepare images for showCamera using the butler

        \param The butler to use.  If no butler is provided an empty image is returned
        \param type The type of image to read (e.g. raw, bias, flat, calexp)
        \param isTrimmed If true, the showCamera command expects to be given trimmed images
        \param verbose  Be chatty (in particular, print any error messages from the butler)
        \param background  The value of any pixels that lie outside the CCDs
        \param  gravity  If the image returned by the butler is trimmed (e.g. some of the SuprimeCam CCDs)
                 Specify how to fit the image into the available space; N => align top, W => align left
        \param *args, *kwargs Passed to the butler
        """
        super(ButlerImage, self).__init__(*args)
        self.isTrimmed = isTrimmed
        self.type = type
        self.butler = butler
        self.kwargs = kwargs
        self.isRaw = False
        self.gravity = gravity
        self.background = background
        self.verbose = verbose
    
    def _prepareImage(self, ccd, im, binSize, allowRotate=True):
        if binSize > 1:
            im = afwMath.binImage(im, binSize)
    
        if allowRotate:
            im = afwMath.rotateImageBy90(im, ccd.getOrientation().getNQuarter())
                
        return im

    def getCcdImage(self, ccd, imageFactory=afwImage.ImageF, binSize=1):
        """Return an image of the specified amp in the specified ccd"""

        if self.isTrimmed:
             bbox = ccd.getBBox()
        else:
             bbox = calcRawCcdBBox(ccd)

        im = None
        if self.butler is not None:
            e = None
            if self.type == "calexp":    # reading the exposure can die if the PSF's unknown
                try:
                    fileName = self.butler.get(self.type + "_filename", ccd=ccd.getId(),
                                                    **self.kwargs)[0]
                    im = imageFactory(fileName)
                except Exception as e:
                    pass
            else:
                try:
                    im = self.butler.get(self.type, ccd=ccd.getId(),
                                         **self.kwargs).getMaskedImage().getImage()
                except Exception as e:
                    pass
                    
            if e:
                if self.verbose:
                    print "Reading %s: %s" % (ccd.getId(), e)

        if im is None:
            return self._prepareImage(ccd, imageFactory(*bbox.getDimensions()), binSize)

        if self.type == "raw":
            if hasattr(im, 'convertF'):
                im = im.convertF()
        else:
            return self._prepareImage(ccd, im, binSize, allowRotate=False) # calexps were rotated by the ISR 

        ccdImage = im.Factory(bbox)

        ampImages = []
        med0 = None
        for a in ccd:
            bias = im[a.getRawHorizontalOverscanBBox()]
            data = im[a.getRawDataBBox()]
            data -= afwMath.makeStatistics(bias, afwMath.MEANCLIP).getValue()
            data *= a.getGain()

            ampImages.append(data)

        ccdImage = imageFactory(bbox)
        for ampImage, amp in itertools.izip(ampImages, ccd):
            if self.isTrimmed:
                assembleAmplifierImage(ccdImage, ampImage, amp)
            else:
                assembleAmplifierRawImage(ccdImage, ampImage, amp)

        return ccdImage

def overlayCcdBoxes(ccd, untrimmedCcdBbox, nQuarter, isTrimmed, ccdOrigin, display, binSize):
    """!Overlay bounding boxes on an image display

    @param[in] ccd  Detector to iterate for the amp bounding boxes
    @param[in] untrimmedCcdBbox  Bounding box of the un-trimmed Detector
    @param[in] nQuarter  number of 90 degree rotations to apply to the bounding boxes
    @param[in] isTrimmed  Is the Detector image over which the boxes are layed trimmed?
    @param[in] ccdOrigin  Detector origin relative to the  parent origin if in a larger pixel grid
    @param[in] display image display to display on
    @param[in] binSize  binning factor
    """
    with display.Buffering():
        ccdDim = untrimmedCcdBbox.getDimensions()
        ccdBbox = rotateBBoxBy90(untrimmedCcdBbox, nQuarter, ccdDim)
        for amp in ccd:
            if isTrimmed:
                ampbbox = amp.getBBox()
            else:
                ampbbox = amp.getRawBBox()
                ampbbox.shift(amp.getRawXYOffset())
            if nQuarter != 0:
                ampbbox = rotateBBoxBy90(ampbbox, nQuarter, ccdDim)

            displayUtils.drawBBox(ampbbox, origin=ccdOrigin, borderWidth=0.49,
                                  display=display, bin=binSize)

            if not isTrimmed and amp.getHasRawInfo():
                for bbox, ctype in ((amp.getRawHorizontalOverscanBBox(), afwDisplay.RED),
                                    (amp.getRawDataBBox(), afwDisplay.BLUE),
                                    (amp.getRawVerticalOverscanBBox(), afwDisplay.MAGENTA),
                                    (amp.getRawPrescanBBox(), afwDisplay.YELLOW)):
                    if amp.getRawFlipX():
                        bbox.flipLR(amp.getRawBBox().getDimensions().getX())
                    if amp.getRawFlipY():
                        bbox.flipTB(amp.getRawBBox().getDimensions().getY())
                    bbox.shift(amp.getRawXYOffset())
                    if nQuarter != 0:
                        bbox = rotateBBoxBy90(bbox, nQuarter, ccdDim)
                    displayUtils.drawBBox(bbox, origin=ccdOrigin, borderWidth=0.49, ctype=ctype,
                                          display=display, bin=binSize)
            # Label each Amp
            xc, yc = (ampbbox.getMin()[0] + ampbbox.getMax()[0])//2, (ampbbox.getMin()[1] +
                    ampbbox.getMax()[1])//2
            #
            # Rotate the amp labels too
            #
            if nQuarter == 0:
                c, s = 1, 0
            elif nQuarter == 1:
                c, s = 0, -1
            elif nQuarter == 2:
                c, s = -1, 0
            elif nQuarter == 3:
                c, s = 0, 1
            c, s = 1, 0
            ccdHeight = ccdBbox.getHeight()
            ccdWidth = ccdBbox.getWidth()
            xc -= 0.5*ccdHeight
            yc -= 0.5*ccdWidth

            xc, yc = 0.5*ccdHeight + c*xc + s*yc, 0.5*ccdWidth + -s*xc + c*yc

            if ccdOrigin:
                xc += ccdOrigin[0]
                yc += ccdOrigin[1]
            display.dot(str(amp.getName()), xc/binSize, yc/binSize, textAngle=nQuarter*90)

        displayUtils.drawBBox(ccdBbox, origin=ccdOrigin,
                              borderWidth=0.49, ctype=afwDisplay.MAGENTA, display=display, bin=binSize)

def showAmp(amp, imageSource=FakeImageDataSource(isTrimmed=False), display=None, overlay=True,
            imageFactory=afwImage.ImageU):
    """!Show an amp in an image display

    @param[in] amp  amp record to use in display
    @param[in] imageSource  Source for getting the amp image.  Must have a getAmpImage method.
    @param[in] display image display to use
    @param[in] overlay  Overlay bounding boxes?
    @param[in] imageFactory  Type of image to display (only used if ampImage is None)
    """

    if not display:
        display = _getDisplayFromDisplayOrFrame()

    ampImage = imageSource.getAmpImage(amp, imageFactory=imageFactory)
    ampImSize = ampImage.getDimensions()
    title = amp.getName()
    display.mtv(ampImage, title=title)
    if overlay:
        with display.Buffering():
            if amp.getHasRawInfo() and ampImSize == amp.getRawBBox().getDimensions():
                bboxes = [(amp.getRawBBox(), 0.49, afwDisplay.GREEN),]
                xy0 = bboxes[0][0].getMin()
                bboxes.append((amp.getRawHorizontalOverscanBBox(), 0.49, afwDisplay.RED)) 
                bboxes.append((amp.getRawDataBBox(), 0.49, afwDisplay.BLUE))
                bboxes.append((amp.getRawPrescanBBox(), 0.49, afwDisplay.YELLOW))
                bboxes.append((amp.getRawVerticalOverscanBBox(), 0.49, afwDisplay.MAGENTA))
            else:
                bboxes = [(amp.getBBox(), 0.49, None),]
                xy0 = bboxes[0][0].getMin()

            for bbox, borderWidth, ctype in bboxes:
                if bbox.isEmpty():
                    continue
                bbox = afwGeom.Box2I(bbox)
                bbox.shift(-afwGeom.ExtentI(xy0))
                displayUtils.drawBBox(bbox, borderWidth=borderWidth, ctype=ctype, display=display)

def showCcd(ccd, imageSource=FakeImageDataSource(), display=None, frame=None, overlay=True,
            imageFactory=afwImage.ImageF, binSize=1, inCameraCoords=False):
    """!Show a CCD on display

    @param[in] ccd  Detector to use in display
    @param[in] imageSource  Source for producing images to display.  Must have a getCcdImage method.
    @param[in] display image display to use
    @param[in] overlay  Show amp bounding boxes on the displayed image?
    @param[in] imageFactory  The image factory to use in generating the images.
    @param[in] binSize  Binning factor
    @param[in] inCameraCoords  Show the Detector in camera coordinates?
    """
    display = _getDisplayFromDisplayOrFrame(display, frame)

    ccdOrigin = afwGeom.Point2I(0,0)
    nQuarter = 0
    ccdImage = imageSource.getCcdImage(ccd, imageFactory=imageFactory, binSize=binSize)

    ccdBbox = ccdImage.getBBox()
    if ccdBbox.getDimensions() == ccd.getBBox().getDimensions():
        isTrimmed = True
    else:
        isTrimmed = False

    if inCameraCoords:
        nQuarter = ccd.getOrientation().getNQuarter()
        ccdImage = afwMath.rotateImageBy90(ccdImage, nQuarter)
    title = ccd.getName()
    if isTrimmed:
        title += "(trimmed)"

    if display:
        display.mtv(ccdImage, title=title)

        if overlay:
            overlayCcdBoxes(ccd, ccdBbox, nQuarter, isTrimmed, ccdOrigin, display, binSize)

    return ccdImage

def getCcdInCamBBoxList(ccdList, binSize, pixelSize_o, origin):
    """!Get the bounding boxes of a list of Detectors within a camera sized pixel grid

    @param[in] ccdList  List of Detector
    @param[in] binSize  Binning factor
    @param[in] pixelSize_o  Size of the pixel in mm.
    @param[in] origin  origin of the camera pixel grid in pixels
    @return a list of bounding boxes in camera pixel coordinates
    """
    boxList = []
    for ccd in ccdList:
        if not pixelSize_o == ccd.getPixelSize():
            raise RuntimeError(
                "Cameras with detectors with different pixel scales are not currently supported")

        dbbox = afwGeom.Box2D()
        for corner in ccd.getCorners(FOCAL_PLANE):
            dbbox.include(corner)
        llc = dbbox.getMin()
        nQuarter = ccd.getOrientation().getNQuarter()
        cbbox = ccd.getBBox()
        ex = cbbox.getDimensions().getX()//binSize
        ey = cbbox.getDimensions().getY()//binSize
        bbox = afwGeom.Box2I(cbbox.getMin(), afwGeom.Extent2I(int(ex), int(ey)))
        bbox = rotateBBoxBy90(bbox, nQuarter, bbox.getDimensions())
        bbox.shift(afwGeom.Extent2I(int(llc.getX()//pixelSize_o.getX()/binSize),
                                    int(llc.getY()//pixelSize_o.getY()/binSize)))
        bbox.shift(afwGeom.Extent2I(-int(origin.getX()//binSize), -int(origin.getY())//binSize))
        boxList.append(bbox)
    return boxList

def getCameraImageBBox(camBbox, pixelSize, bufferSize):
    """!Get the bounding box of a camera sized image in pixels

    @param[in] camBbox  Camera bounding box in focal plane coordinates (mm)
    @param[in] pixelSize  Size of a detector pixel in mm
    @param[in] bufferSize  Buffer around edge of image in pixels
    @return the resulting bounding box
    """
    pixMin = afwGeom.Point2I(int(camBbox.getMinX()//pixelSize.getX()),
                             int(camBbox.getMinY()//pixelSize.getY()))
    pixMax = afwGeom.Point2I(int(camBbox.getMaxX()//pixelSize.getX()),
                             int(camBbox.getMaxY()//pixelSize.getY()))
    retBox = afwGeom.Box2I(pixMin, pixMax)
    retBox.grow(bufferSize)
    return retBox

def makeImageFromCamera(camera, detectorNameList=None, background=numpy.nan, bufferSize=10,
                        imageSource=FakeImageDataSource(), imageFactory=afwImage.ImageU, binSize=1):
    """!Make an Image of a Camera

    @param[in] camera  Camera object to use to make the image
    @param[in] detectorNameList  List of detector names to use in building the image.
               Use all Detectors if None.
    @param[in] background  Value to use where there is no Detector
    @param[in] bufferSize  Size of border in binned pixels to make around the camera image
    @param[in] imageSource  Source to get ccd images.  Must have a getCcdImage method
    @param[in] imageFactory  Type of image to build
    @param[in] binSize  bin factor
    @return an image of the camera
    """
    if detectorNameList is None:
        ccdList = camera
    else:
        ccdList = [camera[name] for name in detectorNameList]

    if detectorNameList is None:
        camBbox = camera.getFpBBox()
    else:
        camBbox = afwGeom.Box2D()
        for detName in detectorNameList:
            for corner in camera[detName].getCorners(FOCAL_PLANE):
                camBbox.include(corner)

    pixelSize_o = camera[camera.getNameIter().next()].getPixelSize()
    camBbox = getCameraImageBBox(camBbox, pixelSize_o, bufferSize*binSize)
    origin = camBbox.getMin()

    camIm = imageFactory(int(math.ceil(camBbox.getDimensions().getX()/binSize)),
                         int(math.ceil(camBbox.getDimensions().getY()/binSize)))
    camIm[:] = imageSource.background

    assert imageSource.isTrimmed, "isTrimmed is False isn't supported by getCcdInCamBBoxList"

    boxList = getCcdInCamBBoxList(ccdList, binSize, pixelSize_o, origin) 
    for det, bbox in itertools.izip(ccdList, boxList):
        im = imageSource.getCcdImage(det, imageFactory, binSize)

        nQuarter = det.getOrientation().getNQuarter()
        im = afwMath.rotateImageBy90(im, nQuarter)
        
        imView = camIm.Factory(camIm, bbox, afwImage.LOCAL)
        try:
            imView[:] = im
        except Exception as e:
            pass

    return camIm

def showCamera(camera, imageSource=FakeImageDataSource(), imageFactory=afwImage.ImageF,
               detectorNameList=None, binSize=10, bufferSize=10, frame=None, overlay=True, title="",
               ctype=afwDisplay.GREEN, textSize=1.25, originAtCenter=True, display=None, **kwargs):
    """!Show a Camera on display, with the specified display

    The rotation of the sensors is snapped to the nearest multiple of 90 deg. 
    Also note that the pixel size is constant over the image array. The lower left corner (LLC) of each
    sensor amp is snapped to the LLC of the pixel containing the LLC of the image. 
    if overlay show the IDs and detector boundaries

    @param[in] camera  Camera to show
    @param[in] imageSource  Source to get Ccd images from.  Must have a getCcdImage method.
    @param[in] imageFactory  Type of image to make
    @param[in] detectorNameList  List of names of Detectors to use. If None use all
    @param[in] binSize  bin factor
    @param[in] bufferSize  size of border in binned pixels to make around camera image.
    @param[in] frame  specify image display (@deprecated; new code should use display)
    @param[in] overlay  Overlay Detector IDs and boundaries?
    @param[in] title  Title in display
    @param[in] ctype  Color to use when drawing Detector boundaries
    @param[in] textSize  Size of detector labels
    @param[in] originAtCenter Put origin of the camera WCS at the center of the image? Else it will be LL
    @param[in] display  image display on which to display
    @param[in] **kwargs all remaining keyword arguments are passed to makeImageFromCamera
    @return the mosaic image
    """
    display = _getDisplayFromDisplayOrFrame(display, frame)

    if binSize < 1:
        binSize = 1
    cameraImage = makeImageFromCamera(camera, detectorNameList=detectorNameList, bufferSize=bufferSize,
                                      imageSource=imageSource, imageFactory=imageFactory, binSize=binSize,
                                      **kwargs)

    if detectorNameList is None:
        ccdList = [camera[name] for name in camera.getNameIter()]
    else:
        ccdList = [camera[name] for name in detectorNameList]

    if detectorNameList is None:
        camBbox = camera.getFpBBox()
    else:
        camBbox = afwGeom.Box2D()
        for detName in detectorNameList:
            for corner in camera[detName].getCorners(FOCAL_PLANE):
                camBbox.include(corner)
    pixelSize = ccdList[0].getPixelSize()
    if originAtCenter:
        #Can't divide SWIGGED extent type things when division is imported
        #from future.  This is DM-83
        ext = cameraImage.getBBox().getDimensions()

        wcsReferencePixel = afwGeom.PointI(ext.getX()//2, ext.getY()//2)
    else:
        wcsReferencePixel = afwGeom.Point2I(0,0)
    wcs = makeFocalPlaneWcs(pixelSize*binSize, wcsReferencePixel)

    if display:
        if title == "":
            title = camera.getName()
        display.mtv(cameraImage, title=title, wcs=wcs)

        if overlay:
            with display.Buffering():
                camBbox = getCameraImageBBox(camBbox, pixelSize, bufferSize*binSize)
                bboxList = getCcdInCamBBoxList(ccdList, binSize, pixelSize, camBbox.getMin())
                for bbox, ccd in itertools.izip(bboxList, ccdList):
                    nQuarter = ccd.getOrientation().getNQuarter()
                    # borderWidth to 0.5 to align with the outside edge of the pixel
                    displayUtils.drawBBox(bbox, borderWidth=0.5, ctype=ctype, display=display)
                    dims = bbox.getDimensions()
                    display.dot(ccd.getName(), bbox.getMinX()+dims.getX()/2, bbox.getMinY()+dims.getY()/2,
                                ctype=ctype, size=textSize, textAngle=nQuarter*90)

    return cameraImage

def makeFocalPlaneWcs(pixelSize, referencePixel):
    """!Make a WCS for the focal plane geometry (i.e. returning positions in "mm")

    @param[in] pixelSize  Size of the image pixels in physical units
    @param[in] referencePixel  Pixel for origin of WCS
    @return Wcs object for mapping between pixels and focal plane.
    """

    md = dafBase.PropertySet()
    if referencePixel is None:
        referencePixel = afwGeom.PointD(0,0)
    for i in range(2):
        md.set("CRPIX%d"%(i+1), referencePixel[i])
        md.set("CRVAL%d"%(i+1), 0.)
    md.set("CDELT1", pixelSize[0])
    md.set("CDELT2", pixelSize[1])
    md.set("CTYPE1", "CAMERA_X")
    md.set("CTYPE2", "CAMERA_Y")
    md.set("CUNIT1", "mm")
    md.set("CUNIT2", "mm")

    return afwImage.makeWcs(md)

def findAmp(ccd, pixelPosition):
    """!Find the Amp with the specified pixel position within the composite

    @param[in] ccd  Detector to look in
    @param[in] pixelPosition  Point2I containing the pixel position
    @return Amp record in which pixelPosition falls or None if no Amp found.
    """

    for amp in ccd:
        if amp.getBBox().contains(pixelPosition):
            return amp

    return None
