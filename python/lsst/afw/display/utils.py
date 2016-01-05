# 
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
# 
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and 
# the GNU General Public License along with this program.  If not, 
# see <http://www.lsstcorp.org/LegalNotices/>.
#

## \file
## \brief Utilities to use with displaying images

from __future__ import absolute_import, division, print_function

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom

__all__ = (
    "Mosaic",
    "drawBBox", "drawFootprint", "drawCoaddInputs",
    )

def _getDisplayFromDisplayOrFrame(display, frame=None):
    """!Return an afwDisplay.Display given either a display or a frame ID.

    If the two arguments are consistent, return the desired display; if they are not,
    raise a RuntimeError exception.

    If the desired display is None, return None;
    if (display, frame) == ("deferToFrame", None), return the default display"""

    import lsst.afw.display as afwDisplay # import locally to allow this file to be imported by __init__

    if display in ("deferToFrame", None):
        if display is None and frame is None:
            return None

        # "deferToFrame" is the default value, and  means "obey frame"
        display = None

    if display and not hasattr(display, "frame"):
        raise RuntimeError("display == %s doesn't support .frame" % display)

    if frame and display and display.frame != frame:
        raise RuntimeError("Please specify display *or* frame")

    if display:
        frame = display.frame

    display = afwDisplay.getDisplay(frame, create=True)

    return display

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Mosaic(object):
    """A class to handle mosaics of one or more identically-sized images (or Masks or MaskedImages)
    E.g.
    m = Mosaic()
    m.setGutter(5)
    m.setBackground(10)
    m.setMode("square")                     # the default; other options are "x" or "y"

    mosaic = m.makeMosaic(im1, im2, im3)    # build the mosaic
    display = afwDisplay.getDisplay()
    display.mtv(mosaic)                         # display it
    m.drawLabels(["Label 1", "Label 2", "Label 3"], display) # label the panels

    # alternative way to build a mosaic
    images = [im1, im2, im3]               
    labels = ["Label 1", "Label 2", "Label 3"]

    mosaic = m.makeMosaic(images)
    display.mtv(mosaic)
    m.drawLabels(labels, display)

    # Yet another way to build a mosaic (no need to build the images/labels lists)
    for i in range(len(images)):
        m.append(images[i], labels[i])
    # You may optionally include a colour, e.g. afwDisplay.YELLOW, as a third argument

    mosaic = m.makeMosaic()
    display.mtv(mosaic)
    m.drawLabels(display=display)

    Or simply:
    mosaic = m.makeMosaic(display=display)

    You can return the (ix, iy)th (or nth) bounding box (in pixels) with getBBox()
    """
    def __init__(self, gutter=3, background=0, mode="square"):
        self.gutter = gutter            # number of pixels between panels in a mosaic
        self.background = background    # value in gutters
        self.setMode(mode)              # mosaicing mode
        self.xsize = 0                  # column size of panels
        self.ysize = 0                  # row size of panels

        self.reset()

    def reset(self):
        """Reset the list of images to be mosaiced"""
        self.images = []                # images to mosaic together
        self.labels = []                # labels for images
        
    def append(self, image, label=None, ctype=None):
        """Add an image to the list of images to be mosaiced
        Set may be cleared with Mosaic.reset()

        Returns the index of this image (may be passed to getBBox())
        """
        if not self.xsize:
            self.xsize = image.getWidth()
            self.ysize = image.getHeight()

        self.images.append(image)
        self.labels.append((label, ctype))

        return len(self.images)

    def makeMosaic(self, images=None, display="deferToFrame", mode=None, background=None, title="", frame=None):
        """Return a mosaic of all the images provided; if none are specified,
        use the list accumulated with Mosaic.append().

        Note that this mosaic is a patchwork of the input images;  if you want to
        make a mosaic of a set images of the sky, you probably want to use the coadd code
        
        If display or frame (deprecated) is specified, display the mosaic
        """

        if not images:
            images = self.images

        self.nImage = len(images)
        if self.nImage == 0:
            raise RuntimeError, "You must provide at least one image"

        self.xsize, self.ysize = 0, 0
        for im in images:
            w, h = im.getWidth(), im.getHeight()
            if w > self.xsize:
                self.xsize = w
            if h > self.ysize:
                self.ysize = h

        if background is None:
            background = self.background
        if mode is None:
            mode = self.mode

        if mode == "square":
            nx, ny = 1, self.nImage
            while nx*im.getWidth() < ny*im.getHeight():
                nx += 1
                ny = self.nImage//nx

                if nx*ny < self.nImage:
                    ny += 1
                if nx*ny < self.nImage:
                    nx += 1

            if nx > self.nImage:
                nx = self.nImage
                
            assert(nx*ny >= self.nImage)
        elif mode == "x":
            nx, ny = self.nImage, 1
        elif mode == "y":
            nx, ny = 1, self.nImage
        elif isinstance(mode, int):
            nx = mode
            ny = self.nImage//nx
            if nx*ny < self.nImage:
                ny += 1
        else:
            raise RuntimeError, ("Unknown mosaicing mode: %s" % mode)

        self.nx, self.ny = nx, ny

        mosaic = images[0].Factory(
            afwGeom.Extent2I(nx*self.xsize + (nx - 1)*self.gutter, ny*self.ysize + (ny - 1)*self.gutter)
            )
        try:
            mosaic.set(self.background)
        except AttributeError:
            raise RuntimeError("Attempt to mosaic images of type %s which don't support set" %
                               type(mosaic))

        for i in range(len(images)):
            smosaic = mosaic.Factory(mosaic, self.getBBox(i%nx, i//nx), afwImage.LOCAL)
            im = images[i]

            if smosaic.getDimensions() != im.getDimensions(): # im is smaller than smosaic
                llc = afwGeom.PointI((smosaic.getWidth() - im.getWidth())//2,
                                     (smosaic.getHeight() - im.getHeight())//2)
                smosaic = smosaic.Factory(smosaic, afwGeom.Box2I(llc, im.getDimensions()), afwImage.LOCAL)

            smosaic[:] = im

        display = _getDisplayFromDisplayOrFrame(display, frame)
        if display:
            display.mtv(mosaic, title=title)

            if images == self.images:
                self.drawLabels(display=display)
            
        return mosaic

    def setGutter(self, gutter):
        """Set the number of pixels between panels in a mosaic"""
        self.gutter = gutter

    def setBackground(self, background):
        """Set the value in the gutters"""
        self.background = background

    def setMode(self, mode):
        """Set mosaicing mode.  Valid options:
           square       Make mosaic as square as possible
           x            Make mosaic one image high
           y            Make mosaic one image wide
    """

        if mode not in ("square", "x", "y"):
            raise RuntimeError, ("Unknown mosaicing mode: %s" % mode)

        self.mode = mode

    def getBBox(self, ix, iy=None):
        """Get the BBox for the nth or (ix, iy)the panel"""

        if iy is None:
            ix, iy = ix % self.nx, ix//self.nx

        return afwGeom.Box2I(afwGeom.PointI(ix*(self.xsize + self.gutter), iy*(self.ysize + self.gutter)),
                             afwGeom.ExtentI(self.xsize, self.ysize))

    def drawLabels(self, labels=None, display="deferToFrame", frame=None):
        """Draw the list labels at the corners of each panel.  If labels is None, use the ones
        specified by Mosaic.append()"""

        if not labels:
            labels = self.labels

        if not labels:
            return

        if len(labels) != self.nImage:
            raise RuntimeError, ("You provided %d labels for %d panels" % (len(labels), self.nImage))

        display = _getDisplayFromDisplayOrFrame(display, frame)
        if not display:
            return
            
        with display.Buffering():
            for i in range(len(labels)):
                if labels[i]:
                    label, ctype = labels[i], None
                    try:
                        label, ctype = label
                    except:
                        pass

                    if not label:
                        continue

                    display.dot(str(label), self.getBBox(i).getMinX(), self.getBBox(i).getMinY(), ctype=ctype)

def drawBBox(bbox, borderWidth=0.0, origin=None, display="deferToFrame", ctype=None, bin=1, frame=None):
    """Draw an afwImage::BBox on a display frame with the specified ctype.  Include an extra borderWidth pixels
If origin is present, it's Added to the BBox

All BBox coordinates are divided by bin, as is right and proper for overlaying on a binned image
    """
    x0, y0 = bbox.getMinX(), bbox.getMinY()
    x1, y1 = bbox.getMaxX(), bbox.getMaxY()

    if origin:
        x0 += origin[0]; x1 += origin[0]
        y0 += origin[1]; y1 += origin[1]

    x0 /= bin; y0 /= bin
    x1 /= bin; y1 /= bin
    borderWidth /= bin
    
    display = _getDisplayFromDisplayOrFrame(display, frame)
    display.line([(x0 - borderWidth, y0 - borderWidth),
              (x0 - borderWidth, y1 + borderWidth),
              (x1 + borderWidth, y1 + borderWidth),
              (x1 + borderWidth, y0 - borderWidth),
              (x0 - borderWidth, y0 - borderWidth),
              ], ctype=ctype)

def drawFootprint(foot, borderWidth=0.5, origin=None, XY0=None, frame=None, ctype=None, bin=1,
                  peaks=False, symb="+", size=0.4, ctypePeak=None, display="deferToFrame"):
    """Draw an afwDetection::Footprint on a display frame with the specified ctype.  Include an extra borderWidth
pixels If origin is present, it's Added to the Footprint; if XY0 is present is Subtracted from the Footprint

If peaks is True, also show the object's Peaks using the specified symbol and size and ctypePeak

All Footprint coordinates are divided by bin, as is right and proper for overlaying on a binned image
    """

    if XY0:
        if origin:
            raise RuntimeError("You may not specify both origin and XY0")
        origin = (-XY0[0], -XY0[1])

    display = _getDisplayFromDisplayOrFrame(display, frame)
    with display.Buffering():
        borderWidth /= bin
        for s in foot.getSpans():
            y, x0, x1 = s.getY(), s.getX0(), s.getX1()

            if origin:
                x0 += origin[0]; x1 += origin[0]
                y += origin[1]

            x0 /= bin; x1 /= bin; y /= bin

            display.line([(x0 - borderWidth, y - borderWidth),
                      (x0 - borderWidth, y + borderWidth),
                      (x1 + borderWidth, y + borderWidth),
                      (x1 + borderWidth, y - borderWidth),
                      (x0 - borderWidth, y - borderWidth),
                      ], ctype=ctype)

        if peaks:
            for p in foot.getPeaks():
                x, y = p.getIx(), p.getIy()

                if origin:
                    x += origin[0]; y += origin[1]

                x /= bin; y /= bin

                display.dot(symb, x, y, size=size, ctype=ctypePeak)

def drawCoaddInputs(exposure, frame=None, ctype=None, bin=1, display="deferToFrame"):
    """Draw the bounding boxes of input exposures to a coadd on a display frame with the specified ctype,
    assuming display.mtv() has already been called on the given exposure on this frame.


    All coordinates are divided by bin, as is right and proper for overlaying on a binned image
    """
    coaddWcs = exposure.getWcs()
    catalog = exposure.getInfo().getCoaddInputs().ccds

    offset = afwGeom.PointD() - afwGeom.PointD(exposure.getXY0())

    display = _getDisplayFromDisplayOrFrame(display, frame)

    with display.Buffering():
        for record in catalog:
            ccdBox = afwGeom.Box2D(record.getBBox())
            ccdCorners = ccdBox.getCorners()
            coaddCorners = [coaddWcs.skyToPixel(record.getWcs().pixelToSky(point)) + offset
                            for point in ccdCorners]
            display.line([(coaddCorners[i].getX()/bin, coaddCorners[i].getY()/bin)
                      for i in range(-1, 4)], ctype=ctype)

def drawSourceMatches(exposure, sources, matches, frame):
    """Display an Exposure with overplotted sources and reference catalog matches.

    Source positions are indicated by green circles; catalog positions by
    yellow vertical crosses (+); match positions by red diagonal crosses (X).

    @param[in] exposure  Exposure to display (lsst.afw.image.Exposure)
    @param[in] sources   Sources to overplot (lsst.afw.table.SourceCatalog)
    @param[in] matches   Matches to overplot (lsst.afw.table.ReferenceMatch)
    @param[in] frame     Frame to use for display
    """
    disp = _getDisplayFromDisplayOrFrame(None, frame)
    disp.mtv(exposure)
    x0, y0 = exposure.getMaskedImage().getX0(), exposure.getMaskedImage().getY0()
    wcs = exposure.getWcs()

    with disp.Buffering():
        for source in sources:
            x, y = source.getX() - x0, source.getY() - y0
            disp.dot('o', x, y, ctype="green", size=4)

        for first, second, _ in matches:
            catPos = wcs.skyToPixel(first.getCoord())
            x1, y1 = catPos.getX() - x0, catPos.getY() - y0
            disp.dot("+", x1, y1, ctype="yellow", size=8)
            x2, y2 = second.getX() - x0, second.getY() - y0
            disp.dot("x", x2, y2, ctype="red", size=8)
