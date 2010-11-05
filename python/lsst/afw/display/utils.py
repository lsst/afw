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

import math
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Mosaic(object):
    """A class to handle mosaics of one or more identically-sized images (or Masks or MaskedImages)
    E.g.
    m = Mosaic()
    m.setGutter(5)
    m.setBackground(10)
    m.setMode("square")                     # the default; other options are "x" or "y"

    mosaic = m.makeMosaic(im1, im2, im3)    # build the mosaic
    ds9.mtv(mosaic)                         # display it
    m.drawLabels(["Label 1", "Label 2", "Label 3"]) # label the panels

    # alternative way to build a mosaic
    images = [im1, im2, im3]               
    labels = ["Label 1", "Label 2", "Label 3"]

    mosaic = m.makeMosaic(images)
    ds9.mtv(mosaic)
    m.drawLabels(labels)

    # Yet another way to build a mosaic (no need to build the images/labels lists)
    for i in range(len(images)):
        m.append(images[i], labels[i])
    # You may optionally include a colour, e.g. ds9.YELLOW, as a third argument

    mosaic = m.makeMosaic()
    ds9.mtv(mosaic)
    m.drawLabels()

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
        Set may be cleared with Mosaic.reset()"""
        if not self.xsize:
            self.xsize = image.getWidth()
            self.ysize = image.getHeight()

        self.images.append(image)
        self.labels.append((label, ctype))

    def makeMosaic(self, images=None, frame=None, mode=None, title=""):
        """Return a mosaic of all the images provided; if none are specified,
        use the list accumulated with Mosaic.append()
        
        If frame is specified, display it
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

        if not mode:
            mode = self.mode

        if mode == "square":
            nx, ny = 1, self.nImage
            while nx*im.getWidth() < ny*im.getHeight():
                nx += 1
                ny = int(self.nImage/nx)

                if nx*ny < self.nImage:
                    ny += 1
                if nx*ny < self.nImage:
                    nx += 1
                    
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

        mosaic = images[0].Factory(nx*self.xsize + (nx - 1)*self.gutter, ny*self.ysize + (ny - 1)*self.gutter)
        mosaic.set(self.background)

        for i in range(len(images)):
            smosaic = mosaic.Factory(mosaic, self.getBBox(i%nx, i//nx))
            im = images[i]

            if smosaic.getDimensions() != im.getDimensions(): # im is smaller than smosaic
                llc = afwImage.PointI((smosaic.getWidth() - im.getWidth())//2,
                                      (smosaic.getHeight() - im.getHeight())//2)
                smosaic = smosaic.Factory(smosaic, afwImage.BBox(llc, im.getWidth(), im.getHeight()))

            smosaic <<= im

        if frame is not None:
            ds9.mtv(mosaic, frame=frame, title=title)

            if images == self.images:
                self.drawLabels(frame=frame)
            
        return mosaic

    def setGutter(self, gutter):
        """Set the number of pixels between panels in a mosaic"""
        self.gutter = 3

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
            ix, iy = ix % self.nx, ix/self.nx

        return afwImage.BBox(afwImage.PointI(ix*(self.xsize + self.gutter), iy*(self.ysize + self.gutter)),
                             self.xsize, self.ysize)

    def drawLabels(self, labels=None, frame=None):
        """Draw the list labels at the corners of each panel.  If labels is None, use the ones
        specified by Mosaic.append()"""

        if frame is None:
            return

        if not labels:
            labels = self.labels

        if not labels:
            return

        if len(labels) != self.nImage:
            raise RuntimeError, ("You provided %d labels for %d panels" % (len(labels), self.nImage))

        for i in range(len(labels)):
            if labels[i]:
                label, ctype = labels[i], None
                try:
                    label, ctype = label
                except:
                    pass

                if not label:
                    continue
                    
                ds9.dot(str(label), self.getBBox(i).getX0(), self.getBBox(i).getY0(), frame=frame, ctype=ctype)

def drawBBox(bbox, borderWidth=0.0, origin=None, frame=None, ctype=None, bin=1):
    """Draw an afwImage::BBox on a ds9 frame with the specified ctype.  Include an extra borderWidth pixels
If origin is present, it's Added to the BBox

All BBox coordinates are divided by bin, as is right and proper for overlaying on a binned image
    """
    x0, y0 = bbox.getX0(), bbox.getY0()
    x1, y1 = bbox.getX1(), bbox.getY1()

    if origin:
        x0 += origin[0]; x1 += origin[0]
        y0 += origin[1]; y1 += origin[1]

    x0 /= bin; y0 /= bin
    x1 /= bin; y1 /= bin
    borderWidth /= bin
    
    ds9.line([(x0 - borderWidth, y0 - borderWidth),
              (x0 - borderWidth, y1 + borderWidth),
              (x1 + borderWidth, y1 + borderWidth),
              (x1 + borderWidth, y0 - borderWidth),
              (x0 - borderWidth, y0 - borderWidth),
              ], frame=frame, ctype=ctype)
