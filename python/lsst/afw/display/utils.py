## \file
## \brief Utilities to use with displaying images

import math
import lsst.afw.image as afwImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Mosaic(object):
    """A class to handle mosaics of one or more identically-sized images (or Masks or MaskedImages)
    E.g.
    m = Mosaic()
    m.setGutter(5)
    m.setBackground(10)
    m.setMode("square")                   # the default
    mosaic = m.makeMosaic(im1, im2, im3)

    You can return the (ix, iy)th (or nth) BBox with getBBox()
    """
    def __init__(self, gutter=3, background=0, mode="square"):
        self.gutter = 3                 # number of pixels between panels in a mosaic
        self.background = background    # value in gutters
        self.setMode(mode)              # mosaicing mode
        self.xsize = 0                  # column size of panels
        self.ysize = 0                  # row size of panels

    def makeMosaic(self, images):
        """Return a mosaic of all the images provided"""
        nImage = len(images)
        if nImage == 0:
            raise RuntimeError, "You must provide at least one image"

        image1 = images[0]
        self.xsize, self.ysize = image1.getWidth(), image1.getHeight()

        if self.mode == "square":
            nx = math.sqrt(nImage)
            nx, ny = int(nx), int(nImage/nx)
            if nx*ny < nImage:
                nx += 1
            if nx*ny < nImage:
                ny += 1
            assert(nx*ny >= nImage)
        elif self.mode == "x":
            nx, ny = nImage, 1
        elif self.mode == "y":
            nx, ny = 1, nImage
        else:
            raise RuntimeError, ("Unknown mosaicing mode: %s" % self.mode)

        self.nx, self.ny = nx, ny

        mosaic = image1.Factory(nx*self.xsize + (nx - 1)*self.gutter, ny*self.ysize + (ny - 1)*self.gutter)
        mosaic.set(self.background)

        for i in range(len(images)):
            smosaic = mosaic.Factory(mosaic, self.getBBox(i%nx, i//nx))
            smosaic <<= images[i]
            
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
            ix, iy = ix%self.nx, ix//self.nx

        return afwImage.BBox(afwImage.PointI(ix*(self.xsize + self.gutter), iy*(self.ysize + self.gutter)),
                             self.xsize, self.ysize)
