#############################################################
"""
CCDGeom manages the source image geometry.
"""
__all__ = ["CCDGeom"]

import math
import numarray

DegToRad = math.pi / 180.0

class CCDGeom:
    #---------------------------------------------------------
    def __init__(self, xOff, yOff, rot, flipx, flipy):
        """
        CCDGeom initialization

        Input
            xOff        x offset of CCD (0,0) pixel from focalplane center. Units: pixels
            yOff        y offset of CCD (0,0) pixel from focalplane center. Units: pixels
            rot         rotation of CCD chip wrt focalplane. Units: degrees
            flipx       CCD chip x axis is inverted
            flipy       CCD chip y axis is inverted

        Return
            None
        """
        self.xOff = xOff
        self.yOff = yOff
        self.flipx = flipx
        self.flipy = flipy
        self.rot = rot
        self.rOff = numarray.array([self.xOff, self.yOff])
        return

    #---------------------------------------------------------
    def setSkyParams(self, posAngle, pixScale):
        """
        Stores the position angle and pixel scale for the source image.

        Input
            posAngle        position angle of source image;
                            Format: float; Units: degrees; Default: none; 
            pixScale        pixel scale of source image;
                            Format: float; Units: arcsec/pixel; Default: none; 
        Return
            none
        """
        self.posAngle = posAngle
        self.pixScale = pixScale
        return

    #---------------------------------------------------------
    def xyToXiEta(self, x, y):
        """
        Apply the transformation implied by xOff, yOff, rot, posAngle, 
        flipx, flipy - to get pixel coords in frame with same orientation 
        as xi, eta.
        Then apply pixscale to go from pixel units to arcsec.

        Input
            x       x axis pixel location of source image
            y       y axis pixel location of source image
        Return
            xi      tangent plane x coordinate of source
            eta     tangent plane y coordinate of source
        """
        r = numarray.array([x, y])
        rFlip = self._Flip(r)
        rFP = self._Rot(rFlip, self.rot) + self.rOff
#        (xi, eta) = self.pixScale*self._Rot(rFP, -self.posAngle)
        (xi, eta) = self.pixScale*self._Rot(rFP, 0)
        return (xi, eta)

    #---------------------------------------------------------
    def _Rot(self, r, theta):
        """
        Returns the vector r=(x, y) rotated by theta (deg)

        Input
            r           (x,y) vector; 
                        Format: list; Default: none.
            theta       amount to rotate vector r; 
                        Format: float; Default: none; Units: degrees.
        Return
            rotatedr    vector r rotated by theta degrees;
                        Format: float numarray; Units:  .....
        """
        costh = math.cos(theta*DegToRad)
        sinth = math.sin(theta*DegToRad)
        rotM = numarray.array([[costh, -sinth],[sinth, costh]])
        return numarray.matrixmultiply(rotM, r)

    #---------------------------------------------------------
    def _Flip(self, r):
        """
        Flips the sign of x and/or y  if the corresponding self.flipx
        and/or self.flipy is set.

        Input
            r           (x,y) ; Format: list; Default: none.
        Return
            rprime      possibly-flipped vector r;
                        Format: numarray; Default: (x,y)
        """
        (x, y) = r
        if (self.flipx):
            x = -x
        if (self.flipy):
            y = -y
        return numarray.array([x,y])
    #---------------------------------------------------------
