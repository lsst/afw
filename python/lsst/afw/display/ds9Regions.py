from __future__ import division
#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2015 LSST Corporation.
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

##
## \file
## \brief Convert the display primitives into lists of ds9 region commands
##
## See e.g. http://ds9.si.edu/doc/ref/region.html

import math
import re
import lsst.afw.geom as afwGeom

def dot(symb, c, r, size, ctype=None, fontFamily="helvetica", textAngle=None):
    """Draw a symbol onto the specified DS9 frame at (col,row) = (c,r) [0-based coordinates]
Possible values are:
        +                Draw a +
        x                Draw an x
        *                Draw a *
        o                Draw a circle
        @:Mxx,Mxy,Myy    Draw an ellipse with moments (Mxx, Mxy, Myy) (argument size is ignored)
        An object derived from afwGeom.ellipses.BaseCore Draw the ellipse (argument size is ignored)
Any other value is interpreted as a string to be drawn. Strings obey the fontFamily (which may be extended
with other characteristics, e.g. "times bold italic".  Text will be drawn rotated by textAngle (textAngle is
ignored otherwise).

N.b. objects derived from BaseCore include Axes and Quadrupole.
"""
    if ctype == None:
        color = ""                       # the default
    else:
        color = ' # color=%s' % ctype

    regions = []

    r += 1
    c += 1                      # ds9 uses 1-based coordinates
    if isinstance(symb, afwGeom.ellipses.Axes):
        regions.append('ellipse %g %g %gi %gi %g%s' % (c, r, symb.getA(), symb.getB(),
                                                     math.degrees(symb.getTheta()), color))
    elif symb == '+':
        regions.append('line %g %g %g %g%s' % (c, r+size, c, r-size, color))
        regions.append('line %g %g %g %g%s' % (c-size, r, c+size, r, color))
    elif symb == 'x':
        size = size/math.sqrt(2)
        regions.append('line %g %g %g %g%s' % (c+size, r+size, c-size, r-size, color))
        regions.append('line %g %g %g %g%s' % (c-size, r+size, c+size, r-size, color))
    elif symb == '*':
        size30 = 0.5*size
        size60 = 0.5*math.sqrt(3)*size
        regions.append('line %g %g %g %g%s' % (c+size, r, c-size, r, color))
        regions.append('line %g %g %g %g%s' % (c-size30, r+size60, c+size30, r-size60, color))
        regions.append('line %g %g %g %g%s' % (c+size30, r+size60, c-size30, r-size60, color))
    elif symb == 'o':
        regions.append('circle %g %g %gi%s' % (c, r, size, color))
    else:
        color = re.sub("^ # ", "", color) # skip the leading " # "

        angle = ""
        if textAngle is not None:
            angle += " textangle=%.1f"%(textAngle)

        font = ""
        if size != 2 or fontFamily != "helvetica":
            fontFamily = fontFamily.split()
            font += ' font="%s %d' % (fontFamily.pop(0), int(10*size/2.0 + 0.5))
            if not fontFamily:
                fontFamily = ["normal"] # appears to be needed at least for 7.4b1
            font += " %s" % " ".join(fontFamily)
            font += '"'
        extra = ""
        if color or angle or font:
            extra = " # "
            extra += color
            extra += angle
            extra += font

        regions.append('text %g %g \"%s\"%s' % (c, r, symb, extra))

    return regions

def drawLines(points, ctype=None):
    """!Draw a line by connecting the points
    \param points a list of (col,row)
    \param ctype the name of the desired colour (e.g. 'red', 'orchid')
    """

    if ctype == None:                # default
        color = ""
    else:
        color = "# color=%s" % ctype

    regions = []
    if len(points) > 0:
        c0, r0 = points[0]
        r0 += 1
        c0 += 1             # ds9 uses 1-based coordinates
        for (c, r) in points[1:]:
            r += 1
            c += 1            # ds9 uses 1-based coordinates
            regions.append('line %g %g %g %g %s' % (c0, r0, c, r, color))
            c0, r0 = c, r

    return regions

