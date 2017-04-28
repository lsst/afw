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
# \file
# \brief Support for talking to ds9 from python
# \deprecated  New code should use lsst.afw.display and set the backend to ds9

from __future__ import absolute_import, division, print_function

import lsst.afw.display
import lsst.afw.image as afwImage
from .interface import getDisplay as _getDisplay, setDefaultBackend
try:
    loaded
except NameError:
    try:
        setDefaultBackend("lsst.display.ds9")
        getDisplay = _getDisplay
    except Exception as e:
        # No usable version of display_ds9.
        # Let's define a version of getDisplay() which will throw an exception.

        # Prior to DM-9433, we used a closure to implicitly capture the
        # exception being thrown for future reference. Following changes to
        # exception scope rules in Python 3 (see PEP 3110), it's now regarded
        # as clearer to make the capture explicit using the following class.
        class _RaiseException(object):
            def __init__(self, exception):
                # The exception being caught above may have a bulky context which we
                # don't want to capture in a closure for all time: see DM-9504 for a
                # full discussion. We therefore define a new exception of the same
                # type, which just encodes the error text.
                self.exception = type(e)("%s (is display_ds9 setup?)" % e)

            def __call__(self, *args, **kwargs):
                raise self.exception

        getDisplay = _RaiseException(e)

        class DisplayImpl(object):
            __init__ = getDisplay

        loaded = False
    else:
        loaded = True

#
# Backwards compatibility.  Downstream code should be converted to use display.RED etc.
#
from lsst.afw.display import BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, WHITE

def Buffering():
    # always use the real one
    return _getDisplay(None, create=True).Buffering()

#
# Functions provided for backwards compatibility
#


def setMaskPlaneColor(name, color=None, frame=None):
    return getDisplay(frame, create=True).setMaskPlaneColor(name, color)


def getMaskPlaneColor(name, frame=None):
    return getDisplay(frame, create=True).getMaskPlaneColor(name)


def setMaskTransparency(name, frame=None):
    return getDisplay(frame, create=True).setMaskTransparency(name)


def getMaskTransparency(name, frame=None):
    return getDisplay(frame, create=True).getMaskTransparency(name)


def show(frame=None):
    return getDisplay(frame, create=True).show()


def mtv(data, frame=None, title="", wcs=None, *args, **kwargs):
    return getDisplay(frame, create=True).mtv(data, title, wcs, *args, **kwargs)


def erase(frame=None):
    return getDisplay(frame, create=True).erase()


def dot(symb, c, r, frame=None, size=2, ctype=None, origin=afwImage.PARENT, *args, **kwargs):
    return getDisplay(frame, create=True).dot(symb, c, r, size, ctype, origin, *args, **kwargs)


def line(points, frame=None, origin=afwImage.PARENT, symbs=False, ctype=None, size=0.5):
    return getDisplay(frame, create=True).line(points, origin, symbs, ctype, size)


def scale(algorithm, min, max=None, frame=None):
    return getDisplay(frame, create=True).scale(algorithm, min, max)


def pan(colc=None, rowc=None, frame=None, origin=afwImage.PARENT):
    disp = getDisplay(frame, create=True)

    disp.pan(colc, rowc, origin)


def zoom(zoomfac=None, colc=None, rowc=None, frame=None, origin=afwImage.PARENT):
    disp = getDisplay(frame, create=True)

    disp.zoom(zoomfac)
    disp.pan(colc, rowc, origin)


def interact(frame=None):
    return getDisplay(frame, create=True).interact()


def setCallback(k, func=lsst.afw.display.noop_callback, noRaise=False, frame=None):
    return getDisplay(frame, create=True).setCallback(k, noRaise=False)


def getActiveCallbackKeys(onlyActive=True, frame=None):
    return getDisplay(frame, create=True).getActiveCallbackKeys(onlyActive)
