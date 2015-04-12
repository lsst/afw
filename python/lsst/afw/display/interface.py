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
## \brief Support for talking to image displays from python

import os, re, math, sys, time
import importlib
import lsst.afw.geom  as afwGeom
import lsst.afw.image as afwImage

#
# Symbolic names for mask/line colours.  N.b. ds9 supports any X11 colour for masks
#
WHITE = "white"
BLACK = "black"
RED = "red"
GREEN = "green"
BLUE = "blue"
CYAN = "cyan"
MAGENTA = "magenta"
YELLOW = "yellow"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _makeDisplayImpl(display, backend, *args, **kwargs):
    """!Return the DisplayImpl for the named backend

    \param backend Name of desired display.  Should be importable, either absolutely or relative to .
    \param frame  Identifier for this instance of the backend
    \param args   Arguments passed to DisplayImpl.__init__
    \param kwrgs  Keywords arguments passed to DisplayImpl.__init__

    E.g.
         import lsst.afw.display as afwDisplay
         display = afwDisplay.Display("ds9", frame=1)
     would call
         _makeDisplayImpl(..., "ds9", 1)
    and import the ds9 implementation of DisplayImpl from lsst.afw.display.ds9
    """
    _disp = None
    for dt in (backend, ".%s" % backend):
        try:
            _disp = importlib.import_module(dt, package="lsst.afw.display")
            break
        except ImportError as e:
            pass

    if not _disp:
        raise e

    return _disp.DisplayImpl(display, *args, **kwargs)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Display(object):
    def __init__(self, frame, backend=None, *args, **kwargs):
        """!Create an object able to display images and overplot glyphs

        \param frame An identifier for the display
        \param backend The backend to use (defaults to value set by setDefaultBackend())
        \param args Arguments to pass to the backend
        \param kwargs Arguments to pass to the backend
        """
        if backend is None:
            backend = _defaultBackend

        self.frame = frame
        self.impl = _makeDisplayImpl(self, backend, *args, **kwargs)

        self._data = None               # the data displayed on the frame
        self.setMaskTransparency(_defaultMaskTransparency)
        self._maskPlaneColors = {}
        self.setMaskPlaneColor(_defaultMaskPlaneColor)

        self._callbacks = {}

        for ik in range(ord('a'), ord('z') + 1):
            k = "%c" % ik
            self.setCallback(k, noRaise=True)
            self.setCallback(k.upper(), noRaise=True)

        for k in ('Return', 'Shift_L', 'Shift_R'):
            self.setCallback(k)

        for k in ('q', 'Escape'):
            self.setCallback(k, lambda k, x, y: True)

        def _h_callback(k, x, y):
            h_callback(k, x, y)

            for k in sorted(self._callbacks.keys()):
                doc = self._callbacks[k].__doc__
                print "   %-6s %s" % (k, doc.split("\n")[0] if doc else "???")

        self.setCallback('h', _h_callback)

    def __str__(self):
        return str(self.frame)

    def maskColorGenerator(self, omitBW=True):
        """!A generator for "standard" colours

        \param omitBW  Don't include Black and White

        e.g.
        colorGenerator = interface.maskColorGenerator(omitBW=True)
        for p in planeList:
            print p, next(colorGenerator)
        """
        _maskColors = [WHITE, BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]

        i = -1
        while True:
            i += 1
            color = _maskColors[i%len(_maskColors)]
            if omitBW and color in (BLACK, WHITE):
                continue

            yield color

    def setMaskPlaneColor(self, name, color=None):
        """!Request that mask plane name be displayed as color

        \param name Name of mask plane or a dictionary of name -> colourName
        \param color The name of the colour to use (must be None if name is a dict)

        Colours may be specified as any X11-compliant string (e.g. <tt>"orchid"</tt>), or by one
        of the following constants defined in \c afwDisplay: \c BLACK, \c WHITE, \c RED, \c BLUE,
        \c GREEN, \c CYAN, \c MAGENTA, \c YELLOW.

        The advantage of using the symbolic names is that the python interpreter can detect typos.
        
        """

        if isinstance(name, dict):
            assert color == None
            for k, v in name.items():
                self.setMaskPlaneColor(k, v)
            return

        self._maskPlaneColors[name] = color

    def getMaskPlaneColor(self, name):
        """!Return the colour associated with the specified mask plane name"""
        
        return self._maskPlaneColors.get(name)

    def setMaskTransparency(self, transparency=None, name=None):
        """!Specify display's mask transparency (percent); or None to not set it when loading masks"""

        if isinstance(transparency, dict):
            assert name == None
            for k, v in transparency.items():
                self.setMaskTransparency(k, v)
            return

        if transparency is not None and (transparency < 0 or transparency > 100):
            print >> sys.stderr, "Mask transparency should be in the range [0, 100]; clipping"
            if transparency < 0:
                transparency = 0
            else:
                transparency = 100

        if transparency is not None:
            self.impl._setMaskTransparency(transparency, name)

    def getMaskTransparency(self, name=None):
        """!Return the current display's mask transparency"""

        self.impl._getMaskTransparency(name)

    def show(self):
        """!Uniconify and Raise display.  N.b. throws an exception if frame doesn't exit"""
        self.impl._show()

    def mtv(self, data, title="", wcs=None, *args, **kwargs):
        """!Display an Image or Mask on a DISPLAY display

        If lowOrderBits is True, give low-order-bits priority in display (i.e.
        overlay them last)

        Historical note: the name "mtv" comes from Jim Gunn's forth imageprocessing
        system, Mirella (named after Mirella Freni); The "m" stands for Mirella.
        """
        self._data = data

        self.impl._mtv(data, title, wcs, *args, **kwargs)
    #
    # Graphics commands
    #
    class Buffering(object):
        """!A class intended to be used with python's with statement:
    E.g.
        with display.Buffering():
            display.dot("+", xc, yc)
        """
        def __init__(self, display=None):
            self.display = display
        def __enter__(self):
            getDisplay(self.display).impl._buffer(True)
        def __exit__(self, *args):
            getDisplay(self.display).impl._buffer(False)

    def flush():
        """!Flush the buffers"""
        self.impl._flush()

    def erase(self):
        """!Erase the specified DISPLAY frame
        """
        self.impl._erase()

    def dot(self, symb, c, r, size=2, ctype=None, origin=afwImage.PARENT, *args, **kwargs):
        """!Draw a symbol onto the specified DISPLAY frame at (col,row) = (c,r) [0-based coordinates]

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
        if isinstance(symb, int):
            symb = "%d" % (symb)

        if origin == afwImage.PARENT and self._data is not None:
            x0, y0 = self._data.getXY0()
            r -= x0
            c -= y0

        if isinstance(symb, afwGeom.ellipses.BaseCore) or re.search(r"^@:", symb):
            try:
                mat = re.search(r"^@:([^,]+),([^,]+),([^,]+)", symb)
            except TypeError:
                pass
            else:
                if mat:
                    mxx, mxy, myy = [float(_) for _ in mat.groups()]
                    symb = afwGeom.ellipses.Quadrupole(mxx, myy, mxy)

            symb = afwGeom.ellipses.Axes(symb)

        self.impl._dot(symb, c, r, size, ctype, **kwargs)

    def line(self, points, origin=afwImage.PARENT, symbs=False, ctype=None, size=0.5):
        """!Draw a set of symbols or connect the points, a list of (col,row)
    If symbs is True, draw points at the specified points using the desired symbol,
    otherwise connect the dots.  Ctype is the name of a colour (e.g. 'red')

    If symbs supports indexing (which includes a string -- caveat emptor) the elements are used to label the points
        """
        if symbs:
            try:
                symbs[1]
            except:
                symbs = len(points)*list(symbs)

            for i, xy in enumerate(points):
                self.dot(symbs[i], *xy, size=size, ctype=ctype)
        else:
            if len(points) > 0:
                if origin == afwImage.PARENT and self._data is not None:
                    x0, y0 = self._data.getXY0()
                    _points = list(points)  # make a mutable copy
                    for i, p in enumerate(points):
                        _points[i] = (p[0] - x0, p[1] - y0)
                    points = _points

                self.impl._drawLines(points, ctype)
    #
    # Set gray scale
    #
    def scale(self, min=None, max=None, type=None):
        """!Set the scale limits and type in one function call"""
        self.scaleLimits(min, max)
        self.scaleType(type)

    def scaleLimits(self, min, max=None):
        """!Set the range of the scaling from DN in the image to the image display
        \param min Minimum value, or "zscale"
        \param max Maximum value (must be None for zscale)
        \param frame The frame to apply the scaling too
        """    
        if min == "zscale":
            assert max == None, "You may not specify \"zscale\" and max"
            self.impl._setScaleType("zscale")
        else:
            if max is None:
                raise DisplayError("Please specify max")

            self.impl._setScaleLimits(min, max)

    def scaleType(self, name, params=None):
        """!Set the type of scaling from DN in the image to the image display
        \param name Desired scaling (e.g. "linear" or "asinh")
        \param frame The frame to apply the scaling too
        \param params Extra parameters for scaling (e.g. Q for asinh scalings)
        """    
        self.impl._setScaleType(name)

    #
    # Zoom and Pan
    #
    def zoom(self, zoomfac=None, colc=None, rowc=None, origin=afwImage.PARENT):
        """!Zoom frame by specified amount, optionally panning also"""

        if (rowc and colc is None) or (colc and rowc is None):
            raise DisplayError, "Please specify row and column center to pan about"

        if rowc is not None and origin == afwImage.PARENT and self._data is not None:
            x0, y0 = self._data.getXY0()
            rowc -= x0
            colc -= y0

            self.impl._pan(colc, rowc)

        if zoomfac == None and rowc == None:
            zoomfac = 2

        if zoomfac is not None:
            self.impl._zoom(zoomfac)

    def pan(self, colc=None, rowc=None, origin=afwImage.PARENT):
        """!Pan to (rowc, colc); see also zoom"""

        self.zoom(None, colc, rowc, origin)

    def interact(self):
        """!Enter an interactive loop, listening for key presses in display and firing callbacks.

        Exit with q, <carriage return>, or <escape>
    """

        while True:
            ev = self.impl._getEvent()
            if not ev:
                continue
            k, x, y = ev.k, ev.x, ev.y      # for now

            try:
                if self.callbacks[k](k, x, y):
                    break
            except KeyError:
                print >> sys.stderr, "No callback is registered for %s" % k
            except Exception, e:
                print >> sys.stderr, "Display.callbacks[%s](%s, %s, %s) failed: %s" % \
                    (k, k, x, y, e)

    def setCallback(self, k, func=None, noRaise=False):
        """!Set the callback for key k to be func, returning the old callback
        """

        if k in "f":
            if noRaise:
                return
            raise RuntimeError("Key '%s' is already in use by display, so I can't add a callback for it" % k)

        ofunc = self._callbacks.get(k)
        self._callbacks[k] = func if func else noop_callback

        return ofunc

    def getActiveCallbackKeys(self, onlyActive=True):
        """!Return all callback keys
    \param onlyActive  If true only return keys that do something
        """

        return sorted([k for k, func in callbacks.items() if not (onlyActive and func == noop_callback)])
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Callbacks for display events
#
class Event(object):
    """!A class to handle events such as key presses in image display windows"""
    def __init__(self, k, x=float('nan'), y=float('nan')):
        self.k = k
        self.x = x
        self.y = y

    def __str__(self):
        return "%s (%.2f, %.2f)" % (self.k, self.x, self.y)
#
# Default fallback function
#
def noop_callback(k, x, y):
    """!Callback function: arguments key, x, y"""
    return False

def h_callback(k, x, y):
    print "Enter q or <ESC> to leave interactive mode, h for this help, or a letter to fire a callback"
    return False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Handle Displays, including the default one (the frame to use when a user specifies None)
#
# If the default frame is None, image display is disabled
# 
def setDefaultBackend(backend):
    global _defaultBackend
    _defaultBackend = backend

def getDefaultBackend():
    global _defaultBackend
    return _defaultBackend

def setDefaultFrame(frame=0):
    """Set the default frame for display"""
    global _defaultFrame
    _defaultFrame = frame

def getDefaultFrame():
    """Get the default frame for display"""
    return _defaultFrame

def incrDefaultFrame():
    """Increment the default frame for display"""
    global _defaultFrame
    _defaultFrame += 1
    return _defaultFrame

def setDefaultMaskTransparency(maskPlaneTransparency={}):
    global _defaultMaskTransparency
    _defaultMaskTransparency = maskPlaneTransparency.copy()

def setDefaultMaskPlaneColor(name=None, color=None):
    """!Set the default mapping from mask plane names to colours
    \param name name of mask plane, or a dict mapping names to colours
    \param color Desired color, or None if name is a dict

    If name is None, use the hard-coded default dictionary
    """

    if name is None:
        name = dict(
            BAD=RED,
            CR=MAGENTA,
            EDGE=YELLOW,
            INTERPOLATED=GREEN,
            SATURATED=GREEN,
            DETECTED=BLUE,
            DETECTED_NEGATIVE=CYAN,
            SUSPECT=YELLOW,
            # deprecated names
            INTRP=GREEN,
            SAT=GREEN,
        )

    if isinstance(name, dict):
        assert color == None
        for k, v in name.items():
            setDefaultMaskPlaneColor(k, v)
        return
    #
    # Set the individual colour values
    #
    global _defaultMaskPlaneColor
    try:
        _defaultMaskPlaneColor
    except NameError:
        _defaultMaskPlaneColor = {}

    _defaultMaskPlaneColor[name] = color

def getDisplay(frame=None, backend=None, create=True, *args, **kwargs):
    """!Return the Display indexed by frame, creating it if needs be
    \param frame The desired frame (None => use defaultFrame (see setDefaultFrame))
    \param backend  create the specified frame using this backend (or the default if None) if it doesn't already exist.  If backend == "", it's an error to specify a non-existent frame
    """

    if frame is None:
        frame = _defaultFrame

    global _displays
    if not frame in _displays:
        if backend == "":
            raise RuntimeError("Frame %s does not exist" % frame)

        _displays[frame] = Display(frame, backend, *args, **kwargs)

    return _displays[frame]

def delDisplay(frame=None):
    """!Delete the Display indexed by frame
    \param frame The desired frame (None => defaultFrame (see setDefaultFrame); "all" => all)
    """
    global _displays
    if frame.lower() == "all":
        _displays = {}
        return

    if frame is None:
        frame = _defaultFrame
    
    if frame in _displays:
        del _displays[frame]
    else:
        raise RuntimeError("Frame %s does not exist" % frame)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

try:
    _displays
except NameError:
    _displays = {}
    
    setDefaultFrame(0)
    setDefaultMaskTransparency()
    setDefaultMaskPlaneColor()

    setDefaultBackend("virtualDevice")
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Functions provided for backwards compatibility
#
def setMaskPlaneColor(name, color=None, frame=None):
    return getDisplay(frame).setMaskPlaneColor(name, color)

def getMaskPlaneColor(name, frame=None):
    return getDisplay(frame).getMaskPlaneColor(name)

def setMaskTransparency(name, show=True, frame=None):
    return setMaskTransparency(frame).setMaskTransparency(name, show)

def getMaskTransparency(name, frame=None):
    return getDisplay(frame).getMaskTransparency(name)

def show(frame=None):
    return getDisplay(frame).show()

def mtv(data, frame=None, title="", wcs=None, *args, **kwargs):
    return getDisplay(frame).mtv(data, title, wcs, *args, **kwargs)

def erase(frame=None):
    return getDisplay(frame).erase()

def dot(symb, c, r, frame=None, size=2, ctype=None, origin=afwImage.PARENT, *args, **kwargs):
    return getDisplay(frame).dot(symb, c, r, size, ctype, origin, *args, **kwargs)

def line(points, frame=None, origin=afwImage.PARENT, symbs=False, ctype=None, size=0.5):
    return getDisplay(frame).line(points, origin, symbs, ctype, size)

def scaleLimits(min, max=None, frame=None):
    return getDisplay(frame).scaleLimits(min, max)

def scaleType(name, frame=None, params=None):
    return getDisplay(frame).scaleType(name, params)

def zoom(zoomfac=None, colc=None, rowc=None, frame=None, origin=afwImage.PARENT):
    disp = getDisplay(frame)

    disp.zoom(zoomfac)
    disp.pan(colc, rowc, origin)

def interact(frame=None):
    return getDisplay(frame).interact()

def setCallback(k, func=noop_callback, noRaise=False, frame=None):
    return getDisplay(frame).setCallback(k, noRaise=False)

def getActiveCallbackKeys(onlyActive=True, frame=None):
    return getDisplay(frame).getActiveCallbackKeys(onlyActive)
