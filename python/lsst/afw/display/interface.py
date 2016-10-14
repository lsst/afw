from __future__ import print_function
from builtins import range
from builtins import object
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

import re
import sys
import importlib
import lsst.afw.geom  as afwGeom
import lsst.afw.image as afwImage
import lsst.log

logger = lsst.log.Log.getLogger("afw.display.interface")

__all__ = (
    "WHITE", "BLACK", "RED", "GREEN", "BLUE", "CYAN", "MAGENTA", "YELLOW", "ORANGE",
    "Display", "Event", "noop_callback", "h_callback",
    "setDefaultBackend", "getDefaultBackend",
    "setDefaultFrame", "getDefaultFrame", "incrDefaultFrame",
    "setDefaultMaskTransparency", "setDefaultMaskPlaneColor",
    "getDisplay", "delAllDisplays",
)

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
ORANGE = "orange"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def _makeDisplayImpl(display, backend, *args, **kwargs):
    """!Return the DisplayImpl for the named backend

    \param display Name of device.  Should be importable, either absolutely or relative to lsst.display
    \param backend The desired backend
    \param args   Arguments passed to DisplayImpl.__init__
    \param kwargs Keywords arguments passed to DisplayImpl.__init__

    E.g.
         import lsst.afw.display as afwDisplay
         display = afwDisplay.Display("ds9", frame=1)
     would call
         _makeDisplayImpl(..., "ds9", 1)
    and import the ds9 implementation of DisplayImpl from lsst.display.ds9
    """
    _disp = None
    exc = None
    for dt in ("lsst.display.%s" % backend, backend, ".%s" % backend, "lsst.afw.display.%s" % backend):
        exc = None
        # only specify the root package if we are not doing an absolute import
        impargs = {}
        if dt.startswith("."):
            impargs["package"] = "lsst.display"
        try:
            _disp = importlib.import_module(dt, **impargs)
            break
        except (ImportError, SystemError) as e:
            # SystemError can be raised in Python 3.5 if a relative import
            # is attempted when the root package, lsst.display, does not exist.
            # Copy the exception into outer scope
            exc = e

    if not _disp:
        if exc is not None:
            # re-raise the final exception
            raise exc
        else:
            raise ImportError("Could not load the requested backend: {}".format(backend))

    if display:
        return _disp.DisplayImpl(display, *args, **kwargs)
    else:
        return True

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class Display(object):
    _displays = {}
    _defaultBackend = None
    _defaultFrame = 0
    _defaultMaskPlaneColor = dict(
        BAD=RED,
        CR=MAGENTA,
        EDGE=YELLOW,
        INTERPOLATED=GREEN,
        SATURATED=GREEN,
        DETECTED=BLUE,
        DETECTED_NEGATIVE=CYAN,
        SUSPECT=YELLOW,
        NO_DATA=ORANGE,
        # deprecated names
        INTRP=GREEN,
        SAT=GREEN,
    )
    _defaultMaskTransparency = {}

    def __init__(self, frame=None, backend=None, *args, **kwargs):
        """!Create an object able to display images and overplot glyphs

        \param frame An identifier for the display
        \param backend The backend to use (defaults to value set by setDefaultBackend())
        \param args Arguments to pass to the backend
        \param kwargs Arguments to pass to the backend
        """
        if frame is None:
            frame = getDefaultFrame()

        if backend is None:
            if Display._defaultBackend is None:
                try:
                    setDefaultBackend("ds9")
                except RuntimeError:
                    setDefaultBackend("virtualDevice")

            backend = Display._defaultBackend

        self.frame = frame
        self._impl = _makeDisplayImpl(self, backend, *args, **kwargs)
        self.name = backend

        self._xy0 = None                # the data displayed on the frame's XY0
        self.setMaskTransparency(Display._defaultMaskTransparency)
        self._maskPlaneColors = {}
        self.setMaskPlaneColor(Display._defaultMaskPlaneColor)

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
                print("   %-6s %s" % (k, doc.split("\n")[0] if doc else "???"))

        self.setCallback('h', _h_callback)

        Display._displays[frame] = self

    def __enter__(self):
        """!Support for python's with statement"""
        return self

    def __exit__(self, *args):
        """!Support for python's with statement"""
        self.close()

    def __del__(self):
        self.close()

    def __getattr__(self, name, *args, **kwargs):
        """Try to call self._impl.name(*args, *kwargs)"""
        
        if not (hasattr(self, "_impl") and self._impl):
            raise AttributeError("Device has no _impl attached")
        #
        # We need a wrapper to get the arguments passed through
        #
        try:
            attr = getattr(self._impl, name)
        except AttributeError:
            raise AttributeError("Device %s has no attribute \"%s\"" % (self.name, name))

        def wrapper(*args, **kwargs):
            return attr(*args, **kwargs)
        
        return wrapper

    def close(self):
        if hasattr(self, "_impl") and self._impl:
            del self._impl
            self._impl = None

        if self.frame in Display._displays:
            del Display._displays[self.frame]

    @property
    def verbose(self):
        """!The backend's verbosity"""
        return self._impl.verbose

    @verbose.setter
    def verbose(self, value):
        if self._impl:
            self._impl.verbose = value

    def __str__(self):
        return "Display[%s]" % (self.frame)

    #
    # Handle Displays, including the default one (the frame to use when a user specifies None)
    #
    @staticmethod
    def setDefaultBackend(backend):
        try:
            _makeDisplayImpl(None, backend)
        except Exception as e:
            raise RuntimeError("Unable to set backend to %s: \"%s\"" % (backend, e))

        Display._defaultBackend = backend

    @staticmethod
    def getDefaultBackend():
        return Display._defaultBackend

    @staticmethod
    def setDefaultFrame(frame=0):
        """Set the default frame for display"""
        Display._defaultFrame = frame

    @staticmethod
    def getDefaultFrame():
        """Get the default frame for display"""
        return Display._defaultFrame

    @staticmethod
    def incrDefaultFrame():
        """Increment the default frame for display"""
        Display._defaultFrame += 1
        return Display._defaultFrame

    @staticmethod
    def setDefaultMaskTransparency(maskPlaneTransparency={}):
        if hasattr(maskPlaneTransparency, "copy"):
            maskPlaneTransparency = maskPlaneTransparency.copy()

        Display._defaultMaskTransparency = maskPlaneTransparency

    @staticmethod
    def setDefaultMaskPlaneColor(name=None, color=None):
        """!Set the default mapping from mask plane names to colours
        \param name name of mask plane, or a dict mapping names to colours
        \param color Desired color, or None if name is a dict

        If name is None, use the hard-coded default dictionary
        """

        if name is None:
            name = Display._defaultMaskPlaneColor

        if isinstance(name, dict):
            assert color == None
            for k, v in name.items():
                setDefaultMaskPlaneColor(k, v)
            return
        #
        # Set the individual colour values
        #
        Display._defaultMaskPlaneColor[name] = color

    @staticmethod
    def getDisplay(frame=None, backend=None, create=True, verbose=False, *args, **kwargs):
        """!Return the Display indexed by frame, creating it if needs be

        \param frame The desired frame (None => use defaultFrame (see setDefaultFrame))
        \param backend  create the specified frame using this backend (or the default if None) \
        if it doesn't already exist.  If backend == "", it's an error to specify a non-existent frame
        \param create create the display if it doesn't already exist.
        \param verbose Allow backend to be chatty
        \param args arguments passed to Display constructor
        \param kwargs keyword arguments passed to Display constructor
        """

        if frame is None:
            frame = Display._defaultFrame

        if not frame in Display._displays:
            if backend == "":
                raise RuntimeError("Frame %s does not exist" % frame)

            Display._displays[frame] = Display(frame, backend, verbose=verbose, *args, **kwargs)

        Display._displays[frame].verbose = verbose
        return Display._displays[frame]

    @staticmethod
    def delAllDisplays():
        """!Delete and close all known display
        """
        for disp in list(Display._displays.values()):
            disp.close()
        Display._displays = {}

    def maskColorGenerator(self, omitBW=True):
        """!A generator for "standard" colours

        \param omitBW  Don't include Black and White

        e.g.
        colorGenerator = interface.maskColorGenerator(omitBW=True)
        for p in planeList:
            print p, next(colorGenerator)
        """
        _maskColors = [WHITE, BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, ORANGE]

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
            print("Mask transparency should be in the range [0, 100]; clipping", file=sys.stderr)
            if transparency < 0:
                transparency = 0
            else:
                transparency = 100

        if transparency is not None:
            self._impl._setMaskTransparency(transparency, name)

    def getMaskTransparency(self, name=None):
        """!Return the current display's mask transparency"""

        self._impl._getMaskTransparency(name)

    def show(self):
        """!Uniconify and Raise display.  N.b. throws an exception if frame doesn't exit"""
        self._impl._show()

    def mtv(self, data, title="", wcs=None):
        """!Display an Image or Mask on a DISPLAY display

        Historical note: the name "mtv" comes from Jim Gunn's forth imageprocessing
        system, Mirella (named after Mirella Freni); The "m" stands for Mirella.
        """
        if hasattr(data, "getXY0"):
            self._xy0 = data.getXY0()
        else:
            self._xy0 = None

        if re.search("::Exposure<", repr(data)): # it's an Exposure; display the MaskedImage with the WCS
            if wcs:
                raise RuntimeError("You may not specify a wcs with an Exposure")
            data, wcs = data.getMaskedImage(), data.getWcs()
        elif re.search("::DecoratedImage<", repr(data)): # it's a DecoratedImage; display it
            data, wcs = data.getImage(), afwImage.makeWcs(data.getMetadata())
            self._xy0 = data.getXY0()   # DecoratedImage doesn't have getXY0()

        if re.search("::Image<", repr(data)): # it's an Image; display it
            self._impl._mtv(data, None, wcs, title)
        elif re.search("::Mask<", repr(data)): # it's a Mask; display it, bitplane by bitplane
            #
            # Some displays can't display a Mask without an image; so display an Image too,
            # with pixel values set to the mask
            #
            self._impl._mtv(afwImage.ImageU(data.getArray()), data, wcs, title)
        elif re.search("::MaskedImage<", repr(data)): # it's a MaskedImage; display Image and overlay Mask
            self._impl._mtv(data.getImage(), data.getMask(True), wcs, title)
        else:
            raise RuntimeError("Unsupported type %s" % repr(data))
    #
    # Graphics commands
    #
    class _Buffering(object):
        """A class intended to be used with python's with statement"""
        def __init__(self, _impl):
            self._impl = _impl
        def __enter__(self):
            self._impl._buffer(True)
        def __exit__(self, *args):
            self._impl._buffer(False)
            self._impl._flush()

    def Buffering(self):
        """Return a class intended to be used with python's with statement
    E.g.
        with display.Buffering():
            display.dot("+", xc, yc)
        """
        return self._Buffering(self._impl)

    def flush(self):
        """!Flush the buffers"""
        self._impl._flush()

    def erase(self):
        """!Erase the specified DISPLAY frame
        """
        self._impl._erase()

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

        if origin == afwImage.PARENT and self._xy0 is not None:
            x0, y0 = self._xy0
            r -= y0
            c -= x0

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

        self._impl._dot(symb, c, r, size, ctype, **kwargs)

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
                if origin == afwImage.PARENT and self._xy0 is not None:
                    x0, y0 = self._xy0
                    _points = list(points)  # make a mutable copy
                    for i, p in enumerate(points):
                        _points[i] = (p[0] - x0, p[1] - y0)
                    points = _points

                self._impl._drawLines(points, ctype)
    #
    # Set gray scale
    #
    def scale(self, algorithm, min, max=None, unit=None, *args, **kwargs):
        """!Set the range of the scaling from DN in the image to the image display
        \param algorithm Desired scaling (e.g. "linear" or "asinh")
        \param min Minimum value, or "minmax" or "zscale"
        \param max Maximum value (must be None for minmax|zscale)
        \param unit Units for min and max (e.g. Percent, Absolute, Sigma; None if min==minmax|zscale)
        \param *args Optional arguments
        \param **kwargs Optional keyword arguments
        """
        if min in ("minmax", "zscale"):
            assert max == None, "You may not specify \"%s\" and max" % min
            assert unit == None, "You may not specify \"%s\" and unit" % min
        elif max is None:
            raise RuntimeError("Please specify max")

        self._impl._scale(algorithm, min, max, unit, *args, **kwargs)
    #
    # Zoom and Pan
    #
    def zoom(self, zoomfac=None, colc=None, rowc=None, origin=afwImage.PARENT):
        """!Zoom frame by specified amount, optionally panning also"""

        if (rowc and colc is None) or (colc and rowc is None):
            raise RuntimeError("Please specify row and column center to pan about")

        if rowc is not None:
            if origin == afwImage.PARENT and self._xy0 is not None:
                x0, y0 = self._xy0
                rowc -= x0
                colc -= y0

            self._impl._pan(colc, rowc)

        if zoomfac == None and rowc == None:
            zoomfac = 2

        if zoomfac is not None:
            self._impl._zoom(zoomfac)

    def pan(self, colc=None, rowc=None, origin=afwImage.PARENT):
        """!Pan to (rowc, colc); see also zoom"""

        self.zoom(None, colc, rowc, origin)

    def interact(self):
        """!Enter an interactive loop, listening for key presses in display and firing callbacks.
            Exit with q, \c CR, \c ESC, or any other callback function that returns a ``True`` value.
        """
        interactFinished = False

        while not interactFinished:
            ev = self._impl._getEvent()
            if not ev:
                continue
            k, x, y = ev.k, ev.x, ev.y      # for now

            if k not in self._callbacks:
                logger.warn("No callback registered for {0}".format(k))
            else:
                try:
                    interactFinished = self._callbacks[k](k, x, y)
                except Exception as e:
                    logger.error("Display._callbacks[{0}]({0},{1},{2}) failed: {3}".format(k, x, y, e))

    def setCallback(self, k, func=None, noRaise=False):
        """!Set the callback for key k to be func, returning the old callback
        """

        if k in "f":
            if noRaise:
                return
            raise RuntimeError(
                "Key '%s' is already in use by display, so I can't add a callback for it" % k)

        ofunc = self._callbacks.get(k)
        self._callbacks[k] = func if func else noop_callback

        self._impl._setCallback(k, self._callbacks[k])

        return ofunc

    def getActiveCallbackKeys(self, onlyActive=True):
        """!Return all callback keys
    \param onlyActive  If true only return keys that do something
        """

        return sorted([k for k, func in self._callbacks.items() if
                       not (onlyActive and func == noop_callback)])

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
    print("Enter q or <ESC> to leave interactive mode, h for this help, or a letter to fire a callback")
    return False

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Handle Displays, including the default one (the frame to use when a user specifies None)
#
# If the default frame is None, image display is disabled
#
def setDefaultBackend(backend):
    Display.setDefaultBackend(backend)

def getDefaultBackend():
    return Display.getDefaultBackend()

def setDefaultFrame(frame=0):
    return Display.setDefaultFrame(frame)

def getDefaultFrame():
    """Get the default frame for display"""
    return Display.getDefaultFrame()

def incrDefaultFrame():
    """Increment the default frame for display"""
    return Display.incrDefaultFrame()

def setDefaultMaskTransparency(maskPlaneTransparency={}):
    return Display.setDefaultMaskTransparency(maskPlaneTransparency)

def setDefaultMaskPlaneColor(name=None, color=None):
    """!Set the default mapping from mask plane names to colours
    \param name name of mask plane, or a dict mapping names to colours
    \param color Desired color, or None if name is a dict

    If name is None, use the hard-coded default dictionary
    """

    return Display.setDefaultMaskPlaneColor(name, color)

def getDisplay(frame=None, backend=None, create=True, verbose=False, *args, **kwargs):
    """!Return the Display indexed by frame, creating it if needs be

    See Display.getDisplay

    \param frame The desired frame (None => use defaultFrame (see setDefaultFrame))
    \param backend  create the specified frame using this backend (or the default if None) \
    if it doesn't already exist.  If backend == "", it's an error to specify a non-existent frame
    \param create create the display if it doesn't already exist.
    \param verbose Allow backend to be chatty
    \param args arguments passed to Display constructor
    \param kwargs keyword arguments passed to Display constructor
    """

    return Display.getDisplay(frame, backend, create, verbose, *args, **kwargs)

def delAllDisplays():
    """!Delete and close all known display
    """
    return Display.delAllDisplays()
