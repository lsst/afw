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

__all__ = [
    "WHITE", "BLACK", "RED", "GREEN", "BLUE", "CYAN", "MAGENTA", "YELLOW", "ORANGE", "IGNORE",
    "Display", "Event", "noop_callback", "h_callback",
    "setDefaultBackend", "getDefaultBackend",
    "setDefaultFrame", "getDefaultFrame", "incrDefaultFrame",
    "setDefaultMaskTransparency", "setDefaultMaskPlaneColor",
    "getDisplay", "delAllDisplays",
]

import re
import sys
import importlib
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.log

logger = lsst.log.Log.getLogger("afw.display.interface")

#
# Symbolic names for mask/line colors.  N.b. ds9 supports any X11 color for masks
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
IGNORE = "ignore"


def _makeDisplayImpl(display, backend, *args, **kwargs):
    """Return the ``DisplayImpl`` for the named backend

    Parameters
    ----------
    display : `str`
        Name of device. Should be importable, either absolutely or relative to lsst.display
    backend : `str`
        The desired backend
    *args
        Arguments passed to DisplayImpl.__init__
    *kwargs
        Keywords arguments passed to DisplayImpl.__init__

    Examples
    --------
    E.g.
    .. code-block:: py

         import lsst.afw.display as afwDisplay
         display = afwDisplay.Display(display=1, backend="ds9")
     would call
    .. code-block:: py

         _makeDisplayImpl(..., "ds9", 1)
    and import the ds9 implementation of ``DisplayImpl`` from `lsst.display.ds9`
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

    if not _disp or not hasattr(_disp.DisplayImpl, "_show"):
        if exc is not None:
            # re-raise the final exception
            raise exc
        else:
            raise ImportError(
                "Could not load the requested backend: {}".format(backend))

    if display:
        _impl = _disp.DisplayImpl(display, *args, **kwargs)
        if not hasattr(_impl, "frame"):
            _impl.frame = display.frame

        return _impl
    else:
        return True


class Display:
    """Create an object able to display images and overplot glyphs

    Parameters
    ----------
    frame
        An identifier for the display
    backend : `str`
        The backend to use (defaults to value set by setDefaultBackend())
    *args
        Arguments to pass to the backend
    **kwargs
        Arguments to pass to the backend
    """
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
    _defaultImageColormap = "gray"

    def __init__(self, frame=None, backend=None, *args, **kwargs):
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

        self._xy0 = None                # displayed data's XY0
        self.setMaskTransparency(Display._defaultMaskTransparency)
        self._maskPlaneColors = {}
        self.setMaskPlaneColor(Display._defaultMaskPlaneColor)
        self.setImageColormap(Display._defaultImageColormap)

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
        """Support for python's with statement
        """
        return self

    def __exit__(self, *args):
        """Support for python's with statement
        """
        self.close()

    def __del__(self):
        self.close()

    def __getattr__(self, name):
        """Return the attribute of ``self._impl``, or ``._impl`` if it is requested

        Parameters:
        -----------
        name : `str`
            name of the attribute requested

        Returns:
        --------
        attribute : `object`
            the attribute of self._impl for the requested name
        """

        if name == '_impl':
            return object.__getattr__(self, name)

        if not (hasattr(self, "_impl") and self._impl):
            raise AttributeError("Device has no _impl attached")

        try:
            return getattr(self._impl, name)
        except AttributeError:
            raise AttributeError(
                "Device %s has no attribute \"%s\"" % (self.name, name))

    def close(self):
        if getattr(self, "_impl", None) is not None:
            self._impl._close()
            del self._impl
            self._impl = None

        if self.frame in Display._displays:
            del Display._displays[self.frame]

    @property
    def verbose(self):
        """The backend's verbosity
        """
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
            raise RuntimeError(
                "Unable to set backend to %s: \"%s\"" % (backend, e))

        Display._defaultBackend = backend

    @staticmethod
    def getDefaultBackend():
        return Display._defaultBackend

    @staticmethod
    def setDefaultFrame(frame=0):
        """Set the default frame for display
        """
        Display._defaultFrame = frame

    @staticmethod
    def getDefaultFrame():
        """Get the default frame for display
        """
        return Display._defaultFrame

    @staticmethod
    def incrDefaultFrame():
        """Increment the default frame for display
        """
        Display._defaultFrame += 1
        return Display._defaultFrame

    @staticmethod
    def setDefaultMaskTransparency(maskPlaneTransparency={}):
        if hasattr(maskPlaneTransparency, "copy"):
            maskPlaneTransparency = maskPlaneTransparency.copy()

        Display._defaultMaskTransparency = maskPlaneTransparency

    @staticmethod
    def setDefaultMaskPlaneColor(name=None, color=None):
        """Set the default mapping from mask plane names to colors

        Parameters
        ----------
        name : `str` or `dict`
            name of mask plane, or a dict mapping names to colors
            If name is `None`, use the hard-coded default dictionary
        color
            Desired color, or `None` if name is a dict
        """

        if name is None:
            name = Display._defaultMaskPlaneColor

        if isinstance(name, dict):
            assert color is None
            for k, v in name.items():
                setDefaultMaskPlaneColor(k, v)
            return
        #
        # Set the individual color values
        #
        Display._defaultMaskPlaneColor[name] = color

    @staticmethod
    def setDefaultImageColormap(cmap):
        """Set the default colormap for images

        Parameters
        ----------
        cmap : `str`
            Name of colormap, as interpreted by the backend

        Notes
        -----
        The only colormaps that all backends are required to honor
        (if they pay any attention to setImageColormap) are "gray" and "grey"
        """

        Display._defaultImageColormap = cmap

    def setImageColormap(self, cmap):
        """Set the colormap to use for images

         Parameters
        ----------
        cmap : `str`
            Name of colormap, as interpreted by the backend

        Notes
        -----
        The only colormaps that all backends are required to honor
        (if they pay any attention to setImageColormap) are "gray" and "grey"
        """

        self._impl._setImageColormap(cmap)

    @staticmethod
    def getDisplay(frame=None, backend=None, create=True, verbose=False, *args, **kwargs):
        """Return a specific `Display`, creating it if need be

        Parameters
        ----------
        frame
            The desired frame (`None` => use defaultFrame (see `~Display.setDefaultFrame`))
        backend : `str`
            create the specified frame using this backend (or the default if
            `None`) if it doesn't already exist. If ``backend == ""``, it's an
            error to specify a non-existent ``frame``.
        create : `bool`
            create the display if it doesn't already exist.
        verbose : `bool`
            Allow backend to be chatty
        *args
            arguments passed to `Display` constructor
        **kwargs
            keyword arguments passed to `Display` constructor
        """

        if frame is None:
            frame = Display._defaultFrame

        if frame not in Display._displays:
            if backend == "":
                raise RuntimeError("Frame %s does not exist" % frame)

            Display._displays[frame] = Display(
                frame, backend, verbose=verbose, *args, **kwargs)

        Display._displays[frame].verbose = verbose
        return Display._displays[frame]

    @staticmethod
    def delAllDisplays():
        """Delete and close all known displays
        """
        for disp in list(Display._displays.values()):
            disp.close()
        Display._displays = {}

    def maskColorGenerator(self, omitBW=True):
        """A generator for "standard" colors

        Parameters
        ----------
        omitBW : `bool`
            Don't include `BLACK` and `WHITE`

        Examples
        --------

        .. code-block:: py

           colorGenerator = interface.maskColorGenerator(omitBW=True)
           for p in planeList:
               print p, next(colorGenerator)
        """
        _maskColors = [WHITE, BLACK, RED, GREEN,
                       BLUE, CYAN, MAGENTA, YELLOW, ORANGE]

        i = -1
        while True:
            i += 1
            color = _maskColors[i%len(_maskColors)]
            if omitBW and color in (BLACK, WHITE):
                continue

            yield color

    def setMaskPlaneColor(self, name, color=None):
        """Request that mask plane name be displayed as color

        Parameters
        ----------
        name : `str` or `dict`
            Name of mask plane or a dictionary of name -> colorName
        color : `str`
            The name of the color to use (must be `None` if ``name`` is a `dict`)

            Colors may be specified as any X11-compliant string (e.g. `"orchid"`), or by one
            of the following constants in `lsst.afw.display` : `BLACK`, `WHITE`, `RED`, `BLUE`,
            `GREEN`, `CYAN`, `MAGENTA`, `YELLOW`.

            If the color is "ignore" (or `IGNORE`) then that mask plane is not displayed

            The advantage of using the symbolic names is that the python interpreter can detect typos.
        """

        if isinstance(name, dict):
            assert color is None
            for k, v in name.items():
                self.setMaskPlaneColor(k, v)
            return

        self._maskPlaneColors[name] = color

    def getMaskPlaneColor(self, name=None):
        """Return the color associated with the specified mask plane name

        Parameters
        ----------
        name : `str`
            Desired mask plane; if `None`, return entire dict
        """

        if name is None:
            return self._maskPlaneColors
        else:
            return self._maskPlaneColors.get(name)

    def setMaskTransparency(self, transparency=None, name=None):
        """Specify display's mask transparency (percent); or `None` to not set it when loading masks
        """

        if isinstance(transparency, dict):
            assert name is None
            for k, v in transparency.items():
                self.setMaskTransparency(v, k)
            return

        if transparency is not None and (transparency < 0 or transparency > 100):
            print(
                "Mask transparency should be in the range [0, 100]; clipping", file=sys.stderr)
            if transparency < 0:
                transparency = 0
            else:
                transparency = 100

        if transparency is not None:
            self._impl._setMaskTransparency(transparency, name)

    def getMaskTransparency(self, name=None):
        """Return the current display's mask transparency
        """

        return self._impl._getMaskTransparency(name)

    def show(self):
        """Uniconify and Raise display.

        Notes
        -----
        Throws an exception if frame doesn't exit
        """
        return self._impl._show()

    def mtv(self, data, title="", wcs=None):
        """Display an `~lsst.afw.image.Image` or `~lsst.afw.image.Mask` on a display

        Notes
        -----
        Historical note: the name "mtv" comes from Jim Gunn's forth imageprocessing
        system, Mirella (named after Mirella Freni); The "m" stands for Mirella.
        """
        if hasattr(data, "getXY0"):
            self._xy0 = data.getXY0()
        else:
            self._xy0 = None

        # it's an Exposure; display the MaskedImage with the WCS
        if isinstance(data, afwImage.Exposure):
            if wcs:
                raise RuntimeError(
                    "You may not specify a wcs with an Exposure")
            data, wcs = data.getMaskedImage(), data.getWcs()
        elif isinstance(data, afwImage.DecoratedImage):  # it's a DecoratedImage; display it
            try:
                wcs = afwGeom.makeSkyWcs(data.getMetadata())
            except TypeError:
                wcs = None
            data = data.image

            self._xy0 = data.getXY0()   # DecoratedImage doesn't have getXY0()

        if isinstance(data, afwImage.Image):  # it's an Image; display it
            self._impl._mtv(data, None, wcs, title)
        # it's a Mask; display it, bitplane by bitplane
        elif isinstance(data, afwImage.Mask):
            #
            # Some displays can't display a Mask without an image; so display an Image too,
            # with pixel values set to the mask
            #
            self._impl._mtv(afwImage.ImageI(data.getArray()), data, wcs, title)
        # it's a MaskedImage; display Image and overlay Mask
        elif isinstance(data, afwImage.MaskedImage):
            self._impl._mtv(data.getImage(), data.getMask(), wcs, title)
        else:
            raise RuntimeError("Unsupported type %s" % repr(data))
    #
    # Graphics commands
    #

    class _Buffering:
        """A class intended to be used with python's with statement
        """

        def __init__(self, _impl):
            self._impl = _impl

        def __enter__(self):
            self._impl._buffer(True)

        def __exit__(self, *args):
            self._impl._buffer(False)
            self._impl._flush()

    def Buffering(self):
        """Return a class intended to be used with python's with statement

        Examples
        --------
        .. code-block:: py

           with display.Buffering():
               display.dot("+", xc, yc)
        """
        return self._Buffering(self._impl)

    def flush(self):
        """Flush the buffers
        """
        self._impl._flush()

    def erase(self):
        """Erase the specified display frame
        """
        self._impl._erase()

    def dot(self, symb, c, r, size=2, ctype=None, origin=afwImage.PARENT, *args, **kwargs):
        """Draw a symbol onto the specified display frame

        Parameters
        ----------
        symb
            Possible values are:

                ``"+"``
                    Draw a +
                ``"x"``
                    Draw an x
                ``"*"``
                    Draw a *
                ``"o"``
                    Draw a circle
                ``"@:Mxx,Mxy,Myy"``
                    Draw an ellipse with moments (Mxx, Mxy, Myy) (argument size is ignored)
                `lsst.afw.geom.ellipses.BaseCore`
                    Draw the ellipse (argument size is ignored). N.b. objects
                    derived from `~lsst.afw.geom.ellipses.BaseCore` include
                    `~lsst.afw.geom.ellipses.Axes` and `~lsst.afw.geom.ellipses.Quadrupole`.
                Any other value
                    Interpreted as a string to be drawn.
        c, r
            The column and row where the symbol is drawn [0-based coordinates]
        size : `int`
            Size of symbol, in pixels
        ctype : `str`
            The desired color, either e.g. `lsst.afw.display.RED` or a color name known to X11
        origin : `lsst.afw.image.ImageOrigin`
            Coordinate system for the given positions.
        *args
            Extra arguments to backend
        **kwargs
            Extra keyword arguments to backend
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
                    symb = afwGeom.Quadrupole(mxx, myy, mxy)

            symb = afwGeom.ellipses.Axes(symb)

        self._impl._dot(symb, c, r, size, ctype, **kwargs)

    def line(self, points, origin=afwImage.PARENT, symbs=False, ctype=None, size=0.5):
        """Draw a set of symbols or connect points

        Parameters
        ----------
        points : `list`
            a list of (col, row)
        origin : `lsst.afw.image.ImageOrigin`
            Coordinate system for the given positions.
        symbs : `bool` or sequence
            If ``symbs`` is `True`, draw points at the specified points using the desired symbol,
            otherwise connect the dots.

            If ``symbs`` supports indexing (which includes a string -- caveat emptor) the
            elements are used to label the points
        ctype : `str`
            ``ctype`` is the name of a color (e.g. 'red')
        size : `float`
        """
        if symbs:
            try:
                symbs[1]
            except TypeError:
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
        """Set the range of the scaling from DN in the image to the image display

        Parameters
        ----------
        algorithm : `str`
            Desired scaling (e.g. "linear" or "asinh")
        min
            Minimum value, or "minmax" or "zscale"
        max
            Maximum value (must be `None` for minmax|zscale)
        unit
            Units for min and max (e.g. Percent, Absolute, Sigma; `None` if min==minmax|zscale)
        *args
            Optional arguments to the backend
        **kwargs
            Optional keyword arguments to the backend
        """
        if min in ("minmax", "zscale"):
            assert max is None, "You may not specify \"%s\" and max" % min
            assert unit is None, "You may not specify \"%s\" and unit" % min
        elif max is None:
            raise RuntimeError("Please specify max")

        self._impl._scale(algorithm, min, max, unit, *args, **kwargs)
    #
    # Zoom and Pan
    #

    def zoom(self, zoomfac=None, colc=None, rowc=None, origin=afwImage.PARENT):
        """Zoom frame by specified amount, optionally panning also
        """

        if (rowc and colc is None) or (colc and rowc is None):
            raise RuntimeError(
                "Please specify row and column center to pan about")

        if rowc is not None:
            if origin == afwImage.PARENT and self._xy0 is not None:
                x0, y0 = self._xy0
                colc -= x0
                rowc -= y0

            self._impl._pan(colc, rowc)

        if zoomfac is None and rowc is None:
            zoomfac = 2

        if zoomfac is not None:
            self._impl._zoom(zoomfac)

    def pan(self, colc=None, rowc=None, origin=afwImage.PARENT):
        """Pan to a location

        Parameters
        ----------
        colc, rowc
            the coordinates to pan to
        origin : `lsst.afw.image.ImageOrigin`
            Coordinate system for the given positions.

        See also
        --------
        Display.zoom
        """

        self.zoom(None, colc, rowc, origin)

    def interact(self):
        """Enter an interactive loop, listening for key presses in display and firing callbacks.

        Exit with ``q``, ``CR``, ``ESC``, or any other callback function that returns a `True` value.
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
                    logger.error(
                        "Display._callbacks['{0}']({0},{1},{2}) failed: {3}".format(k, x, y, e))

    def setCallback(self, k, func=None, noRaise=False):
        """Set the callback for a key

        Parameters
        ----------
        k
            The key to assign the callback to
        func : callable
            The callback assigned to ``k``
        noRaise : `bool`

        Returns
        -------
        oldFunc : callable
            The callback previously assigned to ``k``.
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
        """Return all callback keys

        Parameters
        ----------
        onlyActive : `bool`
            If `True` only return keys that do something
        """

        return sorted([k for k, func in self._callbacks.items() if
                       not (onlyActive and func == noop_callback)])

#
# Callbacks for display events
#


class Event:
    """A class to handle events such as key presses in image display windows
    """

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
    """Callback function

    Parameters
    ----------
    key
    x
    y
    """
    return False


def h_callback(k, x, y):
    print("Enter q or <ESC> to leave interactive mode, h for this help, or a letter to fire a callback")
    return False

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
    """Get the default frame for display
    """
    return Display.getDefaultFrame()


def incrDefaultFrame():
    """Increment the default frame for display
    """
    return Display.incrDefaultFrame()


def setDefaultMaskTransparency(maskPlaneTransparency={}):
    return Display.setDefaultMaskTransparency(maskPlaneTransparency)


def setDefaultMaskPlaneColor(name=None, color=None):
    """Set the default mapping from mask plane names to colors

    Parameters
    ----------
    name : `str` or `dict`
        name of mask plane, or a dict mapping names to colors.
        If ``name`` is `None`, use the hard-coded default dictionary
    color : `str`
        Desired color, or `None` if ``name`` is a dict
    """

    return Display.setDefaultMaskPlaneColor(name, color)


def getDisplay(frame=None, backend=None, create=True, verbose=False, *args, **kwargs):
    """Return a specific `Display`, creating it if need be

    Parameters
    ----------
    frame
        The desired frame (`None` => use defaultFrame (see `setDefaultFrame`))
    backend : `str`
        Create the specified frame using this backend (or the default if
        `None`) if it doesn't already exist. If ``backend == ""``, it's an
        error to specify a non-existent ``frame``.
    create : `bool`
        Create the display if it doesn't already exist.
    verbose : `bool`
        Allow backend to be chatty
    *args
        arguments passed to `Display` constructor
    **kwargs
        keyword arguments passed to `Display` constructor

    See also
    --------
    Display.getDisplay
    """

    return Display.getDisplay(frame, backend, create, verbose, *args, **kwargs)


def delAllDisplays():
    """Delete and close all known displays
    """
    return Display.delAllDisplays()
