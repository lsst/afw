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
## \brief Definitions to talk to ds9 from python

import os, re, math, sys, time

import interface
import virtualDevice

try:
    import xpa
except ImportError, e:
    print >> sys.stderr, "Cannot import xpa: %s" % e

import displayLib
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

try:
    needShow
except NameError:
    needShow = True;                        # Used to avoid a bug in ds9 5.4

## An error talking to ds9
class Ds9Error(IOError):
    """Some problem talking to ds9"""

#
# Backwards compatibility.  Downstream code should be converted to use display.RED etc.
#
from . import BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, WHITE

try:
    _maskTransparency
except NameError:
    _maskTransparency = None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def getXpaAccessPoint():
    """Parse XPA_PORT and send return an identifier to send ds9 commands there, instead of "ds9"
    If you don't have XPA_PORT set, the usual xpans tricks will be played when we return "ds9".
    """
    xpa_port = os.environ.get("XPA_PORT")
    if xpa_port:
        mat = re.search(r"^DS9:ds9\s+(\d+)\s+(\d+)", xpa_port)
        if mat:
            port1, port2 = mat.groups()

            return "127.0.0.1:%s" % (port1)
        else:
            print >> sys.stderr, "Failed to parse XPA_PORT=%s" % xpa_port

    return "ds9"

def ds9Version():
    """Return the version of ds9 in use, as a string"""
    try:
        v = ds9Cmd("about", get=True)
        return v.splitlines()[1].split()[1]
    except Exception, e:
        print >> sys.stderr, "Error reading version: %s" % e
        return "0.0.0"

try:
    cmdBuffer
except NameError:
    XPA_SZ_LINE = 4096 - 100            # internal buffersize in xpa. Sigh; esp. as the 100 is some needed slop

    class Buffer(object):
        """Control buffering the sending of commands to ds9;
        annoying but necessary for anything resembling performance

        The usual usage pattern (from a module importing this file, ds9.py) is:

            with ds9.Buffering():
                # bunches of ds9.{dot,line} commands
                ds9.flush()
                # bunches more ds9.{dot,line} commands
        """

        def __init__(self, size=0):
            """Create a command buffer, with a maximum depth of size"""
            self._commands = ""         # list of pending commands
            self._lenCommands = len(self._commands)
            self._bufsize = []          # stack of bufsizes

            self._bufsize.append(size)  # don't call self.size() as ds9Cmd isn't defined yet

        def set(self, size, silent=True):
            """Set the ds9 buffer size to size"""
            if size < 0:
                size = XPA_SZ_LINE - 5

            if size > XPA_SZ_LINE:
                print >> sys.stderr, \
                      "xpa silently hardcodes a limit of %d for buffer sizes (you asked for %d) " % \
                      (XPA_SZ_LINE, size)
                self.set(-1)            # use max buffersize
                return

            if self._bufsize:
                self._bufsize[-1] = size # change current value
            else:
                self._bufsize.append(size) # there is no current value; set one

            self.flush(silent=silent)

        def _getSize(self):
            """Get the current ds9 buffer size"""
            return self._bufsize[-1]

        def pushSize(self, size=-1):
            """Replace current ds9 command buffer size with size (see also popSize)
            @param:  Size of buffer (-1: largest possible given bugs in xpa)"""
            self.flush(silent=True)
            self._bufsize.append(0)
            self.set(size, silent=True)

        def popSize(self):
            """Switch back to the previous command buffer size (see also pushSize)"""
            self.flush(silent=True)

            if len(self._bufsize) > 1:
                self._bufsize.pop()

        def flush(self, silent=True):
            """Flush the pending commands"""
            ds9Cmd(flush=True, silent=silent)

    cmdBuffer = Buffer(0)

def selectFrame(frame):
    return "frame %d" % (frame)

def ds9Cmd(cmd=None, trap=True, flush=False, silent=True, frame=None, get=False):
    """Issue a ds9 command, raising errors as appropriate"""

    global cmdBuffer
    if cmd:
        if frame is not None:
            cmd = "%s;" % selectFrame(frame) + cmd

        if get:
            return xpa.get(None, getXpaAccessPoint(), cmd, "").strip()

        # Work around xpa's habit of silently truncating long lines
        if cmdBuffer._lenCommands + len(cmd) > XPA_SZ_LINE - 5: # 5 to handle newlines and such like
            ds9Cmd(flush=True, silent=silent)

        cmdBuffer._commands += ";" + cmd
        cmdBuffer._lenCommands += 1 + len(cmd)

    if flush or cmdBuffer._lenCommands >= cmdBuffer._getSize():
        cmd = cmdBuffer._commands + "\n"
        cmdBuffer._commands = ""
        cmdBuffer._lenCommands = 0
    else:
        return

    cmd = cmd.rstrip()
    if not cmd:
        return

    try:
        ret = xpa.set(None, getXpaAccessPoint(), cmd, "", "", 0)
        if ret:
            raise IOError(ret)
    except IOError, e:
        if not trap:
            raise Ds9Error, "XPA: %s, (%s)" % (e, cmd)
        elif not silent:
            print >> sys.stderr, "Caught ds9 exception processing command \"%s\": %s" % (cmd, e)

def initDS9(execDs9=True):
    try:
        xpa.reset()
        ds9Cmd("iconify no; raise", False)
        ds9Cmd("wcs wcsa", False)         # include the pixel coordinates WCS (WCSA)

        v0, v1 = ds9Version().split('.')[0:2]
        global needShow
        needShow = False
        try:
            if int(v0) == 5:
                needShow = (int(v1) <= 4)
        except:
            pass
    except Ds9Error, e:
        if execDs9:
            print "ds9 doesn't appear to be running (%s), I'll exec it for you" % e
        if not re.search('xpa', os.environ['PATH']):
            raise Ds9Error, 'You need the xpa binaries in your path to use ds9 with python'

        os.system('ds9 &')
        for i in range(10):
            try:
                ds9Cmd(selectFrame(1), False)
                break
            except Ds9Error:
                print "waiting for ds9...\r",
                sys.stdout.flush()
                time.sleep(0.5)
            else:
                print "                  \r",
                break

        sys.stdout.flush()

        raise Ds9Error

class Ds9Event(interface.Event):
    """An event generated by a mouse or key click on the display"""
    def __init__(self, k, x, y):
        interface.Event.__init__(self, k, x, y)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DisplayImpl(virtualDevice.DisplayImpl):
    def __init__(self, display, verbose=False, *args, **kwargs):
        virtualDevice.DisplayImpl.__init__(self, display, verbose)

    def _setMaskTransparency(self, transparency, maskplane):
        """Specify ds9's mask transparency (percent); or None to not set it when loading masks"""
        if maskplane != None:
            print >> sys.stderr, ("ds9 is unable to set transparency for individual maskplanes" % maskplane)
            return
        ds9Cmd("mask transparency %d" % transparency, frame=self.display.frame)

    def _getMaskTransparency(self, maskplane):
        """Return the current ds9's mask transparency"""

        selectFrame(self.display.frame)
        return float(ds9Cmd("mask transparency", get=True))

    def _show(self):
        """Uniconify and Raise ds9.  N.b. throws an exception if frame doesn't exit"""
        ds9Cmd("raise", trap=False, frame=self.display.frame)

    def _mtv(self, data, title="", wcs=None, init=True, isMask=False, lowOrderBits=False, settings=None):
        """Display an Image or Mask on a DS9 display

        If lowOrderBits is True, give low-order-bits priority in display (i.e.
        overlay them last)

        Historical note: the name "mtv" comes from Jim Gunn's forth imageprocessing
        system, Mirella (named after Mirella Freni); The "m" stands for Mirella.
        """

        if init:
            for i in range(3):
                try:
                    initDS9(i == 0)
                except Ds9Error:
                    print "waiting for ds9...\r",
                    sys.stdout.flush()
                    time.sleep(0.5)
                else:
                    if i > 0:
                        print "                                     \r",
                        sys.stdout.flush()
                    break

        ds9Cmd(selectFrame(self.display.frame))
        ds9Cmd("smooth no")
        self._erase()

        if settings:
            for setting in settings:
                ds9Cmd("%s %s" % (setting, settings[setting]))

        if re.search("::DecoratedImage<", repr(data)): # it's a DecorateImage; display it
            _i_mtv(data.getImage(), wcs, title, False)
        elif re.search("::MaskedImage<", repr(data)): # it's a MaskedImage; display Image and overlay Mask
            _i_mtv(data.getImage(), wcs, title, False)
            mask = data.getMask(True)
            if mask:
                self._mtv(mask, title, wcs, False, True, lowOrderBits=lowOrderBits, settings=settings)
                if self.display.getMaskTransparency() is not None:
                    ds9Cmd("mask transparency %d" % self.display.getMaskTransparency())

        elif re.search("::Exposure<", repr(data)): # it's an Exposure; display the MaskedImage with the WCS
            if wcs:
                raise RuntimeError, "You may not specify a wcs with an Exposure"

            self._mtv(data.getMaskedImage(), title, data.getWcs(),
                      False, False, lowOrderBits=lowOrderBits, settings=settings)

        elif re.search("::Mask<", repr(data)): # it's a Mask; display it, bitplane by bitplane
            maskPlanes = data.getMaskPlaneDict()
            nMaskPlanes = max(maskPlanes.values()) + 1

            planes = {}                      # build inverse dictionary
            for key in maskPlanes.keys():
                planes[maskPlanes[key]] = key

            colorIndex = 0                   # index into maskColors

            if lowOrderBits:
                planeList = range(nMaskPlanes - 1, -1, -1)
            else:
                planeList = range(nMaskPlanes)

            usedPlanes = long(afwMath.makeStatistics(data, afwMath.SUM).getValue())
            mask = data.Factory(data.getDimensions())
            #
            # ds9 can't display a Mask without an image; so display an Image first
            #
            if not isMask:
                im = afwImage.ImageU(data.getDimensions())
                self._mtv(im)

            colorGenerator = self.display.maskColorGenerator(omitBW=True)
            for p in planeList:
                if planes.get(p):
                    pname = planes[p]

                if not ((1 << p) & usedPlanes): # no pixels have this bitplane set
                    continue

                mask <<= data
                mask &= (1 << p)

                color = self.display.getMaskPlaneColor(pname)

                if not color:            # none was specified
                    color = next(colorGenerator)
                elif color.lower() == "ignore":
                    continue

                ds9Cmd("mask color %s" % color)
                _i_mtv(mask, wcs, title, True)
            return
        elif re.search("::Image<", repr(data)): # it's an Image; display it
            _i_mtv(data, wcs, title, False)
        else:
            raise RuntimeError, "Unsupported type %s" % repr(data)

    #
    # Graphics commands
    #
    def _buffer(self, enable=True):
        if enable:
            cmdBuffer.pushSize()
        else:
            cmdBuffer.popSize()

    def _flush(self):
        cmdBuffer.flush()

    def _erase(self):
        """Erase the specified DS9 frame"""
        ds9Cmd("regions delete all", flush=True, frame=self.display.frame)

    def _dot(self, symb, c, r, size, ctype, fontFamily="helvetica", textAngle=None):
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

        cmd = selectFrame(self.display.frame) + "; "
        r += 1
        c += 1                      # ds9 uses 1-based coordinates
        if isinstance(symb, afwGeom.ellipses.Axes):
            cmd += 'regions command {ellipse %g %g %g %g %g%s}; ' % (c, r, symb.getA(), symb.getB(),
                                                                     math.degrees(symb.getTheta()), color)
        elif symb == '+':
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c, r+size, c, r-size, color)
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c-size, r, c+size, r, color)
        elif symb == 'x':
            size = size/math.sqrt(2)
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c+size, r+size, c-size, r-size, color)
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c-size, r+size, c+size, r-size, color)
        elif symb == '*':
            size30 = 0.5*size
            size60 = 0.5*math.sqrt(3)*size
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c+size, r, c-size, r, color)
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c-size30, r+size60, c+size30, r-size60, color)
            cmd += 'regions command {line %g %g %g %g%s}; ' % (c+size30, r+size60, c-size30, r-size60, color)
        elif symb == 'o':
            cmd += 'regions command {circle %g %g %g%s}; ' % (c, r, size, color)
        else:
            try:
                # We have to check for the frame's existance with show() as the text command crashed ds9 5.4
                # if it doesn't
                if needShow:
                    self._show()

                color = re.sub("^ # ", "", color) # skip the leading " # "

                angle = ""
                if textAngle is not None:
                    angle += " textangle=%.1f"%(textAngle) 

                font = ""
                if size != 2 or fontFamily != "helvetica":
                    fontFamily = fontFamily.split()
                    font += ' font="%s %d' % (fontFamily.pop(0), int(10*size/2.0 + 0.5))
                    if fontFamily:
                        font += " %s" % " ".join(fontFamily)
                    font += '"'
                extra = ""
                if color or angle or font:
                    extra = " # "
                    extra += color
                    extra += angle
                    extra += font

                cmd += 'regions command {text %g %g \"%s\"%s };' % (c, r, symb, extra)
            except Exception, e:
                print >> sys.stderr, ("Ds9 frame %d doesn't exist" % self.display.frame), e

        ds9Cmd(cmd, silent=True)

    def _drawLines(self, points, ctype):
        """Connect the points, a list of (col,row)
        Ctype is the name of a colour (e.g. 'red')"""

        if ctype == None:                # default
            color = ""
        else:
            color = "# color=%s" % ctype

        if len(points) > 0:
            cmd = selectFrame(self.display.frame) + "; "

            c0, r0 = points[0]
            r0 += 1
            c0 += 1             # ds9 uses 1-based coordinates
            for (c, r) in points[1:]:
                r += 1
                c += 1            # ds9 uses 1-based coordinates
                cmd += 'regions command { line %g %g %g %g %s};' % (c0, r0, c, r, color)
                c0, r0 = c, r

            ds9Cmd(cmd)
    #
    # Set gray scale
    #
    def _setScaleType(self, type):
        ds9Cmd("scale %s" % type, frame=self.display.frame)

    def _setScaleLimits(self, min, max):
        ds9Cmd("scale limits %g %g" % (min, max), frame=self.display.frame)

    #
    # Zoom and Pan
    #
    def _zoom(self, zoomfac):
        """Zoom frame by specified amount"""

        cmd = selectFrame(self.display.frame) + "; "
        cmd += "zoom to %d; " % zoomfac

        ds9Cmd(cmd, flush=True)

    def _pan(self, colc, rowc):
        """Pan frame to (colc, rowc)"""

        cmd = selectFrame(self.display.frame) + "; "
        cmd += "pan to %g %g physical; " % (colc + 1, rowc + 1) # ds9 is 1-indexed. Grrr

        ds9Cmd(cmd, flush=True)

    def _getEvent(self):
        """Listen for a key press on frame in ds9, returning (key, x, y)"""

        vals = ds9Cmd("imexam key coordinate", get=True).split()
        if vals[0] == "XPA$ERROR":
            if vals[1:4] == ['unknown', 'option', '"-state"']:
                pass                    # a ds9 bug --- you get this by hitting TAB
            else:
                print >> sys.stderr, "Error return from imexam:", " ".join(vals)
            return None

        k = vals.pop(0)
        try:
            x = float(vals[0]); y = float(vals[1])
        except:
            x = float("NaN"); y = float("NaN")

        return Ds9Event(k, x, y)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

try:
    haveGzip
except NameError:
    haveGzip = not os.system("gzip < /dev/null > /dev/null 2>&1") # does gzip work?

def _i_mtv(data, wcs, title, isMask):
    """Internal routine to display an Image or Mask on a DS9 display"""

    title = str(title) if title else ""

    if True:
        if isMask:
            xpa_cmd = "xpaset %s fits mask" % getXpaAccessPoint()
            if re.search(r"unsigned short|boost::uint16_t", data.__str__()):
                data |= 0x8000  # Hack. ds9 mis-handles BZERO/BSCALE in masks. This is a copy we're modifying
        else:
            xpa_cmd = "xpaset %s fits" % getXpaAccessPoint()

        if haveGzip:
            xpa_cmd = "gzip | " + xpa_cmd

        pfd = os.popen(xpa_cmd, "w")
    else:
        pfd = file("foo.fits", "w")

    try:
        displayLib.writeFitsImage(pfd.fileno(), data, wcs, title)
    except Exception, e:
        try:
            pfd.close()
        except:
            pass

        raise e

    try:
        pfd.close()
    except:
        pass

if False:
    try:
        definedCallbacks
    except NameError:
        definedCallbacks = True

        for k in ('XPA$ERROR',):
            interface.setCallback(k)
