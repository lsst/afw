##
## \file
## \brief Definitions to talk to ds9 from python

import os, re, math, sys, time

try:
    import xpa
except ImportError, e:
    print >> sys.stderr, "Cannot import xpa: %s" % e

import displayLib
import lsst.afw.math as afwMath

needShow = True;                        # Used to avoid a bug in ds9 5.4

## An error talking to ds9
class Ds9Error(IOError):
    """Some problem talking to ds9"""

try:
    type(_defaultFrame)
except NameError:
    def setDefaultFrame(frame=0):
        """Set the default frame for ds9"""
        global _defaultFrame
        _defaultFrame = frame

    def getDefaultFrame():
        """Get the default frame for ds9"""
        return _defaultFrame

    def incrDefaultFrame():
        """Increment the default frame for ds9"""
        global _defaultFrame
        _defaultFrame += 1
        return _defaultFrame

    setDefaultFrame(0)
#
# Symbolic names for mask/line colours.  N.b. ds9 5.3+ supports any X11 colour for masks
#
WHITE = "white"
BLACK = "black"
RED = "red"
GREEN = "green"
BLUE = "blue"
CYAN = "cyan"
MAGENTA = "magenta"
YELLOW = "yellow"
_maskColors = [WHITE, BLACK, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW]

def setMaskPlaneColor(name, color=None):
    """Request that mask plane name be displayed as color; name may be a dictionary
    (in which case color should be omitted"""

    if isinstance(name, dict):
        assert color == None
        for k in name.keys():
            setMaskPlaneColor(k, name[k])
        return

    global _maskPlaneColors
    try:
        type(_maskPlaneColors)
    except:
        _maskPlaneColors = {}

    _maskPlaneColors[name] = color

#
# Default mapping from mask plane names to colours
#
setMaskPlaneColor({
    "BAD": RED,
    "CR" : MAGENTA,
    "EDGE": YELLOW,
    "INTERPOLATED" : GREEN,
    "SATURATED" : GREEN,
    "DETECTED" : BLUE,
    "DETECTED_NEGATIVE" : CYAN,
    # deprecated names
    "INTRP" : GREEN,
    "SAT" : GREEN,
    })

def getMaskPlaneColor(name):
    """Return the colour associated with the specified mask plane name"""

    if _maskPlaneColors.has_key(name):
        return _maskPlaneColors[name]
    else:
        return None

def setMaskPlaneVisibility(name, show=True):
    """Specify the visibility of a given mask plane;
    name may be a dictionary (in which case show will be ignored)"""

    global _maskPlaneVisibility
    try:
        type(_maskPlaneVisibility)
    except NameError, e:
        _maskPlaneVisibility = {}

    if isinstance(name, dict):
        for k in name.keys():
            setMaskPlaneVisibility(k, name[k])
        return

    _maskPlaneVisibility[name] = show

setMaskPlaneVisibility({})

def getMaskPlaneVisibility(name):
    """Should we display the specified mask plane name?"""

    if _maskPlaneVisibility.has_key(name):
        return _maskPlaneVisibility[name]
    else:
        return True

def setMaskTransparency(transparency=None):
    """Specify ds9's mask transparency (percent); or None to not set it when loading masks"""

    global _maskTransparency
    if transparency is not None and (transparency < 0 or transparency > 100):
        print >> sys.stderr, "Mask transparency should be in the range [0, 100]; clipping"
        if transparency < 0:
            transparency = 0
        else:
            transparency = 100

    _maskTransparency = transparency

setMaskTransparency()

def getMaskTransparency():
    """Return ds9's mask transparency"""

    return _maskTransparency

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
    return xpa.get(None, getXpaAccessPoint(), "about", "").splitlines()[1].split()[1]

def ds9Cmd(cmd, trap=True):
    """Issue a ds9 command, raising errors as appropriate"""
   
    if getDefaultFrame() is None:
        return

    try:
        xpa.set(None, getXpaAccessPoint(), cmd, "", "", 0)
    except IOError, e:
        if not trap:
            raise Ds9Error, "XPA: %s, (%s)" % (e, cmd)
        else:
            print >> sys.stderr, "Caught ds9 exception processing command \"%s\": %s" % (cmd, e)

def initDS9(execDs9=True):
    try:
        xpa.reset()
        ds9Cmd("iconify no; raise", False)
        ds9Cmd("wcs wcsa", False)         # include the pixel coordinates WCS (WCSA)
        
        v0, v1 = [int(v) for v in ds9Version().split('.')][0:2]
        global needShow
        needShow = (v0 == 5 and v1 <= 4)
    except Ds9Error, e:
        if execDs9:
            print "ds9 doesn't appear to be running (%s), I'll exec it for you" % e
        if not re.search('xpa', os.environ['PATH']):
            raise Ds9Error, 'You need the xpa binaries in your path to use ds9 with python'

        os.system('ds9 &')
        for i in range(10):
            try:
                ds9Cmd("frame 0", False)
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

def show(frame=None):
    """Uniconify and Raise ds9.  N.b. throws an exception if frame doesn't exit"""
    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return
        
    ds9Cmd("frame %d; raise" % frame, trap=False)

def setMaskColor(color=GREEN):
    """Set the ds9 mask colour to; eg. ds9.setMaskColor(ds9.RED)"""
    ds9Cmd("mask color %s" % color)


def mtv(data, frame=None, init=True, wcs=None, isMask=False, lowOrderBits=False, title=None, settings=None):
    """Display an Image or Mask on a DS9 display

    If lowOrderBits is True, give low-order-bits priority in display (i.e.
    overlay them last)

    Historical note: the name "mtv" comes from Jim Gunn's forth imageprocessing
    system, Mirella (named after Mirella Freni); The "m" stands for Mirella.
    """

    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return
   
    if init:
        for i in range(3):
            try:
                initDS9(i == 0)
            except Ds9Error:
                print "waiting for ds9...\r",
                sys.stdout.flush()
                time.sleep(0.5)
            else:
                print "                                     \r",
                sys.stdout.flush()
                break
         
    ds9Cmd("frame %d" % frame)

    if settings:
        for setting in settings:
            ds9Cmd("%s %s" % (setting, settings[setting]))

    if re.search("::DecoratedImage<", data.__repr__()): # it's a DecorateImage; display it
        _mtv(data.getImage(), wcs, title, False)
    elif re.search("::MaskedImage<", data.__repr__()): # it's a MaskedImage; display Image and overlay Mask
        _mtv(data.getImage(), wcs, title, False)
        mask = data.getMask(True)
        if mask:
            mtv(mask, frame, False, wcs, False, lowOrderBits=lowOrderBits, title=title, settings=settings)
            if getMaskTransparency() is not None:
                ds9Cmd("mask transparency %d" % getMaskTransparency())

    elif re.search("::Exposure<", data.__repr__()): # it's an Exposure; display the MaskedImage with the WCS
        if wcs:
            raise RuntimeError, "You may not specify a wcs with an Exposure"

        mtv(data.getMaskedImage(), frame, False, data.getWcs(),
            False, lowOrderBits=lowOrderBits, title=title, settings=settings)

    elif re.search("::Mask<", data.__repr__()): # it's a Mask; display it, bitplane by bitplane
        nMaskPlanes = data.getNumPlanesUsed()
        maskPlanes = data.getMaskPlaneDict()

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
       
        for p in planeList:
            if planes[p] or True:
                if not getMaskPlaneVisibility(planes[p]):
                    continue

                if not ((1 << p) & usedPlanes): # no pixels have this bitplane set
                    continue

                mask <<= data
                mask &= (1 << p)

                color = getMaskPlaneColor(planes[p])

                if not color:            # none was specified
                    while True:
                        color = _maskColors[colorIndex % len(_maskColors)]
                        colorIndex += 1
                        if color != WHITE and color != BLACK:
                            break

                setMaskColor(color)
                _mtv(mask, wcs, title, True)
        return
    elif re.search("::Image<", data.__repr__()): # it's an Image; display it
        _mtv(data, wcs, title, False)
    else:
        raise RuntimeError, "Unsupported type %s" % data.__repr__()

def _mtv(data, wcs, title, isMask):
    """Internal routine to display an Image or Mask on a DS9 display"""

    if True:
        if isMask:
            xpa_cmd = "xpaset %s fits mask" % getXpaAccessPoint()
            if re.search(r"unsigned short|boost::uint16_t", data.__str__()):
                data |= 0x8000  # Hack. ds9 mis-handles BZERO/BSCALE in masks. This is a copy we're modifying
        else:
            xpa_cmd = "xpaset %s fits" % getXpaAccessPoint()
           
        pfd = os.popen(xpa_cmd, "w")
    else:
        pfd = file("foo.fits", "w")

    try:
        #import pdb; pdb.set_trace()
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
#
# Graphics commands
#
def erase(frame=None):
    """Erase the specified DS9 frame"""
    if frame is None:
        frame = getDefaultFrame()
        
    if frame is None:
        return

    ds9Cmd("frame %d; regions delete all" % frame)

def dot(symb, c, r, frame=None, size=2, ctype=None):
    """Draw a symbol onto the specified DS9 frame at (col,row) = (c,r) [0-based coordinates]
Possible values are:
        +                Draw a +
        x                Draw an x
        o                Draw a circle
        @:Mxx,Mxy,Myy    Draw an ellipse with moments (Mxx, Mxy, Myy) (size is ignored)
Any other value is interpreted as a string to be drawn
"""
    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return

    if isinstance(symb, int):
        symb = "%d" % (symb)

    if ctype == None:
        color = ""                       # the default
    else:
        color = ' # color=%s' % ctype

    cmd = "frame %d; " % frame
    r += 1
    c += 1                      # ds9 uses 1-based coordinates
    if symb == '+':
        cmd += 'regions command {line %g %g %g %g%s}; ' % (c, r+size, c, r-size, color)
        cmd += 'regions command {line %g %g %g %g%s}; ' % (c-size, r, c+size, r, color)
    elif symb == 'x':
        size = size/math.sqrt(2)
        cmd += 'regions command {line %g %g %g %g%s}; ' % (c+size, r+size, c-size, r-size, color)
        cmd += 'regions command {line %g %g %g %g%s}; ' % (c-size, r+size, c+size, r-size, color)
    elif symb == 'o':
        cmd += 'regions command {circle %g %g %g%s}; ' % (c, r, size, color)
    elif re.search(r"^@:", symb):
        mat = re.search(r"^@:([^,]+),([^,]+),([^,]+)", symb)
        mxx, mxy, myy = map(lambda x: float(x), mat.groups())

        theta = (0.5*math.atan2(2*mxy, mxx - myy))
        ct, st = math.cos(theta), math.sin(theta)
        theta *= 180/math.pi
        A = math.sqrt(mxx*ct*ct + mxy*2*ct*st + myy*st*st)
        B = math.sqrt(mxx*st*st - mxy*2*ct*st + myy*ct*ct)
        if A < B:
            A, B = B, A
            theta += 90
       
        cmd += 'regions command {ellipse %g %g %g %g %g%s}; ' % (c, r, A, B, theta, color)
    else:
        try:
            # We have to check for the frame's existance with show() as the text command crashed ds9 5.4
            # if it doesn't
            if needShow:
                show(frame)
            cmd += 'regions command {text %g %g \"%s\"%s}' % (c, r, symb, color)
        except Exception, e:
            print >> sys.stderr, "Ds9 frame %d doesn't exist" % frame, e

    ds9Cmd(cmd)

def line(points, frame=None, symbs=False, ctype=None):
    """Draw a set of symbols or connect the points, a list of (col,row)
If symbs is True, draw points at the specified points using the desired symbol,
otherwise connect the dots.  Ctype is the name of a colour (e.g. 'red')"""
   
    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return

    if symbs:
        for (c, r) in points:
            dot(symbs, r, c, frame = frame, size = 0.5, ctype=ctype)
    else:
        if ctype == None:                # default
            color = ""
        else:
            color = "# color=%s" % ctype

        if len(points) > 0:
            cmd = "frame %d; " % (frame)

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
# Zoom and Pan
#
def zoom(zoomfac=None, colc=None, rowc=None, frame=None):
    """Zoom frame by specified amount, optionally panning also"""
    
    if frame < 0:
        frame = getDefaultFrame()
        
    if frame is None:
        return

    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return

    if (rowc and colc is None) or (colc and rowc is None):
        raise Ds9Error, "Please specify row and column center to pan about"
   
    if zoomfac == None and rowc == None:
        zoomfac = 2

    cmd = "frame %d; " % frame
    if zoomfac != None:
        cmd += "zoom to %d; " % zoomfac

    if rowc != None:
        cmd += "pan to %g %g physical; " % (colc + 1, rowc + 1) # ds9 is 1-indexed. Grrr

    ds9Cmd(cmd)

def pan(colc=None, rowc=None, frame=None):
    """Pan to (rowc, colc); see also zoom"""

    if frame is None:
        frame = getDefaultFrame()

    if frame is None:
        return

    zoom(None, colc, rowc, frame)
