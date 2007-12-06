#
# XPA
#
import os, re, math, sys, time

try: import xpa
except: print "Cannot import xpa"

try: import fwDisplay
except: pass

class Ds9Error(IOError):
    """Some problem talking to ds9"""

#
# Symbolic names for mask colours
#
WHITE = "white"; BLACK = "black"
RED = "red"; GREEN = "green"; BLUE = "blue"
CYAN = "cyan"; MAGENTA = "magenta"; YELLOW = "yellow"

def ds9Cmd(cmd):
   """Issue a ds9 command, raising errors as appropriate"""
   
   try:
      xpa.set(None, "ds9", cmd, "", "", 0)
   except IOError:
      raise Ds9Error, "XPA: (%s)" % cmd

def initDS9(execDs9 = True):
   try:
      ds9Cmd("iconify no; raise")
   except IOError:
      if execDs9:
         print "ds9 doesn't appear to be running, I'll exec it for you"
         if not re.search('xpa', os.environ['PATH']):
            raise Ds9Error, 'You need the xpa binaries in your path to use ds9 with python'

         os.system('ds9 &')
         for i in range(0,10):
            try:
               ds9Cmd("frame 0; scale histequ; scale mode minmax")
               break
            except IOError:
               print "waiting for ds9...\r",
               time.sleep(0.5)
            else:
               break

         print "                  \r",
         sys.stdout.flush();

      raise Ds9Error

def setMaskColor(color = GREEN):
    """Set the ds9 mask colour to; eg. ds9.setMaskColor(ds9.RED)"""
    xpa.set(None, "ds9", "mask color %s" % color, "", "", 0)

def mtv(data, frame=0, init=1, WCS=None, isMask=False):
   """Display an Image or Mask on a DS9 display"""
	
   if frame == None:
      return
   
   if init:
      for i in range(0,3):
         try:
            initDS9(i == 0)
         except IOError:
            print "waiting for ds9...\r", ; sys.stdout.flush();
            time.sleep(0.5)
         else:
            break
         
   ds9Cmd("frame %d" % frame)

   if re.search("MaskedImage", data.repr()): # it's a MaskedImage
       mtv(data.getImage(), frame, init, WCS, False)
       setMaskColor(RED)
       mtv(data.getMask(), frame, init, WCS, True)
       return

   if True:
       if isMask:
           xpa_cmd = "xpaset ds9 fits mask"
       else:
           xpa_cmd = "xpaset ds9 fits"
           
       pfd = os.popen(xpa_cmd, "w")
   else:
      pfd = file("foo.fits", "w")

   try:
       #import pdb; pdb.set_trace()
       try:
           fwDisplay.writeFitsImage(pfd.fileno(), data, WCS)
       except NotImplementedError:
           fwDisplay.writeFitsImage(pfd.fileno(), data.get(), WCS)
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
def erase(frame = 0, len = 2):
   """Erase the specified DS9 frame"""
   if frame == None:
      return

   ds9Cmd("frame %d; regions delete all" % frame)

def dot(symb, r, c, frame = 0, size = 2, ctype = 'green'):
   """Draw a symbol onto the specfied DS9 frame at (row,col) = (r,c) [0-based coordinates]
Possible values are:
	+	Draw a +
	x	Draw an x
        o	Draw a circle
Any other value is interpreted as a string to be drawn
"""
   if frame == None:
      return

   cmd = "frame %d; regions physical; " % frame
   r += 1; c += 1;                      # ds9 uses 1-based coordinates
   if (symb == '+'):
      cmd += 'regions line %g %g %g %g # color=%s; ' % (c, r+size, c, r-size, ctype)
      cmd += 'regions line %g %g %g %g ' % (c-size, r, c+size, r)
   elif (symb == 'x'):
      size = size/math.sqrt(2)
      cmd += 'regions line %g %g %g %g # color=%s; ; ' % (c+size, r+size, c-size, r-size, ctype)
      cmd += 'regions line %g %g %g %g ' % (c-size, r+size, c+size, r-size)
   elif (symb == 'o'):
      cmd += 'regions circle %g %g %g ' % (c, r, size)
   else:
      cmd += 'regions text %g %g \"%s\"' % (c, r, symb)

   cmd += ' # color=%s' % ctype

   ds9Cmd(cmd)

def line(points, frame = 0, symbs = False, ctype = 'green'):
   """Draw a set of symbols or connect the points, a list of (row,col)
If symbs is True, draw points at the specified points using the desired symbol,
otherwise connect the dots.  Ctype is the name of a colour (e.g. 'red')"""
   
   if frame == None:
      return

   if symbs:
      for (r, c) in points:
         dot(symbs, r, c, frame = frame, size = 0.5, ctype = ctype)
   else:
      cmd = "frame %d; regions image; regions line " % (frame)

      for (r, c) in points:
         r += 1; c += 1;                   # ds9 uses 1-based coordinates
         cmd += '%g %g ' % (c, r)
         
      cmd += ' # color=%s' % ctype
         
      ds9Cmd(cmd)
#
# Zoom and Pan
#
def zoom(zoomfac = None, rowc = None, colc = None, frame = 0):
   """Zoom frame by specified amount, optionally panning also"""

   if frame == None:
      return

   if (rowc and not colc) or (not rowc and colc):
      raise Ds9Error, "Please specify row and column center to pan about"
   
   if zoomfac == None and rowc == None:
      zoomfac = 2

   cmd = ""
   if zoomfac != None:
      cmd += "zoom to %d; " % zoomfac

   if rowc != None:
      cmd += "pan to %g %g physical; " % (colc + 1, rowc + 1) # ds9 is 1-indexed. Grrr

   ds9Cmd(cmd)

def pan(rowc = None, colc = None, frame = 0):
   """Pan to (rowc, colc); see also zoom"""
   zoom(None, rowc, colc, frame)
