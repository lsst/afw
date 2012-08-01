import os, stat, sys, threading, time

class Ds9CallbackThread(threading.Thread):
    def __init__(self, fileOfInterest, dt=0.2, verbose=False):
        self._verbose = verbose
        if self._verbose:
            print "Creating main thread"

        self.dt = dt
        self._fd = file(fileOfInterest, "w+")
        self.mtime = self.getMtime()

        threading.Thread.__init__(self)

    def getMtime(self):
        return os.fstat(self._fd.fileno())[stat.ST_MTIME]

    def run(self):
        while True:
            time.sleep(self.dt)

            if self.getMtime() != self.mtime: # file's changed
                self._fd.seek(0)
                data = self._fd.readline()
                try:
                    k, x, y = data.split()
                    x = float(x); y = float(y)
                except Exception, e:
                    print >> sys.stderr, "Error reading ds9 callback file: %s" % e

                self.mtime = self.getMtime()
                #
                # Do something
                #
                try:
                    callbacks[k](k, x, y)
                except Exception, e:
                    print >> sys.stderr, "ds9.callbacks[%s](%s, %s, %s) failed: %s" % \
                        (k, k, x, y, e)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# The file that ds9 writes when you hit an "active" key
#
callbackFile = os.path.join(os.environ.get("HOME", "."), ".ds9.xy")
#
# The file instructing ds9 to write callbackFile when you hit an active key
#
callbackAnalysisFile = os.path.join(os.environ.get("HOME", "."), ".ds9.xy.ans")
#
# Default fallback function
#
def noop(k, x, y):
    """Callback function: arguments key, x, y"""
    pass

# These are the letters that will become active.
#
# This is a little magic...  these keys are processed when ds9.py is first sourced, and
# an "analysis" file is sent to ds9 that specifies that each active key should write to
# the callbackFile (usually ~/.ds9.xy).  When this file is modified, a callback (as
# set by setCallback) is dispatched.  So activeKeys need to be set in two places:
#   In the analysis file sent to ds9
#   In the callback dict
# This is done at the bottom of ds9.py when it is first imported
#
# The simplest way to add a new active key is to set activeKeys before importing ds9, in
# which case you need not understand this comment.
#
activeKeys =  "uvwxyz"

def setCallback(k, func=noop):
    """Set the callback for key k to be func (k must be in the list of activeKeys)

To change this set, set activeKeys before importing lsst.afw.display.ds9
(or change it, and then arrange for your key to be active in ds9 --- see callbackAnalysisFile)
    """

    if k in activeKeys:
        callbacks[k] = func
    else:
        raise RuntimeError("You may only assign callbacks to letters in the set %s" %
                           ", ".join([s for s in activeKeys]))
