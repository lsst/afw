"""Support code for running unit tests"""

import unittest
import lsst.fw.Core.fwLib as fw
import os
import sys

try:
    type(memId0)
except NameError:
    memId0 = 0

def init():
    global memId0
    memId0 = fw.Citizen_getNextMemId()  # used by MemoryTestCase

def run(suite):
    """Exit with the status code resulting from running the provided test suite"""
    status = 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1
    sys.exit(status)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
class MemoryTestCase(unittest.TestCase):
    """Check for memory leaks since memId0 was allocated"""
    def setUp(self):
        pass

    def testLeaks(self):
        """Check for memory leaks in the preceding tests"""

        global memId0
        if fw.Citizen_census(0, memId0) != 0:
            if not False:
                print fw.Citizen_census(0, memId0), "Objects leaked:"
                print fw.Citizen_census(fw.cout, memId0)
                
            self.fail("Leaked %d blocks" % fw.Citizen_census(0, memId0))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def findFileFromRoot(ifile):
    """Find file which is specified as a path relative to the toplevel directory;
    we start in $cwd and walk up until we find the file (or throw IOError if it doesn't exist)

    This is useful for running tests that may be run from fw/tests or fw"""
    
    if os.path.isfile(ifile):
        return ifile

    ofile = None
    file = ifile
    while file != "":
        dirname, basename = os.path.split(file)
        if ofile:
            ofile = os.path.join(basename, ofile)
        else:
            ofile = basename

        if os.path.isfile(ofile):
            return ofile

        file = dirname

    raise IOError, "Can't find %s" % ifile
        
        
