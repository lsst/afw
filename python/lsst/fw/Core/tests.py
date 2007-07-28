"""Support code for running unit tests"""

import unittest
import lsst.fw.Core.fwLib as fw
import os
import sys

try:
    type(memId0)
except NameError:
    memId0 = 0

def run(suite):
    """Exit with the status code resulting from running the provided test suite"""
    status = 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1
    sys.exit(status)

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
        
        
