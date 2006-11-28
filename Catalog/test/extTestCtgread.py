#!/usr/bin/env python
"""
Tester demonstrating use of the python interface to ctgread

(count, tnum, tra, tdec, tpra, tpdec, tmag1,tmag2, o_tc) =
    ctgread (catfile, refcat, distsort, cra, cdec, dra, ddec, drad, dradi,
             sysout, eqout, epout, mag1, mag2, sortmag, nsmax, nlog)
                                                                                
Input:
    catfile         Name of reference star catalog file; ascii
    refcat          Catalog code from wcslib.h ; int
    distsort        1 to sort stars by distance from center; else no sort
    cra             Search center J2000 right ascension in degrees; float
    cdec            Search center J2000 declination in degrees; float
    dra             Search half width in right ascension in degrees; float
    ddec            Search half-width in declination in degrees; float
    drad            Limiting separation in degrees (ignore if 0); float
    dradi           Inner edge of annulus in degrees (ignore if 0); float
    sysout          Search coordinate system; int
    eqout           Search coordinate equinox; float
    epout           Proper motion epoch (0.0 for no proper motion); float
    mag1           Limiting magnitudes (none if equal); float
    mag2           Limiting magnitudes (none if equal); float
    sortmag         Number of magnitude by which to limit and sort; int
    nsmax           Maximum number of stars to be returned; int
    nlog           int


Output:
    count          Count of stars (upto nsmax) within specified region 
                   found in catalog; 0 if no match; int
    tnum           Array of ID numbers; float
    tra            Array of right ascensions; float
    tdec           Array of declinations; float
    tpra           Array of right ascension proper motions; float
    tpdec          Array of declination proper motions; float
    tmag1          Array of magnitudes; float
    tmag2          Array of magnitudes; float
    tc             Array of fluxes; int

Review tester code to see how to catch exceptions 

"""

import sys
import os
from lsst.apps.fw.Catalog.ctgread import *
# access via:  ctgread()
#--------------------------------------------------------------


import numarray
import string

# initialize the scalar parameters
catfile = "ty2"    # for GSC => "GSC"  TYCHO-2 -> "ty2"
refcat = 16        # for GSC => 1      TYCHO-2 -> 16   (see wcscat.h)
distsort = 0
cra = 0.0
cdec = 0.0
dra = 1.
ddec = 1.
drad = 1.
dradi = 0.0
sysout = 1
eqout = 2000.
epout = 2000.
mag1 = 0.0
mag2 = 0.0
sortmag = 0
nsmax = 100         # nsmax <1 -> print stars instead of storing to arrays
nlog = 1            # nlog=0 -> quiet; nlog=1 -> verbose

try:
    (t_status) = \
        ctgread(catfile,
            refcat,
            distsort,
            cra,
            cdec,
            dra,
            ddec,
            drad,
            dradi,
            sysout,
            eqout,
            epout,
            mag1,
            mag2,
            sortmag,
            nsmax,
            nlog)
    (count,tnum,tra,tdec,tpra,tpdec,tmag,tc) = t_status
    if ( count > 0 ):
        print "=============================================================="
        print "main.py: Found %d stars" % count
        for i in range ( count) :
            print "i:",i," tnum:",tnum[i]," tra:",tra[i]," tdec:",tdec[i]," tpra:",tpra[i]," tpdec:",tpdec[i]," tmag1:",tmag[0][i]," tmag2:",tmag[1][i]," tc:",tc[i]
    else:
        print "\nNo catalog stars overlapped the region\n"

except:
    print '\nmain.py: Exception: \n',sys.exc_type,sys.exc_value,"\n"
    sys.exit(-1)
    
