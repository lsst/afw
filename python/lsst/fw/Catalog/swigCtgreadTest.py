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
    count          Count of stars within specified region found in catalog;
                   0 if no match; int
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
import wcstools
import string

# initialize the scalar parameters
catfile = "search.gsc" 
refcat = -3             
distsort = 0
cra = 0.0
cdec = 0.0
dra = 0.277778		
ddec = 0.277778
drad = 0.277778
dradi = 0.0
sysout = 1
eqout = 2000.
epout = 2000.
mag1 = 0.0
mag2 = 0.0
sortmag = 0
nsmax = 100         # nsmax <1 -> print stars instead of storing to arrays
nlog = 1000            # nlog=0 -> quiet; nlog=1 -> verbose

try: 
    spp = wcstools.createNullStarCat(5)
except:
    print "could not createNullStarCat()\n"

try:
    tnum = wcstools.new_doubleArray(100)
    tra = wcstools.new_doubleArray(100)
    tdec = wcstools.new_doubleArray(100)
    tpra = wcstools.new_doubleArray(100)
    tpdec = wcstools.new_doubleArray(100)
    tc = wcstools.new_intArray(100)
    tmag = wcstools.new2D_DoubleArray(2,100,0.0)
    tobj = wcstools.new2D_charArray(100,80,"")
except:
    print "could not allocate output arrays\n"

try:
    for i in range(0,100):
       wcstools.doubleArray_setitem(tnum, i, 0.0)
       wcstools.doubleArray_setitem(tra, i, 0.0)
       wcstools.doubleArray_setitem(tdec, i, 0.0)
       wcstools.doubleArray_setitem(tpra, i, 0.0)
       wcstools.doubleArray_setitem(tpdec, i, 0.0)
       wcstools.intArray_setitem(tc, i, 0)
except:
    print "could not init output arrays\n"

try:
    (t_status) = \
        wcstools.ctgread(catfile,
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
            spp,
            tnum,
            tra,
            tdec, 
            tpra,
            tpdec,
            tmag,
            tc,
            tobj,
            nlog)
    (count) = t_status
    print "SUCCESS\n"
except:
    print '\nmain.py: Exception: \n',sys.exc_type,sys.exc_value,"\n"
    sys.exit(-1)
