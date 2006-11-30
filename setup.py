#!/usr/bin/env python
"""Installer for lsst.

To use:
- Install the prerequisites listed in doc/index.html
- Then do the usual LSST thing:

python setup.py install --lsst-home (or --lsst-devel)

To do:
- Build a recursive system that searches for setup.py files
  and runs them, creating the empty structure and __init__.py files
  as needed. It must have support for building in the right order.
"""
import os
import sys
import glob
from distutils.core import setup
from numarray.numarrayext import NumarrayExtension

PkgBase = "lsst.fw"
PyDir = ""


# list all packages and subpackages here
packages = [
	"lsst.fw",
	"lsst.fw.Catalog",
	"lsst.fw.Collection",
	"lsst.fw.Image",
	"lsst.fw.Policy",
]


# get setuputil
currSysPath = sys.path
#sys.path = [os.path.join(PyDir, "apps", "support")] + list(currSysPath)
import lsst.support.setuputil as setuputil
sys.path = currSysPath

# process sys.argv to handle --lsst-home, etc.
setuputil.procArgv()

# extension support needed by extCtgread for WCStools
try:
	WCSTOOLS = os.environ['WCSTOOLS']
except:
	print "Environment variable WCSTOOLS not defined"
	sys.exit(1)
if not os.path.exists(WCSTOOLS):
	print "WCSTOOLS = %r not found; fix before building" % WCSTOOLS
	sys.exit(1)

# try standard options:
if os.path.exists(os.path.join(WCSTOOLS, "libwcs")):
	WCSTOOLS_INC = os.path.join(WCSTOOLS, "libwcs")
	WCSTOOLS_LIB = WCSTOOLS_INC
else:
	WCSTOOLS_INC = os.path.join(WCSTOOLS, 'include')
	WCSTOOLS_LIB = os.path.join(WCSTOOLS, 'lib')
if not (os.path.exists(WCSTOOLS_INC) or os.path.exists(WCSTOOLS_LIB)):
	print "WCSTOOLS=%r must contain 'libwcs' or else 'include' and 'lib'" % WCSTOOLS
	sys.exit(1)
    
# extension support needed by PyWCSlib for WCSlib
try:
	WCSLIB = os.environ['WCSLIB']
except:
	print "Environment variable WCSLIB not defined; \n In your startup script, set to: /lsst_ibrix/DC1/WCSlib-4.2"
	sys.exit(1)
if not os.path.exists(WCSLIB):
	print "WCSLIB = %r not found; fix before building" % WCSLIB
	sys.exit(1)

WCSLIB_INC = os.path.join(WCSLIB, 'include', 'wcslib')
WCSLIB_LIB = os.path.join(WCSLIB, 'lib')
if not (os.path.exists(WCSLIB_INC) or os.path.exists(WCSLIB_LIB)):
	print "WCSLIB=%r must contain 'include/wcslib' and 'lib'" % WCSLIB
	sys.exit(1)
    
ext_modules = [
	NumarrayExtension(
		"lsst.fw.Catalog.ctgread",
		[os.path.join(PyDir, "Catalog", "extCtgread.c")],
		include_dirs = [WCSTOOLS_INC],
		library_dirs = [WCSTOOLS_LIB],
		libraries = ['m','wcs'],
	),
]		

print "packages=", packages
setup(
	name = PkgBase,
	description = "LSST framework",
	package_dir = {PkgBase: PyDir},
	packages = packages,
	ext_modules = ext_modules,
)
