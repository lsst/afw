# -*- python -*-
#
# Setup our environment
#
import glob
import lsst.SConsUtils as scons

env = scons.makeEnv("fw",
                    r"$HeadURL$",
                    [["python", "Python.h"],
                     ["boost", "boost/version.hpp", "boost_filesystem"],
                     ["visionWorkbench", "vw/Core.h", "vw"],
		     ["cfitsio", "fitsio.h", "cfitsio", "ffopen"],
                     ["wcstools", "wcs.h", "wcs", "wcscat"],
                     ["xpa", "xpa.h", "xpa", "XPAPuts"],
                     ])
#
# Libraries that I need to link things.  This should be handled better
#
env.libs = dict([
    ("boost",	Split("boost_filesystem-d")),
    ("fits",	Split("fitsio")),
    ("vw",	Split("vw vwCore vwFileIO")),
    ])
#
# Build/install things
#
SConscript("examples/SConscript")
SConscript("doc/SConscript")
SConscript("include/lsst/SConscript")
SConscript("lib/SConscript")
SConscript("src/SConscript")
SConscript("python/lsst/fw/Catalog/SConscript")
SConscript("python/lsst/fw/Display/SConscript")

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", env.Install(env['prefix'], "python"))
Alias("install", env.Install(env['prefix'], "include"))
Alias("install", env.Install(env['prefix'], "lib"))
Alias("install", env.Install(env['prefix'] + "/bin", glob.glob("bin/*.py")))
Alias("install", env.InstallEups(env['prefix'] + "/ups", glob.glob("ups/*.table")))

scons.CleanTree(r"*~ core *.so *.os *.o")

env.Declare()
env.Help("""
LSST FrameWork packages
""")

