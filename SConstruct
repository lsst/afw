# -*- python -*-
#
# Setup our environment
#
import glob, os
import LSST.SConsUtils as scons

env = scons.makeEnv("fw",
                    r"$HeadURL$",
                    [["python", "Python.h"],
                     ["boost", "boost/version.hpp", "boost_filesystem"],
                     ["visionWorkbench", "vw/vw.h", "vw"],
                     ["support"],
                     ["wcstools", "wcs.h", "wcs", "wcscat"],
                     ])
#
# Build/install things
#
SConscript("examples/SConscript")
SConscript("doc/SConscript")
SConscript("lib/SConscript")
SConscript("src/SConscript")
SConscript("python/lsst/fw/Catalog/SConscript")

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", env.Install(env['prefix'], "python"))
Alias("install", env.Install(env['prefix'] + "/bin", glob.glob("bin/*.py")))
Alias("install", env.InstallEups(env['prefix'] + "/ups", glob.glob("ups/*.table")))

scons.CleanTree(r"*~ core *.so *.os *.o")

env.Declare()
env.Help("""
LSST FrameWork packages
""")
