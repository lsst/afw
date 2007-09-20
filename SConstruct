# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

env = scons.makeEnv("fw",
                    r"$HeadURL$",
                    [["boost", "boost/version.hpp", "boost_filesystem:C++"],
                     ["boost", "boost/regex.hpp", "boost_regex:C++"],
                     ["vw", "vw/Core.h", "vw:C++"],
                     ["vw", "vw/Core.h", "vwCore:C++"],
                     ["vw", "vw/FileIO.h", "vwFileIO:C++"],
                     ["vw", "vw/Image.h", "vwImage:C++"],
                     ["python", "Python.h"],
                     ["cfitsio", "fitsio.h", "m cfitsio", "ffopen"],
                     ["wcslib", "wcslib/wcs.h", "m wcs"], # remove m once SConsUtils bug fixed
                     ["xpa", "xpa.h", "xpa", "XPAPuts"],
                     ["mwi", "lsst/mwi/data.h", "boost_filesystem boost_regex mwi:C++"],
                     ])
#
# Libraries needed to link libraries/executables
#
env.libs["fitsio"] = ["fitsio"]         # Our fitsio library.  This should probably be part of libfw
env.libs["fw"] += env.getlibs("boost mwi vw wcslib") # we'll always want to link these with fw
#
# Build/install things
#
for d in Split("include/lsst/fw doc examples lib src tests"):
    SConscript(os.path.join(d, "SConscript"))

for d in map(lambda str: "python/lsst/fw/" + str,
             Split("Catalog Core Display")):
    SConscript(os.path.join(d, "SConscript"))

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", [env.Install(env['prefix'], "python"),
                  env.Install(env['prefix'], "include"),
                  env.Install(env['prefix'], "lib"),
                  env.InstallEups(env['prefix'] + "/ups", glob.glob("ups/*.table"))])

scons.CleanTree(r"*~ core *.so *.os *.o")
#
# Build TAGS files
#
files = scons.filesToTag()
if files:
    env.Command("TAGS", files, "etags -o $TARGET $SOURCES")

env.Declare()
env.Help("""
LSST FrameWork packages
""")

