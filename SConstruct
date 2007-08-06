# -*- python -*-
#
# Setup our environment
#
import glob, os.path, re, os
import lsst.SConsUtils as scons

def selectBoostLibs(env, boostlibs):
    """convert a list of boost library names into actual boost library file
    names.

    Boost library files, unfortunately, are shipped with platform- and
    verions-specific identifiers in their filenames.  This provides a
    translation from the generic component of the boost library name (e.g.
    "boost_regex") and the best choice among the available versions of
    that library.  Typically the boostlibs parameter only contains generic 
    boost library names; however, if the name does not start with "boost",
    it will be returned unchanged.
    
    @param boostlibs   the list (or space-limited string concatonation) of
                         boost libraries.  
    """
    (topdir, incdir, libdir) = scons.searchEnvForDirs(env, "boost")
    actual = None
    out = []
    
    boostlibs = Split(boostlibs)
    for lib in boostlibs:
        if re.match(r"boost", lib):
#            actual = scons.mangleLibraryName(env, libdir, lib)
# The following works with scons 1.15
            actual = scons.chooseBoostLib(env, libdir, lib)
            if actual is None: actual = lib
        else:
            actual = lib
        out.append(actual)
    
    return out

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
for d in Split("doc examples lib src tests"):
    SConscript(os.path.join(d, "SConscript"))

for d in map(lambda str: "python/lsst/fw/" + str,
             Split("Catalog Core Display")):
    SConscript(os.path.join(d, "SConscript"))

Alias("install", [env.InstallDir(env['prefix'], "python"),
                  env.InstallDir(env['prefix'], "include"),
                  env.InstallDir(env['prefix'], "lib"),
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

