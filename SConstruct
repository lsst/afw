# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

env = scons.makeEnv("afw",
                    r"$HeadURL$",
                    [["boost", "boost/version.hpp", "boost_filesystem:C++"],
                     ["boost", "boost/regex.hpp", "boost_regex:C++"],
                     ["boost", "boost/serialization/base_object.hpp", "boost_serialization:C++"],
                     ["vw", "vw/Core.h", "vw:C++"],
                     ["vw", "vw/Core.h", "vwCore:C++"],
                     ["vw", "vw/FileIO.h", "vwFileIO:C++"],
                     ["vw", "vw/Image.h", "vwImage:C++"],
                     ["python", "Python.h"],
                     ["cfitsio", "fitsio.h", "m cfitsio", "ffopen"],
                     ["wcslib", "wcslib/wcs.h", "m wcs"], # remove m once SConsUtils bug fixed
                     ["xpa", "xpa.h", "xpa", "XPAPuts"],
                     ["minuit", "Minuit/FCNBase.h", "lcg_Minuit:C++"],
                     ["mwi", "lsst/mwi/data.h", "mwi:C++"],
                     ["daf_base", "lsst/daf/base.h", "daf_base:C++"],
                     ["daf_data", "lsst/daf/data.h", "daf_data:C++"],
                     ["daf_persistence", "lsst/daf/persistence.h", "daf_persistence:C++"],
                     ["pex_exceptions", "lsst/pex/exceptions.h", "pex_exceptions:C++"],
                     ["pex_logging", "lsst/pex/logging/Trace.h", "pex_logging:C++"],
                     ["pex_policy", "lsst/pex/policy/Policy.h", "pex_policy:C++"],
                     ])
#
# Libraries needed to link libraries/executables
#
env.libs["afw"] += env.getlibs("boost vw wcslib cfitsio minuit mwi") # we'll always want to link these with afw
#
# Build/install things
#
for d in Split("include/lsst/afw doc examples lib src tests"):
    SConscript(os.path.join(d, "SConscript"))

for d in map(lambda str: "python/lsst/afw/" + str,
             Split("Catalog Core Display")):
    SConscript(os.path.join(d, "SConscript"))

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", [env.Install(env['prefix'], "python"),
                  env.Install(env['prefix'], "include"),
                  env.Install(env['prefix'], "lib"),
                  env.InstallAs(os.path.join(env['prefix'], "doc", "doxygen"),
                                os.path.join("doc", "htmlDir")),
                  env.InstallEups(os.path.join(env['prefix'], "ups"),
                                  glob.glob(os.path.join("ups", "*.table")))])

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

