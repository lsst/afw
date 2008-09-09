# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

env = scons.makeEnv(
    "afw",
    r"$HeadURL$",
    [
        ["boost", "boost/version.hpp", "boost_filesystem:C++"],
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
        ["utils", "lsst/utils/Utils.h", "utils:C++"],
        ["daf_base", "lsst/daf/base.h", "daf_base:C++"],
        ["pex_exceptions", "lsst/pex/exceptions.h", "pex_exceptions:C++"],
        ["pex_logging", "lsst/pex/logging/Trace.h", "pex_logging:C++"],
        ["security", "lsst/security/Security.h", "security:C++"],
        ["pex_policy", "lsst/pex/policy/Policy.h", "pex_policy:C++"],
        ["daf_persistence", "lsst/daf/persistence.h", "daf_persistence:C++"],
        ["daf_data", "lsst/daf/data.h", "daf_data:C++"],
    ],
)
#
# Libraries needed to link libraries/executables
#
env.libs["afw"] += env.getlibs("boost vw wcslib cfitsio minuit utils daf_base daf_data daf_persistence pex_exceptions pex_logging pex_policy security")
#
# Build/install things
#
for d in (
    "doc",
    "examples",
    "include/lsst/afw",
    "lib",
    "python/lsst/afw/detection",
    "python/lsst/afw/display",
    "python/lsst/afw/image",
    "python/lsst/afw/math",
    "tests",
):
    SConscript(os.path.join(d, "SConscript"))

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", [
    env.Install(env['prefix'], "python"),
    env.Install(env['prefix'], "include"),
    env.Install(env['prefix'], "lib"),
    env.InstallAs(os.path.join(env['prefix'], "doc", "doxygen"), os.path.join("doc", "htmlDir")),
    env.InstallEups(os.path.join(env['prefix'], "ups"), glob.glob(os.path.join("ups", "*.table")))
])

scons.CleanTree(r"*~ core *.so *.os *.o")
#
# Build TAGS files
#
files = scons.filesToTag()
if files:
    env.Command("TAGS", files, "etags -o $TARGET $SOURCES")

env.Declare()
env.Help("""
LSST Application Framework packages
""")

