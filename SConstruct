# -*- python -*-
#
# Setup our environment
#
import glob, os.path, sys, traceback
import lsst.SConsUtils as scons

env = scons.makeEnv(
    "afw",
    r"$HeadURL$",
    [
        ["boost", "boost/version.hpp", "boost_system:C++"],
        ["boost", "boost/version.hpp", "boost_filesystem:C++"],
        ["boost", "boost/regex.hpp", "boost_regex:C++"],
        ["boost", "boost/filesystem.hpp", "boost_system:C++"],
        ["boost", "boost/serialization/base_object.hpp", "boost_serialization:C++"],
        ["boost", "boost/test/unit_test.hpp", "boost_unit_test_framework:C++"],
        ["python", "Python.h"],
        ["cfitsio", "fitsio.h", "m cfitsio", "ffopen"],
        ["wcslib", "wcslib/wcs.h", "m wcs"], # remove m once SConsUtils bug fixed
        ["xpa", "xpa.h", "xpa", "XPAPuts"],
        ["minuit", "Minuit/FCNBase.h", "lcg_Minuit:C++"],
        ["gsl", "gsl/gsl_rng.h", "gslcblas gsl"],
        ["pex_exceptions", "lsst/pex/exceptions.h", "pex_exceptions:C++"],
        ["utils", "lsst/utils/Utils.h", "utils:C++"],
        ["daf_base", "lsst/daf/base.h", "daf_base:C++"],
        ["pex_logging", "lsst/pex/logging/Trace.h", "pex_logging:C++"],
        ["security", "lsst/security/Security.h", "security:C++"],
        ["pex_policy", "lsst/pex/policy/Policy.h", "pex_policy:C++"],
        ["daf_persistence", "lsst/daf/persistence.h", "daf_persistence:C++"],
        ["daf_data", "lsst/daf/data.h", "daf_data:C++"],
        ["eigen", "Eigen/Core.h"],
    ],
)
#
# Libraries needed to link libraries/executables
#
env.libs["afw"] += env.getlibs("boost wcslib cfitsio minuit gsl utils daf_base daf_data daf_persistence pex_exceptions pex_logging pex_policy security")
#
# Build/install things
#
for d in (
    ".",
    "doc",
    "examples",
    "lib",
    "python/lsst/afw/detection",
    "python/lsst/afw/display",
    "python/lsst/afw/image",
    "python/lsst/afw/math", 
    "python/lsst/afw/utils",
    "tests",
):
    if d != ".":
        try:
            SConscript(os.path.join(d, "SConscript"))
        except Exception, e:
            print >> sys.stderr, "In processing file %s:" % (os.path.join(d, "SConscript"))
            print >> sys.stderr, traceback.format_exc()
    Clean(d, Glob(os.path.join(d, "*~")))
    Clean(d, Glob(os.path.join(d, "*.pyc")))

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

