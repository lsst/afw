# -*- python -*-
#
# Setup our environment
#
import glob, os.path, sys, traceback
import lsst.SConsUtils as scons

try:
    scons.ConfigureDependentProducts
except AttributeError:
    import lsst.afw.scons.SconsUtils
    scons.ConfigureDependentProducts = lsst.afw.scons.SconsUtils.ConfigureDependentProducts

env = scons.makeEnv(
    "afw",
    r"$HeadURL$",
    scons.ConfigureDependentProducts("afw"),
)

#
# Libraries needed to link libraries/executables
#
env.libs["afw"] += env.getlibs("boost wcslib cfitsio minuit2 gsl utils daf_base daf_data daf_persistence " +
    "pex_exceptions pex_logging pex_policy security fftw3")
if True:
    #
    # Workaround SConsUtils failure to find numpy .h files. Fixed in sconsUtils >= 3.3.2
    #
    import numpy
    env.Append(CCFLAGS = ["-I", numpy.get_include()])
#
# Build/install things
#
for d in (
    ".",
    "doc",
    "examples",
    "lib",
    "python/lsst/afw/cameraGeom",
    "python/lsst/afw/detection",
    "python/lsst/afw/display",
    "python/lsst/afw/image",
    "python/lsst/afw/geom", 
    "python/lsst/afw/math",
    "python/lsst/afw/math/detail",
    "python/lsst/afw/coord", 
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
    env.Install(env['prefix'], "doc"),
    env.Install(env['prefix'], "etc"),
    env.Install(env['prefix'], "examples"),
    env.Install(env['prefix'], "include"),
    env.Install(env['prefix'], "lib"),
    env.Install(env['prefix'], "policy"),
    env.Install(env['prefix'], "python"),
    env.Install(env['prefix'], "src"),
    env.Install(env['prefix'], "tests"),
    env.InstallEups(os.path.join(env['prefix'], "ups")),
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
