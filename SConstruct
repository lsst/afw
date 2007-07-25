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
                     ["python", "Python.h"],
                     ["cfitsio", "fitsio.h", "m cfitsio", "ffopen"],
                     ["wcslib", "wcslib/wcs.h", "m wcs"], # remove m once SConsUtils bug fixed
                     ["xpa", "xpa.h", "xpa", "XPAPuts"],
                     ])
#
# Libraries that I need to link things.  This should be handled better
#
env.libs = dict([
    ("boost",   Split("boost_filesystem boost_regex")),
    ("fits",    Split("fitsio")),
    ("vw",      Split("vw vwCore vwFileIO vwImage")),
    ("wcs",     Split("wcs")),
    ])
#
# Build/install things
#
for d in Split("doc examples include/lsst/fw lib src tests"):
    SConscript(os.path.join(d, "SConscript"))

for d in map(lambda str: "python/lsst/fw/" + str,
             Split("Catalog Core Display")):
    SConscript(os.path.join(d, "SConscript"))

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$|\.o$)"

Alias("install", env.Install(env['prefix'], "python"))
Alias("install", env.Install(env['prefix'], "include"))
Alias("install", env.Install(env['prefix'], "lib"))
Alias("install", env.Install(env['prefix'] + "/bin", glob.glob("bin/*.py")))
Alias("install", env.InstallEups(env['prefix'] + "/ups", glob.glob("ups/*.table")))

scons.CleanTree(r"*~ core *.so *.os *.o")
#
# Build TAGS files
#
if len(filter(lambda t: t == "TAGS", scons.COMMAND_LINE_TARGETS)) > 0:
    try:
        env.Command("TAGS", scons.filesToTag(), "etags -o $TARGET $SOURCES")
    except AttributeError:                  # not in this version of sconsUtils
        pass

env.Declare()
env.Help("""
LSST FrameWork packages
""")

