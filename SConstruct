# -*- python -*-
#
# Setup our environment
#
import glob, os
import LSST.SConsUtils as scons

env = scons.makeEnv("fw",
                    r"$HeadURL: svn+ssh://lsstarchive.ncsa.uiuc.edu/DC2/fw/branches/rhl/SConstruct $",
                    [["python", "Python.h"],
                     ["support"],
                     ["wcstools", "wcs.h", "wcs", "wcscat"],
                     ])
#
# Build/install things
#
SConscript("python/fw/Catalog/SConscript")

env['IgnoreFiles'] = r"(~$|\.pyc$|^\.svn$)"

for dir in ("", "Catalog", "Collection", "Image", "Policy"):
    fulldir = os.path.join("python/fw", dir)
    Alias("install", env.Install(os.path.join(env['prefix'], fulldir), glob.glob(fulldir + "/*.py")))
    Alias("install", env.Install(os.path.join(env['prefix'], fulldir), glob.glob(fulldir + "/*.so")))
Alias("install", env.Install(env['prefix'] + "/bin", glob.glob("bin/*.py")))
Alias("install", env.InstallEups(env['prefix'] + "/ups", glob.glob("ups/*.table")))

env.Declare()
env.Help("""
Your help string here
""")
