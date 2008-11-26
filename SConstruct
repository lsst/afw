# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

env = scons.makeEnv("t493",
                    r"$HeadURL$",
                    [["boost", "boost/version.hpp", "boost_filesystem:C++"],
                     ["boost", "boost/regex.hpp", "boost_regex:C++"],
                     ["vw", "vw/Core.h", "vw:C++"],
                     ["vw", "vw/Core.h", "vwCore:C++"],
                     ["vw", "vw/Math.h", "vwMath:C++"],
                     ["lapack", None, "lapack", "dgesdd_"],
                     ["gsl", "gsl/gsl_matrix.h", "gslcblas gsl"],
                     ]
                    )

env.libs["t493"] += env.getlibs("boost vw gsl lapack")

for d in Split("tests"):
    SConscript(os.path.join(d, "SConscript"))

