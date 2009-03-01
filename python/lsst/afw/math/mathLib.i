// -*- lsst-c++ -*-
%define mathLib_DOCSTRING
"
Python interface to lsst::afw::math classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.math",docstring=mathLib_DOCSTRING) mathLib

%{
#   include "lsst/daf/base.h"
#   include "lsst/pex/policy/Policy.h"
#   include "lsst/pex/policy/PolicyFile.h"
#   include "lsst/afw/image.h"
#   include "lsst/afw/math.h"
%}

%include "lsst/p_lsstSwig.i"

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/math/mathLib.i $"):
    """Return a version given a HeadURL string. If a different version is setup, return that too"""

    version_svn = lsst.utils.guessSvnVersion(HeadURL)

    try:
        import eups
    except ImportError:
        return version_svn
    else:
        try:
            version_eups = eups.setup("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

%template(vectorF) std::vector<float>;
%template(vectorD) std::vector<double>;
%template(vectorI) std::vector<int>;
%template(vectorVectorF) std::vector<std::vector<float> >;
%template(vectorVectorD) std::vector<std::vector<double> >;
%template(vectorVectorI) std::vector<std::vector<int> >;

%import "lsst/afw/image/imageLib.i"

%lsst_exceptions();

%include "functionLib.i"
%include "kernelLib.i"
%include "minimizeLib.i"
%include "statistics.i"
%include "interpolate.i"
%include "background.i"
%include "warpExposure.i"
%include "spatialCell.i"
