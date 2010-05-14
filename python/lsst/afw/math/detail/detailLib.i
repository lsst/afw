// -*- lsst-c++ -*-
%define detailLib_DOCSTRING
"
Python interface to lsst::afw::math::detail classes and functions
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.math.detail", docstring=detailLib_DOCSTRING) detailLib

%{
#   include "lsst/daf/base.h"
#   include "lsst/pex/policy.h"
#   include "lsst/afw/image.h"
#   include "lsst/afw/geom.h"
#   include "lsst/afw/math.h"
%}

%include "lsst/p_lsstSwig.i"

%pythoncode %{
import lsst.utils

def version(HeadURL = r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/afw/trunk/python/lsst/afw/math/detail/detailLib.i $"):
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

%import "lsst/afw/image/imageLib.i"
%import "lsst/afw/math/mathLib.i"

%lsst_exceptions();

%include "convolve.i"
