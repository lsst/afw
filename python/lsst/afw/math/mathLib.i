// -*- lsst-c++ -*-
%define mathLib_DOCSTRING
"
Python interface to lsst::afw::math classes
"
%enddef

%feature("autodoc", "1");
%module(docstring=mathLib_DOCSTRING) imageLib

// Suppress swig complaints
// copied from afw/image/imageLib.i
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=317                 // specialization of non-template
#pragma SWIG nowarn=362                 // operator=  ignored
#pragma SWIG nowarn=389                 // operator[] ignored
#pragma SWIG nowarn=503                 // Can't wrap 'operator unspecified_bool_type'

// define basic vectors
// these are used by Kernel and Function (and possibly other code)
%include "std_vector.i"
%template(vectorF) std::vector<float>;
%template(vectorD) std::vector<double>;
%template(vectorVectorF) std::vector<std::vector<float> >;
%template(vectorVectorD) std::vector<std::vector<double> >;

%{
#   include <fstream>
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/static_assert.hpp>
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include <boost/array.hpp>
#   include "lsst/utils.h"
#   include "lsst/daf/base.h"
#   include "lsst/pex/logging/Trace.h"
#   include "lsst/afw/image.h"
#   include "lsst/afw/math.h"
%}

%init %{
%}

%include "lsst/p_lsstSwig.i"
%include "../image/lsstImageTypes.i"

%pythoncode %{
import lsst.daf.data
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

%include "functionLib.i"
%include "kernelLib.i"
%include "minimizeLib.i"
