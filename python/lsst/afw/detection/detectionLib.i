// -*- lsst-c++ -*-
%define detectionLib_DOCSTRING
"
Python interface to lsst::afw::detection classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.detection", docstring=detectionLib_DOCSTRING) detectionLib

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

%{
#include "lsst/daf/base.h"
#include "lsst/daf/data.h"
#include "lsst/pex/logging/Log.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/logging/LogFormatter.h"
#include "lsst/pex/policy.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image.h"
#include "lsst/afw/detection/Psf.h"
%}

%inline %{
namespace boost {
    typedef short int16_t;
    typedef unsigned short uint16_t;
    typedef int int32_t;
    typedef signed char int8_t;
}
%}

%include "std_string.i"

%include "lsst/p_lsstSwig.i"
%import  "lsst/afw/utils.i" 
%include "lsst/daf/base/persistenceMacros.i"

%import "lsst/afw/image/imageLib.i"
%import "lsst/afw/math/mathLib.i"

%lsst_exceptions()

%include "source.i"
%include "footprints.i"
%include "match.i"
%include "psf.i"
