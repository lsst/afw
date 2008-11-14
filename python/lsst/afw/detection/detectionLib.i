// -*- lsst-c++ -*-
%define detectionLib_DOCSTRING
"
Python interface to lsst::afw::detection classes
"
%enddef

%feature("autodoc", "1");
%module(docstring=detectionLib_DOCSTRING) detectionLib

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

%{
#include "lsst/daf/base.h"
#include "lsst/daf/data.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/pex/policy/PolicyFile.h"
#include "lsst/afw/image.h"
%}

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"

%import "lsst/afw/image/imageLib.i"

%lsst_exceptions()

%include "source.i"
%include "footprints.i"
