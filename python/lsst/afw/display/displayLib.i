// -*- lsst-c++ -*-
%define fwDisplay_DOCSTRING
"
Basic routines to talk to ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwDisplay_DOCSTRING) fwDisplay

// Suppress swig complaints from vw
// 317: Specialization of non-template
// 389: operator[] ignored
// 362: operator=  ignored
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=317
#pragma SWIG nowarn=362
#pragma SWIG nowarn=389

%{
#   include "lsst/afw/image/MaskedImage.h"
#   include "simpleFits.h"
%}

%inline %{
namespace lsst { namespace afw { } }
namespace lsst { namespace afw { namespace display { } } }
namespace vw {}
    
using namespace lsst::afw::display;
using namespace vw;
%}

%init %{
%}

%pythoncode %{
import lsst.fw.exceptions
%}

%include "lsst/p_lsstSwig.i"
%import "lsst/utils/Utils.h"
%include "../image/lsstImageTypes.i"

/******************************************************************************/

%include "simpleFits.h"

%template(writeFitsImage) writeFits<char>;
%template(writeFitsImage) writeFits<boost::uint16_t>;
%template(writeFitsImage) writeFits<float>;
%template(writeFitsImage) writeFits<double>;
%template(writeFitsImage) writeFits<lsst::afw::image::maskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
