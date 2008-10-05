// -*- lsst-c++ -*-
%define displayLib_DOCSTRING
"
Basic routines to talk to ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=displayLib_DOCSTRING) displayLib

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
    
using namespace lsst::afw::display;
%}

%init %{
%}

%pythoncode %{
import lsst.afw.image
%}

%include "lsst/p_lsstSwig.i"
%import "lsst/utils/Utils.h"
%include "../image/lsstImageTypes.i"

/******************************************************************************/

%include "simpleFits.h"

%template(writeFitsImage) writeBasicFits<char>;
%template(writeFitsImage) writeBasicFits<boost::uint16_t>;
%template(writeFitsImage) writeBasicFits<float>;
%template(writeFitsImage) writeBasicFits<double>;
%template(writeFitsImage) writeBasicFits<lsst::afw::image::MaskPixel>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
