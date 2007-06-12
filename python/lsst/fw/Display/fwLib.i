// -*- lsst-c++ -*-
%define fwDisplay_DOCSTRING
"
Basic routines to talk to FW's classes (including visionWorkbench) and ds9
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
#   include "lsst/fw/MaskedImage.h"
#   include "simpleFits.h"
%}

%inline %{
namespace lsst { namespace fw { } }
namespace vw {}
    
using namespace lsst;
using namespace lsst::fw;
using namespace vw;
%}

%init %{
%}

%include "../Core/p_lsstSwig.i"
%import "lsst/fw/Utils.h"

/******************************************************************************/

%include "simpleFits.h"

%template(writeFitsImage) writeFits<int>;
%template(writeFitsImage) writeFits<ImagePixelType>;
%template(writeFitsMask)  writeFits<MaskPixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
