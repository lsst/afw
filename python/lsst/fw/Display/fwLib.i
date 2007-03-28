// -*- lsst-c++ -*-
%define fwLib_DOCSTRING
"
Basic routines to talk to FW's classes (including visionWorkbench) and ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=fwLib_DOCSTRING, naturalvar=1) fwLib

%{
#   include <boost/cstdint.hpp>
#   include "lsst/Citizen.h"
#   include "lsst/DiskImageResourceFITS.h"
#   include "simpleFits.h"
#   include "lsst/Mask.h"
#   include "lsst/MaskedImage.h"
#   include "lsst/Trace.h"

using namespace lsst;
using namespace lsst::image;
using namespace vw;
%}

// Suppress swig complaints from vw
// 317: Specialization of non-template
// 389: operator[] ignored
// 362: operator=  ignored
// 503: Can't wrap 'operator unspecified_bool_type()'
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=317
#pragma SWIG nowarn=362
#pragma SWIG nowarn=389
%warnfilter(503) vw;

%include "../Core/p_lsstSwig.i"
%import "lsst/Utils.h"

%{
typedef vw::PixelGray<float> ImagePixelType;
typedef vw::PixelGray<uint8> MaskPixelType;
%}

%import <vw/Image/ImageResource.h>
%import <vw/FileIO/DiskImageResource.h>

%import "lsst/DiskImageResourceFITS.h"

/******************************************************************************/
// Talk to DS9

%include "simpleFits.h"

using namespace lsst::image;

%template(readFloat) lsst::image::read<ImagePixelType>;

%template(writeFitsFloat) writeFits<ImagePixelType>;
%template(writeFitsFileFloat) writeFitsFile<ImagePixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
