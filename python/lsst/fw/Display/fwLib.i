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
#   include "lsst/fw/Citizen.h"
#   include "lsst/fw/DiskImageResourceFITS.h"
#   include "simpleFits.h"
#   include "lsst/fw/Mask.h"
#   include "lsst/fw/MaskedImage.h"
#   include "lsst/fw/Trace.h"

using namespace lsst;
using namespace lsst::fw;
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
%import "lsst/fw/Utils.h"

%import <vw/Image/ImageResource.h>
%import <vw/FileIO/DiskImageResource.h>

%import "lsst/fw/DiskImageResourceFITS.h"

/******************************************************************************/
// Talk to DS9

using namespace lsst::fw;
//
// I don't know why I need to say this; but I do.
//
%feature("novaluewrapper") boost::shared_ptr<lsst::fw::Image<ImagePixelType> >;
%feature("novaluewrapper") boost::shared_ptr<lsst::fw::Image<MaskPixelType> >;

%import "lsst/fw/Mask.h"
%import "lsst/fw/Image.h"
%import "lsst/fw/MaskedImage.h"

%template(MaskD)          lsst::fw::Mask<MaskPixelType>;
%template(MaskDPtr)       boost::shared_ptr<lsst::fw::Mask<MaskPixelType> >;
%template(ImageD)         lsst::fw::Image<ImagePixelType>;
%template(ImageDPtr)      boost::shared_ptr<lsst::fw::Image<ImagePixelType> >;
%template(ImageMaskD)     lsst::fw::Image<MaskPixelType>;
%template(ImageMaskDPtr)  boost::shared_ptr<lsst::fw::Image<MaskPixelType> >;
%template(MaskedImageD)	  lsst::fw::MaskedImage<ImagePixelType, MaskPixelType>;
%template(MaskedImageDPtr) boost::shared_ptr<lsst::fw::MaskedImage<ImagePixelType, MaskPixelType> >;

%include "simpleFits.h"

%template(readMask) read<MaskPixelType>;
%template(writeFits) writeFits<MaskPixelType>;

%template(readImage) read<ImagePixelType>;
%template(writeFits) writeFits<ImagePixelType>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
