// -*- lsst-c++ -*-
%define ds9Lib_DOCSTRING
"
Basic routines to allow visionWorkbench to talk to ds9
"
%enddef

%feature("autodoc", "1");
%module(docstring=ds9Lib_DOCSTRING, naturalvar=1) ds9Lib

#if 0
   %rename("%(command:perl -pe 's/^act(.)/\l$1/' <<< )s") "";
#else
//   %include "ds9Rename.i"
#endif

%{
#   include "lsst/DiskImageResourceFITS.h"
#   include "simpleFits.h"

using namespace lsst::image;
using namespace vw;
%}

%init %{
%}

%include "p_lsstSwig.i"
%import "lsst/Utils.h"

/******************************************************************************/

//%include "ds9.h"

%ignore vw::ImageView<float>::origin;

%import <vw/Image/ImageView.h>
%import <vw/Image/PixelTypeInfo.h>
%import <vw/Image/PixelTypes.h>
%import <vw/Image/ImageResource.h>

%include "lsst/DiskImageResourceFITS.h"

/******************************************************************************/
// Talk to DS9

%include "simpleFits.h"

using namespace lsst::image;

%template(rhlWriteFitsFloat) rhlWriteFits<float>;
%template(rhlWriteFitsFileFloat) rhlWriteFitsFile<float>;

%template(imageFloat) vw::ImageView<float>;
%template(readFloat) lsst::image::read<float>;

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
