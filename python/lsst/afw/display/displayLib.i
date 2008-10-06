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
#   include <boost/shared_ptr.hpp>
#   include <boost/any.hpp>
#   include <boost/array.hpp>
#   include <lsst/utils/Utils.h>
#   include <lsst/daf/base.h>
#   include <lsst/daf/data.h>
#   include <lsst/daf/persistence.h>
#   include <lsst/pex/exceptions.h>
#   include <lsst/pex/logging/Trace.h>
#   include <lsst/pex/policy/Policy.h>
#   include <lsst/afw/image.h>

#   include "simpleFits.h"
%}

%inline %{
namespace boost { namespace mpl { } } 
namespace lsst { namespace afw { namespace display { } } }
namespace lsst { namespace afw { namespace image { } } }
namespace lsst { namespace daf { namespace data { } } }
    
using namespace lsst::afw::display;
using namespace lsst::afw::image;
using namespace lsst::daf::data;
%}

%init %{
%}

%pythoncode %{
import lsst.afw.image
%}

%include "lsst/p_lsstSwig.i"
%import "lsst/utils/Utils.h"

SWIG_SHARED_PTR(CitizenPtr, lsst::daf::base::Citizen);

%import "lsst/daf/base/Citizen.h"
%import "lsst/daf/base/Persistable.h"
%import "lsst/daf/base/DataProperty.h"
%import "lsst/daf/data/LsstData.h"
%import "lsst/daf/data/LsstImpl_DC3.h"
%import "lsst/daf/data/LsstBase.h"
%import "lsst/daf/data.h"
%import "lsst/daf/persistence/Persistence.h"
%import "../image/image.i"
%import "../image/mask.i"
%import "../image/maskedImage.i"

/******************************************************************************/

%include "simpleFits.h"

%template(writeFitsImage) writeBasicFits<boost::uint16_t>;
%template(writeFitsImage) writeBasicFits<float>;
%template(writeFitsImage) writeBasicFits<double>;
//%template(writeFitsImage) writeBasicFits<lsst::afw::image::MaskPixel>; // == boost::unit16_t

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
