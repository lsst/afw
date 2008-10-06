// -*- lsst-c++ -*-
%define imageLib_DOCSTRING
"
Basic routines to talk to lsst::afw::image classes
"
%enddef

%feature("autodoc", "1");
%module(docstring=imageLib_DOCSTRING) imageLib

// Suppress swig complaints
// I had trouble getting %warnfilter to work; hence the pragmas
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored

%{
#   include <fstream>
#   include <exception>
#   include <map>
#   include <boost/cstdint.hpp>
#   include <boost/static_assert.hpp>
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
%}

%inline %{
namespace lsst { namespace afw { namespace image { } } }
namespace lsst { namespace daf { namespace data { } } }
namespace boost {
    typedef unsigned short uint16_t;
    namespace filesystem {}
    namespace mpl {}
    class bad_any_cast;                 // for lsst/pex/policy/Policy.h
}
    
using namespace lsst;
using namespace lsst::afw::image;
using namespace lsst::daf::data;
%}

%init %{
%}

/************************************************************************************************************/

%include "lsst/p_lsstSwig.i"
%include "lsstImageTypes.i"     // Image/Mask types and typedefs

%pythoncode %{
import lsst.daf.data
import lsst.utils

def version(HeadURL = r"$HeadURL$"):
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

/******************************************************************************/

//SWIG_SHARED_PTR(CitizenPtr, lsst::daf::base::Citizen);

%import "lsst/daf/base/Citizen.h"
%import "lsst/daf/base/Persistable.h"
%import "lsst/daf/base/DataProperty.h"
%import "lsst/daf/data/LsstData.h"
%import "lsst/daf/data/LsstImpl_DC3.h"
%import "lsst/daf/data/LsstBase.h"
%import "lsst/daf/data.h"
%import "lsst/daf/persistence/Persistence.h"
%import "lsst/pex/exceptions.h"
%import "lsst/pex/logging/Trace.h"
%import "lsst/pex/policy/Policy.h"

/******************************************************************************/

%template(pairIntInt)   std::pair<int, int>;
%template(mapStringInt) std::map<std::string, int>;

/************************************************************************************************************/
// Images, Masks, and MaskedImages
%ignore lsst::afw::image::Filter::operator int;
%include "lsst/afw/image/Filter.h"

%include "image.i"
%include "mask.i"
%include "maskedImage.i"

%template(PointD) lsst::afw::image::Point<double>;
%template(PointI) lsst::afw::image::Point<int>;

%apply double &OUTPUT { double & };
%rename(positionToIndexAndResidual) lsst::afw::image::positionToIndex(double &, double);
%clear double &OUTPUT;

%include "lsst/afw/image/ImageUtils.h"

/************************************************************************************************************/

%{
#include "lsst/afw/image/Exposure.h"
%}

%include "lsst/afw/image/Exposure.h"

%template(ExposureU)    lsst::afw::image::Exposure<boost::uint16_t>;
%template(ExposureI)    lsst::afw::image::Exposure<int>;
%template(ExposureF)    lsst::afw::image::Exposure<float>;
%template(ExposureD)    lsst::afw::image::Exposure<double>;

/************************************************************************************************************/

%{
    #include "lsst/afw/image/Wcs.h"
%}

%rename(isValid) operator bool;

%include "lsst/afw/image/Wcs.h"

/************************************************************************************************************/

#if 0
%include "simpleFits.h"

%template(writeFitsImage) writeBasicFits<boost::uint16_t>;
%template(writeFitsImage) writeBasicFits<float>;
%template(writeFitsImage) writeBasicFits<double>;
//%template(writeFitsImage) writeBasicFits<lsst::afw::image::MaskPixel>; // == boost::unit16_t
#endif

/******************************************************************************/
// Local Variables: ***
// eval: (setq indent-tabs-mode nil) ***
// End: ***
