// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
%define imageLib_DOCSTRING
"
Basic routines to talk to lsst::afw::image classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.image", docstring=imageLib_DOCSTRING) imageLib

// Suppress swig complaints
#pragma SWIG nowarn=314                 // print is a python keyword (--> _print)
#pragma SWIG nowarn=362                 // operator=  ignored


%{
#include "boost/cstdint.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/data.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/policy.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/image/Defect.h"
#include "lsst/afw/image/Calib.h"

#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_IMAGE_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.h"
#include "lsst/ndarray/python/eigen.h"

#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/formatters/TanWcsFormatter.h"
#include "lsst/afw/formatters/ExposureFormatter.h"
#include "lsst/afw/formatters/DecoratedImageFormatter.h"
%}

%include "../boost_picklable.i"

%init %{
    import_array();
%}

namespace boost {
    namespace mpl { }
    typedef signed char  int8_t;
    typedef int int32_t;
    typedef unsigned short uint16_t;
}

%apply unsigned long long { boost::uint64_t };

/************************************************************************************************************/

%typemap(typecheck, precedence=SWIG_TYPECHECK_BOOL, noblock=1) bool {
    $1 = PyBool_Check($input) ? 1 : 0;
}

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"

%include "lsst/base.h"

%pythoncode %{
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
            version_eups = eups.getSetupVersion("afw")
        except AttributeError:
            return version_svn

    if version_eups == version_svn:
        return version_svn
    else:
        return "%s (setup: %s)" % (version_svn, version_eups)

%}

/******************************************************************************/

%import "lsst/daf/base/baseLib.i"
%import "lsst/pex/policy/policyLib.i"
%import "lsst/daf/persistence/persistenceLib.i"
%import "lsst/daf/data/dataLib.i"
%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/coord/coordLib.i"

%include "lsst/ndarray/ndarray.i"

%declareNumPyConverters(Eigen::MatrixXd);
%declareNumPyConverters(Eigen::VectorXd);
%declareNumPyConverters(Eigen::Matrix2d);
%declareNumPyConverters(Eigen::Vector2d);
%declareNumPyConverters(Eigen::Matrix3d);
%declareNumPyConverters(Eigen::Vector3d);

%declareNumPyConverters(lsst::ndarray::Array<unsigned short,2,1>);
%declareNumPyConverters(lsst::ndarray::Array<unsigned short const,2,1>);

%declareNumPyConverters(lsst::ndarray::Array<int,2,1>);
%declareNumPyConverters(lsst::ndarray::Array<int const,2,1>);

%declareNumPyConverters(lsst::ndarray::Array<float,2,1>);
%declareNumPyConverters(lsst::ndarray::Array<float const,2,1>);

%declareNumPyConverters(lsst::ndarray::Array<double,2,1>);
%declareNumPyConverters(lsst::ndarray::Array<double const,2,1>);

%lsst_exceptions();

/******************************************************************************/

%template(pairIntInt)       std::pair<int, int>;
%template(pairIntDouble)    std::pair<int, double>;
%template(pairDoubleInt)    std::pair<double, int>;
%template(pairDoubleDouble) std::pair<double, double>;
%template(mapStringInt)     std::map<std::string, int>;

/************************************************************************************************************/
// Images, Masks, and MaskedImages
%include "lsst/afw/image/LsstImageTypes.h"

%ignore lsst::afw::image::Filter::operator int;
%include "lsst/afw/image/Filter.h"

SWIG_SHARED_PTR(CalibPtr, lsst::afw::image::Calib);
%include "lsst/afw/image/Calib.h"
%template(vectorCalib) std::vector<boost::shared_ptr<const lsst::afw::image::Calib> >;

#if defined(IMPORT_FUNCTION_I)
%{
#include "lsst/afw/math.h"
%}
%import "lsst/afw/math/function.i"
#undef IMPORT_FUNCTION_I
#endif

%include "image.i"
%include "mask.i"
%include "maskedImage.i"
%include "imageSlice.i"


%apply double &OUTPUT { double & };
%rename(positionToIndexAndResidual) lsst::afw::image::positionToIndex(double &, double);
%clear double &OUTPUT;

%include "lsst/afw/image/ImageUtils.h"

/************************************************************************************************************/
%{
namespace lsst { namespace afw { namespace image {
    extern Wcs NoWcs;
}}}
using lsst::afw::image::NoWcs;
%}

SWIG_SHARED_PTR(Wcs, lsst::afw::image::Wcs);
SWIG_SHARED_PTR_DERIVED(TanWcs, lsst::afw::image::Wcs, lsst::afw::image::TanWcs);

%ignore lsst::afw::image::NoWcs;

%{
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
%}


%include "lsst/afw/image/Wcs.h"
%include "lsst/afw/image/TanWcs.h"

%lsst_persistable(lsst::afw::image::Wcs);
%lsst_persistable(lsst::afw::image::TanWcs);

%boost_picklable(lsst::afw::image::Wcs);
%boost_picklable(lsst::afw::image::TanWcs);

%newobject makeWcs;

%inline %{
    lsst::afw::image::TanWcs::Ptr
    cast_TanWcs(lsst::afw::image::Wcs::Ptr wcs) {
        lsst::afw::image::TanWcs::Ptr tanWcs = boost::shared_dynamic_cast<lsst::afw::image::TanWcs>(wcs);
        
        if(tanWcs.get() == NULL) {
            throw(LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, "Up cast failed"));
        }
        return tanWcs;
    }
%}

/************************************************************************************************************/

#if !defined(CAMERA_GEOM_LIB_I)
%import "lsst/afw/cameraGeom/cameraGeomLib.i"
#endif

/************************************************************************************************************/
%{
#include "lsst/afw/image/Exposure.h"
%}

// Must go Before the %include
%define %exposurePtr(TYPE, PIXEL_TYPE)
SWIG_SHARED_PTR_DERIVED(Exposure##TYPE, lsst::daf::data::LsstBase, lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%enddef

// Must go After the %include
%define %exposure(TYPE, PIXEL_TYPE)
%newobject makeExposure;
%template(Exposure##TYPE) lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%template(makeExposure) lsst::afw::image::makeExposure<lsst::afw::image::MaskedImage<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%lsst_persistable(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%boost_picklable(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);

%extend lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> {
    %pythoncode {
    def Factory(self, *args):
        """Return an Exposure of this type"""
        return Exposure##TYPE(*args)
    }
}
%enddef

%exposurePtr(U, boost::uint16_t);
%exposurePtr(L, boost::uint64_t);
%exposurePtr(I, int);
%exposurePtr(F, float);
%exposurePtr(D, double);

namespace lsst { namespace afw { namespace detection {
    class Psf;
}}}
SWIG_SHARED_PTR(PsfPtr, lsst::afw::detection::Psf);

%include "lsst/afw/image/Exposure.h"

%exposure(U, boost::uint16_t);
%exposure(L, boost::uint64_t);
%exposure(I, int);
%exposure(F, float);
%exposure(D, double);


%extend lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> {
    %newobject convertF;
    lsst::afw::image::Exposure<float,
         lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> convertF()
    {
        return lsst::afw::image::Exposure<float,
            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(*self, true);
    }
}

%extend lsst::afw::image::Exposure<boost::uint64_t, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> {
    %newobject convertD;
    lsst::afw::image::Exposure<double,
         lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> convertD()
    {
        return lsst::afw::image::Exposure<double,
            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(*self, true);
    }
}

/************************************************************************************************************/

%include "lsst/afw/image/Color.h"

/************************************************************************************************************/

SWIG_SHARED_PTR(DefectPtr, lsst::afw::image::DefectBase);

%include "lsst/afw/image/Defect.h"

%template(DefectSet) std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> >;

/************************************************************************************************************/

