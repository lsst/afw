// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008-2015 AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
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
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging.h"
#include "lsst/pex/policy.h"
#include "lsst/afw/cameraGeom.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/image/Defect.h"
#include "lsst/afw/fits.h" // just for exceptions

#include "lsst/afw/formatters/WcsFormatter.h"
#include "lsst/afw/formatters/TanWcsFormatter.h"
#include "lsst/afw/formatters/ExposureFormatter.h"
#include "lsst/afw/formatters/DecoratedImageFormatter.h"

#pragma clang diagnostic ignored "-Warray-bounds" // PyTupleObject has an array declared as [1]
%}

%include "lsst/afw/fits/fits_reduce.i"

namespace boost {
    namespace mpl { }
}

/************************************************************************************************************/

%typemap(typecheck, precedence=SWIG_TYPECHECK_BOOL, noblock=1) bool {
    $1 = PyBool_Check($input) ? 1 : 0;
}

%include "lsst/p_lsstSwig.i"
%initializeNumPy(afw_image)
%{
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}
%include "lsst/daf/base/persistenceMacros.i"

%include "lsst/base.h"

/******************************************************************************/

%import "lsst/daf/base/baseLib.i"
%import "lsst/pex/policy/policyLib.i"
%import "lsst/daf/persistence/persistenceLib.i"
%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/geom/polygon/polygonLib.i"
%import "lsst/afw/coord/coordLib.i"
%import "lsst/afw/fits/fitsLib.i" // just for FITS exceptions

%include "ndarray.i"

%declareNumPyConverters(Eigen::MatrixXd);
%declareNumPyConverters(Eigen::VectorXd);
%declareNumPyConverters(Eigen::Matrix2d);
%declareNumPyConverters(Eigen::Vector2d);
%declareNumPyConverters(Eigen::Matrix3d);
%declareNumPyConverters(Eigen::Vector3d);

%declareNumPyConverters(ndarray::Array<unsigned short,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned short const,2,1>);

%declareNumPyConverters(ndarray::Array<int,2,1>);
%declareNumPyConverters(ndarray::Array<int const,2,1>);

%declareNumPyConverters(ndarray::Array<float,2,1>);
%declareNumPyConverters(ndarray::Array<float const,2,1>);

%declareNumPyConverters(ndarray::Array<double,2,1>);
%declareNumPyConverters(ndarray::Array<double const,2,1>);

%declareNumPyConverters(ndarray::Array<double,1>);
%declareNumPyConverters(ndarray::Array<double const,1>);

%lsst_exceptions();

/******************************************************************************/

%template(pairIntInt)       std::pair<int, int>;
%template(pairIntDouble)    std::pair<int, double>;
%template(pairDoubleInt)    std::pair<double, int>;
%template(pairDoubleDouble) std::pair<double, double>;
%template(mapStringInt)     std::map<std::string, int>;

%define %defineClone(PY_TYPE, TYPE, PIXEL_TYPES...)
%extend TYPE<PIXEL_TYPES> {
    %pythoncode %{
def clone(self):
    """Return a deep copy of self"""
    return PY_TYPE(self, True)

# Call our ctor with the provided arguments
def Factory(self, *args):
    """Return an object of this type"""
    return PY_TYPE(*args)
    %}
}
%enddef

/************************************************************************************************************/

%pythoncode %{
def _getBBoxFromSliceTuple(img, imageSlice):
    """Given a slice specification return the proper Box2I
    This is the worker routine behind __getitem__ and __setitem__

    The imageSlice may be:
       lsst.afw.geom.Box2I
       slice, slice
       :
    Only the first one or two parts of the slice are recognised (no stride), a single int is
    interpreted as n:n+1, and negative indices are interpreted relative to the end of the array,
    so supported slices include:
       2
       -1
       1:10
       :-2
       : (equivalent to ... (python's Ellipsis) which is also supported)

    E.g.
     im[-1, :]
     im[..., 18]
     im[4,  10]
     im[-3:, -2:]
     im[-2, -2]
     im[1:4, 6:10]
     im[:]
    """
    afwGeom = lsst.afw.geom.geomLib

    if isinstance(imageSlice, afwGeom.Box2I):
        return imageSlice

    if isinstance(imageSlice, slice) and imageSlice.start is None and imageSlice.stop is None:
        imageSlice = (Ellipsis, Ellipsis,)

    if not (isinstance(imageSlice, tuple) and len(imageSlice) == 2 and \
                sum([isinstance(_, (slice, type(Ellipsis), int)) for _ in imageSlice]) == 2):
        raise IndexError("Images may only be indexed as a 2-D slice not %s", imageSlice)

    imageSlice, _imageSlice = [], imageSlice
    for s, wh in zip(_imageSlice, img.getDimensions()):
        if isinstance(s, slice):
            pass
        elif isinstance(s, int):
            if s < 0:
                s += wh
            s = slice(s, s + 1)
        else:
            s = slice(0, wh)

        imageSlice.append(s)

    x, y = [_.indices(wh) for _, wh in zip(imageSlice, img.getDimensions())]
    return afwGeom.Box2I(afwGeom.Point2I(x[0], y[0]), afwGeom.Point2I(x[1] - 1, y[1] - 1))
%}

%define %supportSlicing(TYPE, PIXEL_TYPES...)
%extend TYPE<PIXEL_TYPES> {
    %pythoncode %{
#
# Support image slicing
#
def __getitem__(self, imageSlice):
    """
    __getitem__(self, imageSlice) -> NAME""" + """PIXEL_TYPES
    """
    return self.Factory(self, _getBBoxFromSliceTuple(self, imageSlice), LOCAL)

def __setitem__(self, imageSlice, rhs):
    """
    __setitem__(self, imageSlice, value)
    """
    bbox = _getBBoxFromSliceTuple(self, imageSlice)
    try:
        self.assign(rhs, bbox, LOCAL)
    except NotImplementedError:
        lhs = self.Factory(self, bbox, LOCAL)
        lhs.set(rhs)

def __float__(self):
    """Convert a 1x1 image to a floating scalar"""
    if self.getDimensions() != lsst.afw.geom.geomLib.Extent2I(1, 1):
        raise TypeError("Only single-pixel images may be converted to python scalars")

    try:
        return float(self.get(0, 0))
    except AttributeError:
        raise TypeError("Unable to extract a single pixel for type %s" % "TYPE")
    except TypeError:
        raise TypeError("Unable to convert a %s<%s> pixel to a scalar" % ("TYPE", "PIXEL_TYPES"))

def __int__(self):
    """Convert a 1x1 image to a integral scalar"""
    return int(float(self))
    %}
}

%enddef

/************************************************************************************************************/
// Images, Masks, and MaskedImages
%include "lsst/afw/image/LsstImageTypes.h"

%ignore lsst::afw::image::Filter::operator int;
%include "lsst/afw/image/Filter.h"

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

%include "wcs.i"

/************************************************************************************************************/

#if !defined(CAMERA_GEOM_LIB_I)
%import "lsst/afw/cameraGeom/cameraGeomLib.i"
#endif


/************************************************************************************************************/

%include "lsst/afw/image/Calib.i"
%include "lsst/afw/image/ApCorrMap.i"

%{
#include "lsst/afw/detection.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/Exposure.h"
%}

// Must go Before the %include
%define %exposurePtr(PIXEL_TYPE)
%shared_ptr(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%enddef

// Must go After the %include
%define %exposure(TYPE, PIXEL_TYPE)
%newobject makeExposure;
%template(Exposure##TYPE) lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%template(makeExposure) lsst::afw::image::makeExposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%lsst_persistable(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%fits_reduce(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);

%supportSlicing(lsst::afw::image::Exposure,
                PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel);
%defineClone(Exposure##TYPE, lsst::afw::image::Exposure,
             PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel);
%enddef

%exposurePtr(boost::uint16_t);
%exposurePtr(boost::uint64_t);
%exposurePtr(int);
%exposurePtr(float);
%exposurePtr(double);

namespace lsst { namespace afw { namespace detection {
    class Psf;
}}}
%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::image::CoaddInputs);
%shared_ptr(lsst::afw::image::ExposureInfo);

%import "lsst/afw/table/Exposure.i"

%include "lsst/afw/image/CoaddInputs.h"
%include "lsst/afw/image/ExposureInfo.h"

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

%shared_ptr(lsst::afw::image::DefectBase);

%include "lsst/afw/image/Defect.h"

%template(DefectSet) std::vector<boost::shared_ptr<lsst::afw::image::DefectBase> >;

/************************************************************************************************************/
