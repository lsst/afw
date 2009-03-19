// -*- lsst-c++ -*-
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
#include <lsst/daf/base.h>
#include <lsst/daf/data.h>
#include <lsst/daf/persistence.h>
#include <lsst/pex/exceptions.h>
#include <lsst/pex/logging/Trace.h>
#include <lsst/pex/policy.h>
#include <lsst/afw/image.h>
#include <boost/cstdint.hpp>
%}


namespace boost {
    namespace mpl { }
    typedef signed char  int8_t;
    typedef int int32_t;
    typedef unsigned short uint16_t;
}

/************************************************************************************************************/

%include "lsst/p_lsstSwig.i"
%include "lsst/daf/base/persistenceMacros.i"

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
            version_eups = eups.setup("afw")
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

%lsst_exceptions();

/******************************************************************************/

%template(pairIntInt)   std::pair<int, int>;
%template(mapStringInt) std::map<std::string, int>;

/************************************************************************************************************/
// Images, Masks, and MaskedImages
%include "lsst/afw/image/LsstImageTypes.h"

%ignore lsst::afw::image::Filter::operator int;
%include "lsst/afw/image/Filter.h"

%include "image.i"
%include "mask.i"
%include "maskedImage.i"

%template(PointD) lsst::afw::image::Point<double>;
%template(PointI) lsst::afw::image::Point<int>;

%define %EXTEND_POINT(TYPE)
%extend lsst::afw::image::Point<TYPE> {
    %pythoncode {
    def __repr__(self):
        return "(%.10g, %.10g)" % (self.getX(), self.getY())

    def __str__(self):
        return "(%g, %g)" % (self.getX(), self.getY())

    def __getitem__(self, i):
        """Treat a Point as an array of length 2, [x, y]"""
        if i == 0:
            return self.getX()
        elif i == 1:
            return self.getY()
        else:
            raise IndexError, i

    def __setitem__(self, i, val):
        """Treat a Point as an array of length 2, [x, y]"""
        if i == 0:
            self.setX(val)
        elif i == 1:
            self.setY(val)
        else:
            raise IndexError, i

    def __len__(self):
        return 2
                
    def clone(self):
        return self.__class__(self.getX(), self.getY())
                
    }
}
%enddef

%EXTEND_POINT(double);
%EXTEND_POINT(int);

%extend lsst::afw::image::BBox {
    %pythoncode {
    def __repr__(self):
        return "BBox(PointI(%d, %d), %d, %d)" % (self.getX0(), self.getY0(), self.getWidth(), self.getHeight())

    def __str__(self):
        return "(%d, %d) -- (%d, %d)" % (self.getX0(), self.getY0(), self.getX1(), self.getY1())
    }
}

%apply double &OUTPUT { double & };
%rename(positionToIndexAndResidual) lsst::afw::image::positionToIndex(double &, double);
%clear double &OUTPUT;

%include "lsst/afw/image/ImageUtils.h"

/************************************************************************************************************/

%{
#include "lsst/afw/image/Wcs.h"
%}

SWIG_SHARED_PTR(Wcs, lsst::afw::image::Wcs);

%include "lsst/afw/image/Wcs.h"

%inline {
    /**
     * Create a WCS from crval, image, and the elements of CD
     */
    lsst::afw::image::Wcs::Ptr createWcs(lsst::afw::image::PointD crval,
                                                                lsst::afw::image::PointD crpix,
                                                                double CD11, double CD12, double CD21, double CD22) {

    boost::numeric::ublas::matrix<double> CD(2, 2);
    CD(0, 0) = CD11;
    CD(0, 1) = CD12;
    CD(1, 0) = CD21;
    CD(1, 1) = CD22;
    
    return lsst::afw::image::Wcs::Ptr(new lsst::afw::image::Wcs(crval, crpix, CD));
}
}

/************************************************************************************************************/

%{
#include "lsst/afw/image/Exposure.h"
%}

// Must go Before the %include
%define %exposurePtr(TYPE, PIXEL_TYPE)
SWIG_SHARED_PTR_DERIVED(Exposure##TYPE, lsst::daf::data::LsstBase, lsst::afw::image::Exposure<PIXEL_TYPE>);
%enddef

// Must go After the %include
%define %exposure(TYPE, PIXEL_TYPE)
%template(Exposure##TYPE) lsst::afw::image::Exposure<PIXEL_TYPE>;
%lsst_persistable(lsst::afw::image::Exposure<PIXEL_TYPE>);
%template(makeExposure) lsst::afw::image::makeExposure<lsst::afw::image::MaskedImage<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;
%extend lsst::afw::image::Exposure<PIXEL_TYPE> {
    %pythoncode {
    def Factory(self, *args):
        """Return an Exposure of this type"""
        return Exposure##TYPE(*args)
    }
}
%enddef

%exposurePtr(U, boost::uint16_t);
%exposurePtr(I, int);
%exposurePtr(F, float);
%exposurePtr(D, double);

%include "lsst/afw/image/Exposure.h"

%exposure(U, boost::uint16_t);
%exposure(I, int);
%exposure(F, float);
%exposure(D, double);

