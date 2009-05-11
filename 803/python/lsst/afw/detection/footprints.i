// -*- lsst-c++ -*-
%ignore lsst::afw::detection::FootprintFunctor::operator();

%{
#include "lsst/afw/detection/Footprint.h"
%}

SWIG_SHARED_PTR(Peak,      lsst::afw::detection::Peak);
SWIG_SHARED_PTR(Footprint, lsst::afw::detection::Footprint);
SWIG_SHARED_PTR(Span,      lsst::afw::detection::Span);
SWIG_SHARED_PTR(DetectionSetF, lsst::afw::detection::DetectionSet<float, lsst::afw::image::MaskPixel>);
SWIG_SHARED_PTR(DetectionSetD, lsst::afw::detection::DetectionSet<double, lsst::afw::image::MaskPixel>);

%include "lsst/afw/detection/Peak.h"
%include "lsst/afw/detection/Footprint.h"

%template(PeakContainerT)      std::vector<lsst::afw::detection::Peak::Ptr>;
%template(SpanContainerT)      std::vector<lsst::afw::detection::Span::Ptr>;
%template(FootprintContainerT) std::vector<lsst::afw::detection::Footprint::Ptr>;

%define %footprintOperations(NAME, PIXEL_TYPE)
    %template(FootprintFunctor ##NAME) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(FootprintFunctorMI ##NAME)
                       lsst::afw::detection::FootprintFunctor<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
    %template(setImageFromFootprint) lsst::afw::detection::setImageFromFootprint<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(setImageFromFootprintList)
                       lsst::afw::detection::setImageFromFootprintList<lsst::afw::image::Image<PIXEL_TYPE> >
%enddef

%define %DetectionSet(NAME, PIXEL_TYPE)
%template(DetectionSet##NAME) lsst::afw::detection::DetectionSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%template(makeDetectionSet) lsst::afw::detection::makeDetectionSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%enddef


%footprintOperations(F, float);
%template(FootprintFunctorMaskU) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Mask<boost::uint16_t> >;

%DetectionSet(D, double);
%DetectionSet(F, float);

//%template(MaskU) lsst::afw::image::Mask<maskPixelType>;
%template(footprintAndMask) lsst::afw::detection::footprintAndMask<lsst::afw::image::MaskPixel>;
%template(setMaskFromFootprint) lsst::afw::detection::setMaskFromFootprint<lsst::afw::image::MaskPixel>;
%template(setMaskFromFootprintList) lsst::afw::detection::setMaskFromFootprintList<lsst::afw::image::MaskPixel>;

%extend lsst::afw::detection::Span {
    %pythoncode {
    def __str__(self):
        """Print this Span"""
        return self.toString()
    }
}
