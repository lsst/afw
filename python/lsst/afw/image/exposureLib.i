%{
#include "lsst/afw/image/Exposure.h"
%}

%include "lsst/afw/image/Exposure.h"

%include "lsst/daf/persistenceMacros.i"

%template(ExposureU)    lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::maskPixelType>;
%template(ExposureF)    lsst::afw::image::Exposure<float, lsst::afw::image::maskPixelType>;
%template(ExposureD)    lsst::afw::image::Exposure<double, lsst::afw::image::maskPixelType>;
%lsst_persistable_shared_ptr(ExposureUPtr, lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureFPtr, lsst::afw::image::Exposure<float, lsst::afw::image::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureDPtr, lsst::afw::image::Exposure<double, lsst::afw::image::maskPixelType>)
