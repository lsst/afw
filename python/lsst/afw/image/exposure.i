%{
#include "lsst/fw/Exposure.h"
%}

%include "lsst/fw/Exposure.h"

%include "lsst/mwi/persistenceMacros.i"

%template(ExposureU)    lsst::fw::Exposure<boost::uint16_t, lsst::fw::maskPixelType>;
%template(ExposureF)    lsst::fw::Exposure<float, lsst::fw::maskPixelType>;
%template(ExposureD)    lsst::fw::Exposure<double, lsst::fw::maskPixelType>;
%lsst_persistable_shared_ptr(ExposureUPtr, lsst::fw::Exposure<boost::uint16_t, lsst::fw::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureFPtr, lsst::fw::Exposure<float, lsst::fw::maskPixelType>)
%lsst_persistable_shared_ptr(ExposureDPtr, lsst::fw::Exposure<double, lsst::fw::maskPixelType>)
