%{
#include "lsst/fw/Exposure.h"
%}

%include "lsst/fw/Exposure.h"

%template(ExposureU)    lsst::fw::Exposure<boost::uint16_t, lsst::fw::maskPixelType>;
%template(ExposureF)    lsst::fw::Exposure<float, lsst::fw::maskPixelType>;
%template(ExposureD)    lsst::fw::Exposure<double, lsst::fw::maskPixelType>;
