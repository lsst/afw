
// %{
// #include "lsst/afw/math/warpExposure.h"
// %}
// 
// %include "lsst/afw/math/warpExposure.h"
// 
// %define %warpExp(DESTIMAGEPIXELT, SRCIMAGEPIXELT)
// %template(warpExposure) lsst::afw::math::warpExposure<
//     lsst::afw::exposure<DESTIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
//     lsst::afw::exposure<SRCIMAGEPIXELT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> >;    
// %enddef
// 
// %warpExp(float, int);
// %warpExp(double, int);
// %warpExp(float, boost::uint16_t);
// %warpExp(double, boost::uint16_t);
// %warpExp(float, float);
// %warpExp(double, float);
// %warpExp(double, double);
