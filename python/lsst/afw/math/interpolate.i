// -*- lsst-C++ -*-
%{
#include "lsst/afw/math/Interpolate.h"
%}

// %ignore lsst::afw::math::Interp::Style;
%include "lsst/afw/math/Interpolate.h"

// %inline %{
// enum {
//     Interp_CONSTANT              = lsst::afw::math::Interp::CONSTANT,
//     Interp_LINEAR                = lsst::afw::math::Interp::LINEAR,
//     Interp_NATURAL_SPLINE        = lsst::afw::math::Interp::NATURAL_SPLINE,
//     Interp_CUBIC_SPLINE          = lsst::afw::math::Interp::CUBIC_SPLINE,
//     Interp_CUBIC_SPLINE_PERIODIC = lsst::afw::math::Interp::CUBIC_SPLINE_PERIODIC,
//     Interp_AKIMA_SPLINE          = lsst::afw::math::Interp::AKIMA_SPLINE,
//     Interp_AKIMA_SPLINE_PERIODIC = lsst::afw::math::Interp::AKIMA_SPLINE_PERIODIC,
// };
// %}
