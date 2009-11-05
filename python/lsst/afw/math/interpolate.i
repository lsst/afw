# -*- lsst-C++ -*-
%{
#include "lsst/afw/math/Interpolate.h"
%}

%ignore lsst::afw::math::Interp::Style;
%include "lsst/afw/math/Interpolate.h"

%inline %{
namespace Interp = lsst::afw::math::Interp;
enum {
    Interp_CONSTANT              = Interp::CONSTANT,
    Interp_LINEAR                = Interp::LINEAR,
    Interp_NATURAL_SPLINE        = Interp::NATURAL_SPLINE,
    Interp_CUBIC_SPLINE          = Interp::CUBIC_SPLINE,
    Interp_CUBIC_SPLINE_PERIODIC = Interp::CUBIC_SPLINE_PERIODIC,
    Interp_AKIMA_SPLINE          = Interp::AKIMA_SPLINE,
    Interp_AKIMA_SPLINE_PERIODIC = Interp::AKIMA_SPLINE_PERIODIC,
};
%}
