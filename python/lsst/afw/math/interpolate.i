// -*- lsst-C++ -*-
%{
#include "lsst/afw/math/Interpolate.h"
%}

//%ignore lsst::afw::math::Interpolate::Style;
%include "lsst/afw/math/Interpolate.h"

 //%inline %{
 //enum {
 //   Interpolate_CONSTANT              = Interpolate::CONSTANT,
 //   Interpolate_LINEAR                = Interpolate::LINEAR,
 //   Interpolate_NATURAL_SPLINE        = Interpolate::NATURAL_SPLINE,
 //   Interpolate_CUBIC_SPLINE          = Interpolate::CUBIC_SPLINE,
 //   Interpolate_CUBIC_SPLINE_PERIODIC = Interpolate::CUBIC_SPLINE_PERIODIC,
 //   Interpolate_AKIMA_SPLINE          = Interpolate::AKIMA_SPLINE,
 //   Interpolate_AKIMA_SPLINE_PERIODIC = Interpolate::AKIMA_SPLINE_PERIODIC,
 //};
 //%}
