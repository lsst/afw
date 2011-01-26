// -*- lsst-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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
