// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2015 AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

%define displayLib_DOCSTRING
"
Basic routines to talk to ds9
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.display", docstring=displayLib_DOCSTRING) displayLib

%{
#include <cstdint>

#   include "lsst/daf/base.h"
#   include "lsst/daf/persistence.h"
#   include "lsst/pex/policy.h"
#   include "lsst/pex/logging.h"
#   include "lsst/afw/geom.h"
#   include "lsst/afw/cameraGeom.h"
#   include "lsst/afw/image.h"

#   include "simpleFits.h"
#   include "Rgb.h"
%}

%include "lsst/p_lsstSwig.i"

%import "lsst/afw/image/imageLib.i"

%lsst_exceptions();

%include "simpleFits.h"

%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<std::uint16_t> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<std::uint64_t> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<int> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<float> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Image<double> >;
%template(writeFitsImage) lsst::afw::display::writeBasicFits<lsst::afw::image::Mask<std::uint16_t> >;

%include "Rgb.h"

%template(replaceSaturatedPixels)
        lsst::afw::display::replaceSaturatedPixels<lsst::afw::image::MaskedImage<float> >;

%template(getZScale) lsst::afw::display::getZScale<std::uint16_t>;
%template(getZScale) lsst::afw::display::getZScale<float>;
