// -*- lsst-++ -*-

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

%include "lsst/afw/cameraGeom/cameraGeom_fwd.i"

%{
#include "lsst/afw/cameraGeom/Amp.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/fits.h"
%}

namespace lsst { namespace afw { namespace image {
class Wcs;
}}} // namespace lsst::afw::image
%shared_ptr(lsst::afw::image::Wcs);

%import "lsst/afw/image/image.i"
%import "lsst/afw/image/maskedImage.i"

%include "lsst/afw/cameraGeom/Amp.h"

%inline %{
    lsst::afw::cameraGeom::Amp::Ptr
    cast_Amp(lsst::afw::cameraGeom::Detector::Ptr detector) {
        return boost::shared_dynamic_cast<lsst::afw::cameraGeom::Amp>(detector);
    }
%}

%define Instantiate(PIXEL_TYPE...)
%template(prepareAmpData)
    lsst::afw::cameraGeom::Amp::prepareAmpData<lsst::afw::image::Image<PIXEL_TYPE> >;
%enddef

Instantiate(boost::uint16_t);
Instantiate(float);
Instantiate(double);
%template(prepareAmpData)
    lsst::afw::cameraGeom::Amp::prepareAmpData<lsst::afw::image::Mask<boost::uint16_t> >;

%pythoncode {
class ReadoutCorner(object):
    """A python object corresponding to Amp::ReadoutCorner"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return ["LLC", "LRC", "URC", "ULC"][self.value]
}
