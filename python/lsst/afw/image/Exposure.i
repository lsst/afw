// -*- lsst-c++ -*-

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

%include "lsst/afw/image/image_fwd.i"

%{
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/Exposure.h"
%}

%import "lsst/afw/cameraGeom/cameraGeom_fwd.i"

// Must go Before the %include
%define %exposurePtr(PIXEL_TYPE)
%shared_ptr(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%enddef

// Must go After the %include
%define %exposure(TYPE, PIXEL_TYPE)
%newobject makeExposure;
%template(Exposure##TYPE) lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%template(makeExposure) lsst::afw::image::makeExposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
%lsst_persistable(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);
%boost_picklable(lsst::afw::image::Exposure<PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>);

%supportSlicing(lsst::afw::image::Exposure,
                PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel);
%defineClone(Exposure##TYPE, lsst::afw::image::Exposure,
             PIXEL_TYPE, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel);
%enddef

%exposurePtr(boost::uint16_t);
%exposurePtr(boost::uint64_t);
%exposurePtr(int);
%exposurePtr(float);
%exposurePtr(double);

namespace lsst { namespace afw { namespace detection {
    class Psf;
}}}
%shared_ptr(lsst::afw::detection::Psf);
%shared_ptr(lsst::afw::image::ExposureInfo);

%include "lsst/afw/image/ExposureInfo.h"

%include "lsst/afw/image/Exposure.h"

%exposure(U, boost::uint16_t);
%exposure(L, boost::uint64_t);
%exposure(I, int);
%exposure(F, float);
%exposure(D, double);


%extend lsst::afw::image::Exposure<boost::uint16_t, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> {
    %newobject convertF;
    lsst::afw::image::Exposure<float,
         lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> convertF()
    {
        return lsst::afw::image::Exposure<float,
            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(*self, true);
    }
}

%extend lsst::afw::image::Exposure<boost::uint64_t, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> {
    %newobject convertD;
    lsst::afw::image::Exposure<double,
         lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel> convertD()
    {
        return lsst::afw::image::Exposure<double,
            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(*self, true);
    }
}
