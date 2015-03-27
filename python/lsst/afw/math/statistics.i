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
 
%{
#include "lsst/afw/math/Statistics.h"
%}

%shared_ptr(lsst::afw::math::StatisticsControl);

%include "lsst/afw/math/Statistics.h"


%define %declareStats(PIXTYPE, SUFFIX)
%template(makeStatistics) lsst::afw::math::makeStatistics<PIXTYPE>;
%template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Image<lsst::afw::image::VariancePixel> >;
%enddef

%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)
%declareStats(uint16_t, U)
%declareStats(uint64_t, L)

// We also support Mask<MaskPixel>
%rename(makeStatisticsMU) lsst::afw::math::makeStatistics(lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, int, lsst::afw::math::StatisticsControl const&);
%template(StatisticsMU) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;

