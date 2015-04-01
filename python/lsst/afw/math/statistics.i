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

%pythoncode %{
import numpy as np

def castToNumpyArray(vec, defaultType=np.float32):
    """Cast a sequence to a numpy float array, returning the original if it is already a numpy array

    @param[in] vec  sequence of numeric values
    @param[in] defaultType  type of returned numpy array if vec is not already a numpy array
    @return vec cast to a numpy array
    """
    if isinstance(vec, np.ndarray):
        return vec
    # This strange iteration works around a problem in handling VectorU arrays and similar wrappers;
    # [float(val) for val in vec] fails, as does np.array(vec, dtype=float), probably for the same reason.
    # If we stop supporting those vector wrappers then we can let np.array process the array directly.
    return np.array([defaultType(vec[i]) for i in range(len(vec))], dtype=defaultType)
%}

%rename(_makeStatistics1D) lsst::afw::math::makeStatistics1D;
%pythoncode %{
def makeStatistics1D(arr, flags, sctrl=None, weights=None):
    """!Make a Statistics object for the specified 1D array

    @param[in] arr  array to measure; see note
    @param[in] flags  specify what statistics are to be computed
        (e.g. lsst.afw.math.STDEV | lsst.afw.math.MEAN | lsst.afw.math.ERRORS)
    @param[in] weights  array of weights or None; see note;
        if specified, must be the same length as arr;
        if a numpy array then the type must be np.float32
    @param[in] sctrl  specify how statistics are to be computed
        (an lsst.afw.math.StatisticsControl or None)

    @note arr and weights may be any sequence of numbers but numpy arrays are recommended because:
    - the data is not copied (non-numpy-arrays are cast to a numpy array of type float)
    - arr may be any of these data types: float, np.float32, np.float64, int, np.uint16 and np.uint64
        (where np is numpy)
    """
    if sctrl is None:
        sctrl = lsst.afw.math.StatisticsControl()
    arr = castToNumpyArray(arr)
    if weights is None:
        return _makeStatistics1D(arr, flags, sctrl)
    else:
        weights = castToNumpyArray(weights, defaultType=np.float32)
        if weights.dtype != np.float32:
            raise RuntimeError("weights must be of type np.float32 (if a numpy array)")
        return _makeStatistics1D(arr, weights, flags, sctrl)
%}

%include "lsst/afw/math/Statistics.h"

%define %declareStats(PIXTYPE, SUFFIX)
%template(makeStatistics) lsst::afw::math::makeStatistics<PIXTYPE>;
%template(makeStatistics1D) lsst::afw::math::makeStatistics1D<PIXTYPE>;
%template(Statistics ## SUFFIX) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Image<PIXTYPE>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Image<lsst::afw::image::VariancePixel> >;
%enddef

%declareStats(double, D)
%declareStats(float, F)
%declareStats(int, I)
%declareStats(std::uint16_t, U)
%declareStats(std::uint64_t, L)

// We also support Mask<MaskPixel>
%rename(makeStatisticsMU) lsst::afw::math::makeStatistics(lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, int, lsst::afw::math::StatisticsControl const&);
%template(StatisticsMU) lsst::afw::math::Statistics::Statistics<lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel>, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> >;
