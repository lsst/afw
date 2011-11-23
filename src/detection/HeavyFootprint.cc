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
 
/*****************************************************************************/
/** \file
 *
 * \brief HeavyFootprint and associated classes
 */
#include <cassert>
#include <string>
#include <typeinfo>
#include <algorithm>
#include "boost/format.hpp"
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"

namespace lsst {
namespace afw {
namespace detection {

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::HeavyFootprint(
    Footprint const& foot,
    lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const& mimage
        ) : Footprint(foot),
            _image(lsst::ndarray::allocate(lsst::ndarray::makeVector(foot.getNpix()))),
            _mask(lsst::ndarray::allocate(lsst::ndarray::makeVector(foot.getNpix()))),
            _variance(lsst::ndarray::allocate(lsst::ndarray::makeVector(foot.getNpix())))
{
    flattenArray(*this, mimage.getImage()->getArray(),    _image,    mimage.getXY0());
    flattenArray(*this, mimage.getMask()->getArray(),     _mask,     mimage.getXY0());
    flattenArray(*this, mimage.getVariance()->getArray(), _variance, mimage.getXY0());
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::insert(
        lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> & mimage,
        lsst::afw::geom::Point2I const & origin) const
{
    expandArray(*this, _image,    mimage.getImage()->getArray(),    origin);
    expandArray(*this, _mask,     mimage.getMask()->getArray(),     origin);
    expandArray(*this, _variance, mimage.getVariance()->getArray(), origin);
}

/************************************************************************************************************/
//
// Explicit instantiations
// \cond
//
//
#define INSTANTIATE(TYPE) \
    template class HeavyFootprint<TYPE>;

INSTANTIATE(float);
INSTANTIATE(int);

}}}
// \endcond
