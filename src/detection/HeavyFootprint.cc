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
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"

namespace lsst {
namespace afw {
namespace detection {
namespace {
    template<typename T>
    struct setPixel {
        setPixel(T val) : _val(val) {}

        T operator()(T) const {
            return _val;
        }
    private:
        T _val;
    };

    template<>
    struct setPixel<boost::uint16_t> {
        typedef boost::uint16_t T;

        setPixel(T val) : _mask(~val) {}

        T operator()(T pix) const {
            pix &= _mask;
            return pix;
        }
    private:
        T _mask;
    };
}

/**
 * Create a HeavyFootprint from a regular Footprint and the image that provides the pixel values
 *
 * \note: the HeavyFootprintCtrl is passed by const* not const& so that we needn't provide a definition
 * in Footprint.h
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::HeavyFootprint(
    Footprint const& foot,              ///< The Footprint defining the pixels to set
    lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const& mimage, ///< The pixel values
    HeavyFootprintCtrl const *ctrl     ///< Control how we manipulate HeavyFootprints
        ) : Footprint(foot),
            _image(ndarray::allocate(ndarray::makeVector(foot.getNpix()))),
            _mask(ndarray::allocate(ndarray::makeVector(foot.getNpix()))),
            _variance(ndarray::allocate(ndarray::makeVector(foot.getNpix())))
{
    HeavyFootprintCtrl ctrl_s = HeavyFootprintCtrl();

    if (!ctrl) {
        ctrl = &ctrl_s;
    }

    switch (ctrl->getModifySource()) {
      case HeavyFootprintCtrl::NONE:
        flattenArray(*this, mimage.getImage()->getArray(),    _image,    mimage.getXY0());
        flattenArray(*this, mimage.getMask()->getArray(),     _mask,     mimage.getXY0());
        flattenArray(*this, mimage.getVariance()->getArray(), _variance, mimage.getXY0());
        break;
      case HeavyFootprintCtrl::SET:
        {
        ImagePixelT const ival = ctrl->getImageVal();
        MaskPixelT const mval = ctrl->getMaskVal();
        VariancePixelT const vval = ctrl->getVarianceVal();

        flattenArray(*this, mimage.getImage()->getArray(),    _image,
            setPixel<ImagePixelT>(ival), mimage.getXY0());
        flattenArray(*this, mimage.getMask()->getArray(),     _mask,
            setPixel<MaskPixelT>(mval), mimage.getXY0());
        flattenArray(*this, mimage.getVariance()->getArray(), _variance,
            setPixel<VariancePixelT>(vval), mimage.getXY0());
        break;
        }
    }
}

/**
 * Create a HeavyFootprint from a regular Footprint, allocating space
 * to hold foot.getArea() pixels, but not initializing them.  This is
 * used when unpersisting a HeavyFootprint.
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::HeavyFootprint(
    Footprint const& foot,              ///< The Footprint defining the pixels to set
    HeavyFootprintCtrl const* ctrl)
    : Footprint(foot),
      _image   (ndarray::allocate(ndarray::makeVector(foot.getNpix()))),
      _mask    (ndarray::allocate(ndarray::makeVector(foot.getNpix()))),
      _variance(ndarray::allocate(ndarray::makeVector(foot.getNpix())))
{
}

/**
 * Replace all the pixels in the image with the values in the HeavyFootprint
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::insert(
        lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> & mimage ///< Image to set
                                                                    ) const
{
    expandArray(*this, _image,    mimage.getImage()->getArray(),    mimage.getXY0());
    expandArray(*this, _mask,     mimage.getMask()->getArray(),     mimage.getXY0());
    expandArray(*this, _variance, mimage.getVariance()->getArray(), mimage.getXY0());
}

/**
 * Replace all the pixels in the image with the values in the HeavyFootprint
 */
template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
void HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>::insert(
                                                                     lsst::afw::image::Image<ImagePixelT> & image ///< Image to set
                                                                     ) const
{
    expandArray(*this, _image,    image.getArray(),    image.getXY0());
}


/**
 Sums the two given HeavyFootprints *h1* and *h2*, returning a
 HeavyFootprint with the union footprint, and summed pixels where they
 overlap.
 */
template<typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
PTR(HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT>)
mergeHeavyFootprints(HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> const& h1,
                     HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> const& h2)
{
    // Merge the Footprints (by merging the Spans)
    Footprint foot(h1);
    Footprint::SpanList spans = h2.getSpans();
    for (Footprint::SpanList::iterator sp = spans.begin();
         sp != spans.end(); ++sp) {
        foot.addSpan(**sp);
    }
    foot.normalize();

    // Find the union bounding-box
    geom::Box2I bbox(h1.getBBox());
    bbox.include(h2.getBBox());

    // Create union-bb-sized images and insert the heavies
    image::MaskedImage<ImagePixelT,MaskPixelT,VariancePixelT> im1(bbox);
    image::MaskedImage<ImagePixelT,MaskPixelT,VariancePixelT> im2(bbox);
    h1.insert(im1);
    h2.insert(im2);
    // Add the pixels
    im1 += im2;

    // Build new HeavyFootprint from the merged spans and summed pixels.
    return PTR(HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT>)(
        new HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT>(foot, im1));
}


/************************************************************************************************************/
//
// Explicit instantiations
// \cond
//
//
#define INSTANTIATE(TYPE) \
    template class HeavyFootprint<TYPE>; \
    template PTR(HeavyFootprint<TYPE>) mergeHeavyFootprints<TYPE>( \
        HeavyFootprint<TYPE> const&, HeavyFootprint<TYPE> const&);

INSTANTIATE(boost::uint16_t);
INSTANTIATE(double);
INSTANTIATE(float);
INSTANTIATE(int);

}}}
// \endcond
