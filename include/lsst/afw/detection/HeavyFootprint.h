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
 
#if !defined(LSST_DETECTION_HEAVY_FOOTPRINT_H)
#define LSST_DETECTION_HEAVY_FOOTPRINT_H
/**
 * \file
 * \brief Represent a set of pixels of an arbitrary shape and size,
 *        including values for those pixels; a HeavyFootprint is a
 *        Footprint that also not only a description of a region, but
 *        values within that region.
 */
#include <algorithm>
#include <list>
#include <cmath>
#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include "lsst/afw/detection/Footprint.h"

namespace lsst {
namespace afw { 
namespace detection {

class HeavyFootprintCtrl;

/*!
 * \brief A set of pixels in an Image, including those pixels' actual values
 */
template <typename ImagePixelT, typename MaskPixelT=lsst::afw::image::MaskPixel,
          typename VariancePixelT=lsst::afw::image::VariancePixel>
class HeavyFootprint :
    public afw::table::io::PersistableFacade< HeavyFootprint<ImagePixelT,MaskPixelT,VariancePixelT> >,
    public Footprint
{
public:

    explicit HeavyFootprint(
        Footprint const& foot,
        lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const& mimage,
        HeavyFootprintCtrl const* ctrl=NULL
                           );

    explicit HeavyFootprint(Footprint const& foot,
                            HeavyFootprintCtrl const* ctrl=NULL);

    virtual bool isHeavy() const { return true; }

    void insert(lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> & mimage) const;
    void insert(lsst::afw::image::Image<ImagePixelT> & image) const;

    ndarray::Array<ImagePixelT,1,1>     getImageArray() { return _image; }
    ndarray::Array<MaskPixelT,1,1>      getMaskArray() { return _mask; }
    ndarray::Array<VariancePixelT,1,1>  getVarianceArray() { return _variance; }

    ndarray::Array<ImagePixelT const,1,1>     getImageArray() const { return _image; }
    ndarray::Array<MaskPixelT const,1,1>      getMaskArray() const { return _mask; }
    ndarray::Array<VariancePixelT const,1,1>  getVarianceArray() const { return _variance; }

    /* Returns the OR of all the mask pixels held in this HeavyFootprint. */
    MaskPixelT getMaskBitsSet() const {
        MaskPixelT maskbits = 0;
        for (typename ndarray::Array<MaskPixelT,1,1>::Iterator i = _mask.begin(); i != _mask.end(); ++i) {
            maskbits |= *i;
        }
        return maskbits;
    }

protected:

    class Factory;  // factory class used for persistence, public only so we can instantiate it in .cc file

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:
    HeavyFootprint() {}  // private constructor, only used for persistence.

    ndarray::Array<ImagePixelT, 1, 1> _image;
    ndarray::Array<MaskPixelT, 1, 1> _mask;
    ndarray::Array<VariancePixelT, 1, 1> _variance;
};

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> makeHeavyFootprint(
    Footprint const& foot,
    lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const& img,
    HeavyFootprintCtrl const* ctrl=NULL
                                                                          )
{
    return HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>(foot, img, ctrl);
}

template <typename ImagePixelT, typename MaskPixelT, typename VariancePixelT>
boost::shared_ptr<HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> >
mergeHeavyFootprints(
    HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> const& h1,
    HeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT> const& h2
);

}}}

#endif
