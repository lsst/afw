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
 
/**
 * \file
 * \brief An Image with associated metadata
 */
#include <iostream>
#include "boost/format.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/make_shared.hpp"
#include "boost/gil/gil_all.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace image {

template<typename PixelT>
void DecoratedImage<PixelT>::init() {
    setMetadata(daf::base::PropertySet::Ptr(new daf::base::PropertyList()));
    _gain = 0;
}

/// Create an %image of the specified size
template<typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(
    geom::Extent2I const & dimensions ///< desired number of columns. rows
) :
    daf::base::Citizen(typeid(this)),
    _image(new Image<PixelT>(dimensions))
{
    init();
}
/**
 * Create an %image of the specified size
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(
    geom::Box2I const & bbox // (width, height) and origin of the desired Image
) :
    daf::base::Citizen(typeid(this)),
    _image(new Image<PixelT>(bbox))
{
    init();
}
/**
 * Create a DecoratedImage wrapping \p rhs
 *
 * Note that this ctor shares pixels with the rhs; it isn't a deep copy
 */
template<typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(
    typename Image<PixelT>::Ptr rhs ///< Image to go into DecoratedImage
) :
    daf::base::Citizen(typeid(this)),
    _image(rhs)
{
    init();
}
/**
 * Copy constructor
 *
 * Note that the lhs will share memory with the rhs unless \p deep is true
 */
template<typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(
    const DecoratedImage& src, ///< right hand side
    const bool deep            ///< Make deep copy?
) :
    daf::base::Citizen(typeid(this)),
    _image(new Image<PixelT>(*src._image, deep)), _gain(src._gain) 
{
    setMetadata(src.getMetadata());
}
/**
 * Assignment operator
 *
 * N.b. this is a shallow assignment; use operator<<=() if you want to copy the pixels
 */
template<typename PixelT>
DecoratedImage<PixelT>& DecoratedImage<PixelT>::operator=(const DecoratedImage& src) {
    DecoratedImage tmp(src);
    swap(tmp);                          // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename PixelT>
void DecoratedImage<PixelT>::swap(DecoratedImage &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25
    
    swap(_image, rhs._image);           // just swapping the pointers
    swap(_gain, rhs._gain);
}

template<typename PixelT>
void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b) {
    a.swap(b);
}

/************************************************************************************************************/
//
// FITS code
//
/**
 * Create a DecoratedImage from a FITS file
 */
template<typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(const std::string& fileName, ///< File to read
                                       int hdu,               ///< The HDU to read
                                       geom::Box2I const& bbox,      ///< Only read these pixels
                                       ImageOrigin const origin     ///< Coordinate system of the bbox
) :
    daf::base::Citizen(typeid(this))
{             ///< HDU within the file
    init();
    _image = boost::make_shared< Image<PixelT> >(fileName, hdu, getMetadata(), bbox, origin);
}

/************************************************************************************************************/

template <typename PixelT>
void DecoratedImage<PixelT>::writeFits(
    afw::fits::Fits & fits,
    CONST_PTR(daf::base::PropertySet) metadata_i
) const {
    PTR(daf::base::PropertySet) metadata;

    if (metadata_i.get()) {
        metadata = getMetadata()->deepCopy();
        metadata->combine(metadata_i);
    } else {
        metadata = getMetadata();
    }
    fits.createImage<PixelT>(this->getWidth(), this->getHeight());
    fits.writeMetadata(*metadata);
    fits.writeImage(this->getImage()->getArray());    
}

template<typename PixelT>
void DecoratedImage<PixelT>::writeFits(
    std::string const& fileName,
    CONST_PTR(daf::base::PropertySet) metadata,
    std::string const& mode
) const {
    using afw::fits::Fits;
    Fits fits(fileName, mode, Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
    writeFits(fits, metadata);
}

template<typename PixelT>
void DecoratedImage<PixelT>::writeFits(
    afw::fits::MemFileManager & manager,
    CONST_PTR(daf::base::PropertySet) metadata,
    std::string const& mode
) const {
    using afw::fits::Fits;
    Fits fits(manager, mode, Fits::AUTO_CLOSE | Fits::AUTO_CHECK);
    writeFits(fits, metadata);
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class DecoratedImage<boost::uint16_t>;
template class DecoratedImage<int>;
template class DecoratedImage<float>;
template class DecoratedImage<double>;
template class DecoratedImage<boost::uint64_t>;

}}} // namespace lsst::afw::image
