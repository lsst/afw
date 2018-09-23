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

/*
 * An Image with associated metadata
 */
#include <cstdint>
#include <iostream>

#include "boost/format.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/gil/gil_all.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace lsst {
namespace afw {
namespace image {

template <typename PixelT>
void DecoratedImage<PixelT>::init() {
    // safer to initialize a smart pointer as a named variable
    std::shared_ptr<daf::base::PropertySet> metadata(new daf::base::PropertyList);
    setMetadata(metadata);
    _gain = 0;
}

template <typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(lsst::geom::Extent2I const& dimensions)
        : daf::base::Citizen(typeid(this)), _image(new Image<PixelT>(dimensions)) {
    init();
}
template <typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(lsst::geom::Box2I const& bbox)
        : daf::base::Citizen(typeid(this)), _image(new Image<PixelT>(bbox)) {
    init();
}
template <typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(std::shared_ptr<Image<PixelT>> rhs)
        : daf::base::Citizen(typeid(this)), _image(rhs) {
    init();
}
template <typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(const DecoratedImage& src, const bool deep)
        : daf::base::Citizen(typeid(this)), _image(new Image<PixelT>(*src._image, deep)), _gain(src._gain) {
    setMetadata(src.getMetadata());
}
template <typename PixelT>
DecoratedImage<PixelT>& DecoratedImage<PixelT>::operator=(const DecoratedImage& src) {
    DecoratedImage tmp(src);
    swap(tmp);  // See Meyers, Effective C++, Item 11

    return *this;
}

template <typename PixelT>
void DecoratedImage<PixelT>::swap(DecoratedImage& rhs) {
    using std::swap;  // See Meyers, Effective C++, Item 25

    swap(_image, rhs._image);  // just swapping the pointers
    swap(_gain, rhs._gain);
}

template <typename PixelT>
void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b) {
    a.swap(b);
}

//
// FITS code
//
template <typename PixelT>
DecoratedImage<PixelT>::DecoratedImage(const std::string& fileName, const int hdu,
                                       lsst::geom::Box2I const& bbox, ImageOrigin const origin)
        : daf::base::Citizen(typeid(this)) {
    init();
    _image = std::shared_ptr<Image<PixelT>>(new Image<PixelT>(fileName, hdu, getMetadata(), bbox, origin));
}

template <typename PixelT>
void DecoratedImage<PixelT>::writeFits(std::string const& fileName,
                                       std::shared_ptr<daf::base::PropertySet const> metadata,
                                       std::string const& mode) const {
    fits::ImageWriteOptions const options;
    writeFits(fileName, options, metadata, mode);
}

template <typename PixelT>
void DecoratedImage<PixelT>::writeFits(std::string const& fileName, fits::ImageWriteOptions const& options,
                                       std::shared_ptr<daf::base::PropertySet const> metadata_i,
                                       std::string const& mode) const {
    std::shared_ptr<daf::base::PropertySet> metadata;

    if (metadata_i) {
        metadata = getMetadata()->deepCopy();
        metadata->combine(metadata_i);
    } else {
        metadata = getMetadata();
    }

    getImage()->writeFits(fileName, options, mode, metadata);
}

//
// Explicit instantiations
//
template class DecoratedImage<std::uint16_t>;
template class DecoratedImage<int>;
template class DecoratedImage<float>;
template class DecoratedImage<double>;
template class DecoratedImage<std::int64_t>;
}  // namespace image
}  // namespace afw
}  // namespace lsst
