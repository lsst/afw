// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * Implementation for ImageBase and Image
 */
#include <cstdint>
#include <iostream>
#include "boost/mpl/vector.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic pop
#include "boost/format.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/gil/gil_all.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageAlgorithm.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace lsst {
namespace afw {
namespace image {

template <typename PixelT>
typename ImageBase<PixelT>::_view_t ImageBase<PixelT>::_allocateView(geom::Extent2I const& dimensions,
                                                                     Manager::Ptr& manager) {
    if (dimensions.getX() < 0 || dimensions.getY() < 0) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          str(boost::format("Both width and height must be non-negative: %d, %d") %
                              dimensions.getX() % dimensions.getY()));
    }
    if (dimensions.getX() != 0 && dimensions.getY() > std::numeric_limits<int>::max() / dimensions.getX()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          str(boost::format("Image dimensions (%d x %d) too large; int overflow detected.") %
                              dimensions.getX() % dimensions.getY()));
    }
    std::pair<Manager::Ptr, PixelT*> r =
            ndarray::SimpleManager<PixelT>::allocate(dimensions.getX() * dimensions.getY());
    manager = r.first;
    return boost::gil::interleaved_view(dimensions.getX(), dimensions.getY(),
                                        (typename _view_t::value_type*)r.second,
                                        dimensions.getX() * sizeof(PixelT));
}
template <typename PixelT>
typename ImageBase<PixelT>::_view_t ImageBase<PixelT>::_makeSubView(geom::Extent2I const& dimensions,
                                                                    geom::Extent2I const& offset,
                                                                    const _view_t& view) {
    if (offset.getX() < 0 || offset.getY() < 0 || offset.getX() + dimensions.getX() > view.width() ||
        offset.getY() + dimensions.getY() > view.height()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Box2I(Point2I(%d,%d),Extent2I(%d,%d)) doesn't fit in image %dx%d") %
                           offset.getX() % offset.getY() % dimensions.getX() % dimensions.getY() %
                           view.width() % view.height())
                                  .str());
    }
    return boost::gil::subimage_view(view, offset.getX(), offset.getY(), dimensions.getX(),
                                     dimensions.getY());
}

template <typename PixelT>
ImageBase<PixelT>::ImageBase(geom::Extent2I const& dimensions)
        : daf::base::Citizen(typeid(this)),
          _origin(0, 0),
          _manager(),
          _gilView(_allocateView(dimensions, _manager)) {}

template <typename PixelT>
ImageBase<PixelT>::ImageBase(geom::Box2I const& bbox)
        : daf::base::Citizen(typeid(this)),
          _origin(bbox.getMin()),
          _manager(),
          _gilView(_allocateView(bbox.getDimensions(), _manager)) {}

template <typename PixelT>
ImageBase<PixelT>::ImageBase(ImageBase const& rhs, bool const deep

                             )
        : daf::base::Citizen(typeid(this)),
          _origin(rhs._origin),
          _manager(rhs._manager),
          _gilView(rhs._gilView) {
    if (deep) {
        ImageBase tmp(getBBox());
        tmp.assign(*this);  // now copy the pixels
        swap(tmp);
    }
}

template <typename PixelT>
ImageBase<PixelT>::ImageBase(ImageBase const& rhs, geom::Box2I const& bbox, ImageOrigin const origin,
                             bool const deep

                             )
        : daf::base::Citizen(typeid(this)),
          _origin((origin == PARENT) ? bbox.getMin() : rhs._origin + geom::Extent2I(bbox.getMin())),
          _manager(rhs._manager),  // reference counted pointer, don't copy pixels
          _gilView(_makeSubView(bbox.getDimensions(), _origin - rhs._origin, rhs._gilView)) {
    if (deep) {
        ImageBase tmp(getBBox());
        tmp.assign(*this);  // now copy the pixels
        swap(tmp);
    }
}

template <typename PixelT>
ImageBase<PixelT>::ImageBase(Array const& array, bool deep, geom::Point2I const& xy0)
        : daf::base::Citizen(typeid(this)),
          _origin(xy0),
          _manager(array.getManager()),
          _gilView(boost::gil::interleaved_view(array.template getSize<1>(), array.template getSize<0>(),
                                                (typename _view_t::value_type*)array.getData(),
                                                array.template getStride<0>() * sizeof(PixelT))) {
    if (deep) {
        ImageBase tmp(*this, true);
        swap(tmp);
    }
}

template <typename PixelT>
ImageBase<PixelT>& ImageBase<PixelT>::operator=(ImageBase const& rhs) {
    ImageBase tmp(rhs);
    swap(tmp);  // See Meyers, Effective C++, Item 11

    return *this;
}

template <typename PixelT>
ImageBase<PixelT>& ImageBase<PixelT>::operator<<=(ImageBase const& rhs) {
    assign(rhs);
    return *this;
}

template <typename PixelT>
void ImageBase<PixelT>::assign(ImageBase const& rhs, geom::Box2I const& bbox, ImageOrigin origin) {
    auto lhsDim = bbox.isEmpty() ? getDimensions() : bbox.getDimensions();
    if (lhsDim != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Dimension mismatch: %dx%d v. %dx%d") % lhsDim.getX() %
                           lhsDim.getY() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    if (bbox.isEmpty()) {
        copy_pixels(rhs._gilView, _gilView);
    } else {
        auto lhsOff = (origin == PARENT) ? bbox.getMin() - _origin : geom::Extent2I(bbox.getMin());
        auto lhsGilView = _makeSubView(lhsDim, lhsOff, _gilView);
        copy_pixels(rhs._gilView, lhsGilView);
    }
}

template <typename PixelT>
typename ImageBase<PixelT>::PixelReference ImageBase<PixelT>::operator()(int x, int y) {
    return const_cast<typename ImageBase<PixelT>::PixelReference>(
            static_cast<typename ImageBase<PixelT>::PixelConstReference>(_gilView(x, y)[0]));
}

template <typename PixelT>
typename ImageBase<PixelT>::PixelReference ImageBase<PixelT>::operator()(int x, int y,
                                                                         CheckIndices const& check) {
    if (check && (x < 0 || x >= getWidth() || y < 0 || y >= getHeight())) {
        throw LSST_EXCEPT(
                pex::exceptions::LengthError,
                (boost::format("Index (%d, %d) is out of range [0--%d], [0--%d]") % x % y % (getWidth() - 1) %
                 (getHeight() -
                  1)).str());
    }

    return const_cast<typename ImageBase<PixelT>::PixelReference>(
            static_cast<typename ImageBase<PixelT>::PixelConstReference>(_gilView(x, y)[0]));
}

template <typename PixelT>
typename ImageBase<PixelT>::PixelConstReference ImageBase<PixelT>::operator()(int x, int y) const {
    return _gilView(x, y)[0];
}

template <typename PixelT>
typename ImageBase<PixelT>::PixelConstReference ImageBase<PixelT>::operator()(
        int x, int y, CheckIndices const& check) const {
    if (check && (x < 0 || x >= getWidth() || y < 0 || y >= getHeight())) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Index (%d, %d) is out of range [0--%d], [0--%d]") % x % y %
                           (this->getWidth() - 1) %
                           (this->getHeight() -
                            1)).str());
    }

    return _gilView(x, y)[0];
}

template <typename PixelT>
void ImageBase<PixelT>::swap(ImageBase& rhs) {
    using std::swap;  // See Meyers, Effective C++, Item 25

    swap(_manager, rhs._manager);  // just swapping the pointers
    swap(_gilView, rhs._gilView);
    swap(_origin, rhs._origin);
}

template <typename PixelT>
void swap(ImageBase<PixelT>& a, ImageBase<PixelT>& b) {
    a.swap(b);
}

//
// Iterators
//
template <typename PixelT>
typename ImageBase<PixelT>::iterator ImageBase<PixelT>::begin() const {
    return _gilView.begin();
}

template <typename PixelT>
typename ImageBase<PixelT>::iterator ImageBase<PixelT>::end() const {
    return _gilView.end();
}

template <typename PixelT>
typename ImageBase<PixelT>::reverse_iterator ImageBase<PixelT>::rbegin() const {
    return _gilView.rbegin();
}

template <typename PixelT>
typename ImageBase<PixelT>::reverse_iterator ImageBase<PixelT>::rend() const {
    return _gilView.rend();
}

template <typename PixelT>
typename ImageBase<PixelT>::iterator ImageBase<PixelT>::at(int x, int y) const {
    return _gilView.at(x, y);
}

template <typename PixelT>
typename ImageBase<PixelT>::fast_iterator ImageBase<PixelT>::begin(bool contiguous) const {
    if (!contiguous) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Only contiguous == true makes sense");
    }
    if (!this->isContiguous()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Image's pixels are not contiguous");
    }

    return row_begin(0);
}

template <typename PixelT>
typename ImageBase<PixelT>::fast_iterator ImageBase<PixelT>::end(bool contiguous) const {
    if (!contiguous) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Only contiguous == true makes sense");
    }
    if (!this->isContiguous()) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Image's pixels are not contiguous");
    }

    return row_end(getHeight() - 1);
}

template <typename PixelT>
ImageBase<PixelT>& ImageBase<PixelT>::operator=(PixelT const rhs) {
    fill_pixels(_gilView, rhs);

    return *this;
}

//
// On to Image itself.  ctors, cctors, and operator=
//
template <typename PixelT>
Image<PixelT>::Image(unsigned int width, unsigned int height, PixelT initialValue)
        : ImageBase<PixelT>(geom::ExtentI(width, height)) {
    *this = initialValue;
}

template <typename PixelT>
Image<PixelT>::Image(geom::Extent2I const& dimensions, PixelT initialValue) : ImageBase<PixelT>(dimensions) {
    *this = initialValue;
}

template <typename PixelT>
Image<PixelT>::Image(geom::Box2I const& bbox, PixelT initialValue) : ImageBase<PixelT>(bbox) {
    *this = initialValue;
}

template <typename PixelT>
Image<PixelT>::Image(Image const& rhs, bool const deep) : ImageBase<PixelT>(rhs, deep) {}

template <typename PixelT>
Image<PixelT>::Image(Image const& rhs, geom::Box2I const& bbox, ImageOrigin const origin, bool const deep

                     )
        : ImageBase<PixelT>(rhs, bbox, origin, deep) {}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator=(PixelT const rhs) {
    this->ImageBase<PixelT>::operator=(rhs);

    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator=(Image const& rhs) {
    this->ImageBase<PixelT>::operator=(rhs);

    return *this;
}

#ifndef DOXYGEN  // doc for this section has been moved to header

template <typename PixelT>
Image<PixelT>::Image(std::string const& fileName, int hdu, std::shared_ptr<daf::base::PropertySet> metadata,
                     geom::Box2I const& bbox, ImageOrigin origin)
        : ImageBase<PixelT>() {
    fits::Fits fitsfile(fileName, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    try {
        *this = Image(fitsfile, metadata, bbox, origin);
    } catch (fits::FitsError& e) {
        fitsfile.status = 0;                // reset so we can read NAXIS
        if (fitsfile.getImageDim() == 0) {  // no pixels to read
            LSST_EXCEPT_ADD(e, str(boost::format("HDU %d has NAXIS == 0") % hdu));
        }
        throw e;
    }
}
template <typename PixelT>
Image<PixelT>::Image(fits::MemFileManager& manager, int const hdu,
                     std::shared_ptr<daf::base::PropertySet> metadata, geom::Box2I const& bbox,
                     ImageOrigin const origin)
        : ImageBase<PixelT>() {
    fits::Fits fitsfile(manager, "r", fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    fitsfile.setHdu(hdu);
    *this = Image(fitsfile, metadata, bbox, origin);
}

template <typename PixelT>
Image<PixelT>::Image(fits::Fits& fitsfile, std::shared_ptr<daf::base::PropertySet> metadata,
                     geom::Box2I const& bbox, ImageOrigin const origin)
        : ImageBase<PixelT>() {
    typedef boost::mpl::vector<unsigned char, unsigned short, short, int, unsigned int, float, double,
                               std::uint64_t>
            fits_image_types;

    if (!metadata) {
        metadata.reset(new daf::base::PropertyList());
    }

    fits_read_image<fits_image_types>(fitsfile, *this, *metadata, bbox, origin);
}

template <typename PixelT>
void Image<PixelT>::writeFits(std::string const& fileName,
                              std::shared_ptr<daf::base::PropertySet const> metadata_i,
                              std::string const& mode) const {
    fits::Fits fitsfile(fileName, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename PixelT>
void Image<PixelT>::writeFits(fits::MemFileManager& manager,
                              std::shared_ptr<daf::base::PropertySet const> metadata_i,
                              std::string const& mode) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, metadata_i);
}

template <typename PixelT>
void Image<PixelT>::writeFits(fits::Fits& fitsfile,
                              std::shared_ptr<daf::base::PropertySet const> metadata) const {
    fitsfile.writeImage(*this, fits::ImageWriteOptions(*this), metadata);
}

template <typename PixelT>
void Image<PixelT>::writeFits(
    std::string const& filename,
    fits::ImageWriteOptions const& options,
    std::string const& mode,
    std::shared_ptr<daf::base::PropertySet const> header,
    std::shared_ptr<Mask<MaskPixel> const> mask
) const {
    fits::Fits fitsfile(filename, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header, mask);
}

template <typename PixelT>
void Image<PixelT>::writeFits(
    fits::MemFileManager& manager,
    fits::ImageWriteOptions const& options,
    std::string const& mode,
    std::shared_ptr<daf::base::PropertySet const> header,
    std::shared_ptr<Mask<MaskPixel> const> mask
) const {
    fits::Fits fitsfile(manager, mode, fits::Fits::AUTO_CLOSE | fits::Fits::AUTO_CHECK);
    writeFits(fitsfile, options, header, mask);
}

template <typename PixelT>
void Image<PixelT>::writeFits(
    fits::Fits& fitsfile,
    fits::ImageWriteOptions const& options,
    std::shared_ptr<daf::base::PropertySet const> header,
    std::shared_ptr<Mask<MaskPixel> const> mask
) const {
    fitsfile.writeImage(*this, options, header, mask);
}


#endif  // !DOXYGEN

template <typename PixelT>
void Image<PixelT>::swap(Image& rhs) {
    using std::swap;  // See Meyers, Effective C++, Item 25
    ImageBase<PixelT>::swap(rhs);
    ;  // no private variables to swap
}

template <typename PixelT>
void swap(Image<PixelT>& a, Image<PixelT>& b) {
    a.swap(b);
}

// In-place, per-pixel, sqrt().
template <typename PixelT>
void Image<PixelT>::sqrt() {
    transform_pixels(_getRawView(), _getRawView(),
                     [](PixelT const& l) -> PixelT { return static_cast<PixelT>(std::sqrt(l)); });
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator+=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), [&rhs](PixelT const& l) -> PixelT { return l + rhs; });
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator+=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](PixelT const& l, PixelT const& r) -> PixelT { return l + r; });
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator+=(math::Function2<double> const& function) {
    for (int y = 0; y != this->getHeight(); ++y) {
        double const yPos = this->indexToPosition(y, Y);
        double xPos = this->indexToPosition(0, X);
        for (typename Image<PixelT>::x_iterator ptr = this->row_begin(y), end = this->row_end(y); ptr != end;
             ++ptr, ++xPos) {
            *ptr += function(xPos, yPos);
        }
    }
    return *this;
}

template <typename PixelT>
void Image<PixelT>::scaledPlus(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(
            _getRawView(), rhs._getRawView(), _getRawView(),
            [&c](PixelT const& l, PixelT const& r) -> PixelT { return l + static_cast<PixelT>(c * r); });
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator-=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), [&rhs](PixelT const& l) -> PixelT { return l - rhs; });
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator-=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](PixelT const& l, PixelT const& r) -> PixelT { return l - r; });
    return *this;
}

template <typename PixelT>
void Image<PixelT>::scaledMinus(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(
            _getRawView(), rhs._getRawView(), _getRawView(),
            [&c](PixelT const& l, PixelT const& r) -> PixelT { return l - static_cast<PixelT>(c * r); });
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator-=(math::Function2<double> const& function) {
    for (int y = 0; y != this->getHeight(); ++y) {
        double const yPos = this->indexToPosition(y, Y);
        double xPos = this->indexToPosition(0, X);
        for (typename Image<PixelT>::x_iterator ptr = this->row_begin(y), end = this->row_end(y); ptr != end;
             ++ptr, ++xPos) {
            *ptr -= function(xPos, yPos);
        }
    }
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator*=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), [&rhs](PixelT const& l) -> PixelT { return l * rhs; });
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator*=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](PixelT const& l, PixelT const& r) -> PixelT { return l * r; });
    return *this;
}

template <typename PixelT>
void Image<PixelT>::scaledMultiplies(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(
            _getRawView(), rhs._getRawView(), _getRawView(),
            [&c](PixelT const& l, PixelT const& r) -> PixelT { return l * static_cast<PixelT>(c * r); });
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator/=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), [&rhs](PixelT const& l) -> PixelT { return l / rhs; });
    return *this;
}
//
// Specialize float and double for efficiency
//
template <>
Image<double>& Image<double>::operator/=(double const rhs) {
    double const irhs = 1 / rhs;
    *this *= irhs;
    return *this;
}

template <>
Image<float>& Image<float>::operator/=(float const rhs) {
    float const irhs = 1 / rhs;
    *this *= irhs;
    return *this;
}

template <typename PixelT>
Image<PixelT>& Image<PixelT>::operator/=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(),
                     [](PixelT const& l, PixelT const& r) -> PixelT { return l / r; });
    return *this;
}

template <typename PixelT>
void Image<PixelT>::scaledDivides(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(pex::exceptions::LengthError,
                          (boost::format("Images are of different size, %dx%d v %dx%d") % this->getWidth() %
                           this->getHeight() % rhs.getWidth() % rhs.getHeight())
                                  .str());
    }
    transform_pixels(
            _getRawView(), rhs._getRawView(), _getRawView(),
            [&c](PixelT const& l, PixelT const& r) -> PixelT { return l / static_cast<PixelT>(c * r); });
}

namespace {
/*
 * Worker routine for manipulating images;
 */
template <typename LhsPixelT, typename RhsPixelT>
struct plusEq : public pixelOp2<LhsPixelT, RhsPixelT> {
    LhsPixelT operator()(LhsPixelT lhs, RhsPixelT rhs) const override { return static_cast<LhsPixelT>(lhs + rhs); }
};

template <typename LhsPixelT, typename RhsPixelT>
struct minusEq : public pixelOp2<LhsPixelT, RhsPixelT> {
    LhsPixelT operator()(LhsPixelT lhs, RhsPixelT rhs) const override { return static_cast<LhsPixelT>(lhs - rhs); }
};

template <typename LhsPixelT, typename RhsPixelT>
struct timesEq : public pixelOp2<LhsPixelT, RhsPixelT> {
    LhsPixelT operator()(LhsPixelT lhs, RhsPixelT rhs) const override { return static_cast<LhsPixelT>(lhs * rhs); }
};

template <typename LhsPixelT, typename RhsPixelT>
struct divideEq : public pixelOp2<LhsPixelT, RhsPixelT> {
    LhsPixelT operator()(LhsPixelT lhs, RhsPixelT rhs) const override { return static_cast<LhsPixelT>(lhs / rhs); }
};
}

template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator+=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs) {
    for_each_pixel(lhs, rhs, plusEq<LhsPixelT, RhsPixelT>());
    return lhs;
}

template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator-=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs) {
    for_each_pixel(lhs, rhs, minusEq<LhsPixelT, RhsPixelT>());
    return lhs;
}

template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator*=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs) {
    for_each_pixel(lhs, rhs, timesEq<LhsPixelT, RhsPixelT>());
    return lhs;
}

template <typename LhsPixelT, typename RhsPixelT>
Image<LhsPixelT>& operator/=(Image<LhsPixelT>& lhs, Image<RhsPixelT> const& rhs) {
    for_each_pixel(lhs, rhs, divideEq<LhsPixelT, RhsPixelT>());
    return lhs;
}

geom::Box2I bboxFromMetadata(daf::base::PropertySet & metadata)
{
    geom::Extent2I dims;
    if (metadata.exists("ZNAXIS1") && metadata.exists("ZNAXIS2")) {
        dims = geom::Extent2I(metadata.getAsInt("ZNAXIS1"), metadata.getAsInt("ZNAXIS2"));
    } else {
        dims = geom::Extent2I(metadata.getAsInt("NAXIS1"), metadata.getAsInt("NAXIS2"));
    }
    geom::Point2I xy0 = detail::getImageXY0FromMetadata(detail::wcsNameForXY0, &metadata);
    return geom::Box2I(xy0, dims);
}

//
// Explicit instantiations
//
/// @cond
#define INSTANTIATE_OPERATOR(OP_EQ, T)                                                 \
    template Image<T>& operator OP_EQ(Image<T>& lhs, Image<std::uint16_t> const& rhs); \
    template Image<T>& operator OP_EQ(Image<T>& lhs, Image<int> const& rhs);           \
    template Image<T>& operator OP_EQ(Image<T>& lhs, Image<float> const& rhs);         \
    template Image<T>& operator OP_EQ(Image<T>& lhs, Image<double> const& rhs);        \
    template Image<T>& operator OP_EQ(Image<T>& lhs, Image<std::uint64_t> const& rhs);

#define INSTANTIATE(T)           \
    template class ImageBase<T>; \
    template class Image<T>;     \
    INSTANTIATE_OPERATOR(+=, T); \
    INSTANTIATE_OPERATOR(-=, T); \
    INSTANTIATE_OPERATOR(*=, T); \
    INSTANTIATE_OPERATOR(/=, T)

INSTANTIATE(std::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(std::uint64_t);
/// @endcond
}
}
}  // end lsst::afw::image
