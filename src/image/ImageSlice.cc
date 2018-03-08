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

/*
 * Provide functions to operate on rows/columns of images
 */
#include <vector>
#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/ImageSlice.h"

namespace ex = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace image {

template <typename PixelT>
ImageSlice<PixelT>::ImageSlice(image::Image<PixelT> const &img) : Image<PixelT>(img), _sliceType(ROW) {
    // verify the img is a slice (row or column)
    if (img.getWidth() != 1 && img.getHeight() != 1) {
        throw LSST_EXCEPT(ex::OutOfRangeError, "Input image must be a slice (width or height == 1)");
    } else if (img.getWidth() == 1 && img.getHeight() == 1) {
        throw LSST_EXCEPT(ex::InvalidParameterError,
                          "1x1 image ambiguous (could be row or column).  "
                          "Perhaps a constant would be better than a slice? ");
    } else if (img.getWidth() == 1 && img.getHeight() != 1) {
        _sliceType = COLUMN;
    } else if (img.getHeight() == 1 && img.getWidth() != 1) {
        _sliceType = ROW;
    }

    // what about 1xn images where a 1x1 row slice is desired? ... use a constant instead of a slice
    // what about nx1 images wehre a 1x1 column slice is desired? ... use a constant instead of a slice
}

/* ************************************************************************ *
 *
 * column operators
 *
 * ************************************************************************ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// overload +

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator+(Image<PixelT> const &img, ImageSlice<PixelT> const &slc) {
    std::shared_ptr<Image<PixelT>> retImg(new Image<PixelT>(img, true));
    *retImg += slc;
    return retImg;
}

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator+(ImageSlice<PixelT> const &slc, Image<PixelT> const &img) {
    return operator+(img, slc);
}

template <typename PixelT>
void operator+=(Image<PixelT> &img, ImageSlice<PixelT> const &slc) {
    details::operate<details::Plus<PixelT>>(img, slc, slc.getImageSliceType());
}

// -----------------------------------------------------------------
// overload -

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator-(Image<PixelT> const &img, ImageSlice<PixelT> const &slc) {
    std::shared_ptr<Image<PixelT>> retImg(new Image<PixelT>(img, true));
    *retImg -= slc;
    return retImg;
}

template <typename PixelT>
void operator-=(Image<PixelT> &img, ImageSlice<PixelT> const &slc) {
    details::operate<details::Minus<PixelT>>(img, slc, slc.getImageSliceType());
}

// ******************************************************************
// overload *

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator*(Image<PixelT> const &img, ImageSlice<PixelT> const &slc) {
    std::shared_ptr<Image<PixelT>> retImg(new Image<PixelT>(img, true));
    *retImg *= slc;
    return retImg;
}

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator*(ImageSlice<PixelT> const &slc, Image<PixelT> const &img) {
    return operator*(img, slc);
}

template <typename PixelT>
void operator*=(Image<PixelT> &img, ImageSlice<PixelT> const &slc) {
    details::operate<details::Mult<PixelT>>(img, slc, slc.getImageSliceType());
}

// overload

template <typename PixelT>
std::shared_ptr<Image<PixelT>> operator/(Image<PixelT> const &img, ImageSlice<PixelT> const &slc) {
    std::shared_ptr<Image<PixelT>> retImg(new Image<PixelT>(img, true));
    *retImg /= slc;
    return retImg;
}

template <typename PixelT>
void operator/=(Image<PixelT> &img, ImageSlice<PixelT> const &slc) {
    details::operate<details::Div<PixelT>>(img, slc, slc.getImageSliceType());
}

/*
 * Explicit Instantiations
 *
 */
/// @cond
#define INSTANTIATE_SLICE_OP_SYM(TYPE, OP)                                                                  \
    template std::shared_ptr<Image<TYPE>> operator OP(Image<TYPE> const &img, ImageSlice<TYPE> const &slc); \
    template std::shared_ptr<Image<TYPE>> operator OP(ImageSlice<TYPE> const &slc, Image<TYPE> const &img)

#define INSTANTIATE_SLICE_OP_ASYM(TYPE, OP) \
    template std::shared_ptr<Image<TYPE>> operator OP(Image<TYPE> const &img, ImageSlice<TYPE> const &slc)

#define INSTANTIATE_SLICE_OPEQ(TYPE, OP) \
    template void operator OP(Image<TYPE> &img, ImageSlice<TYPE> const &slc)

#define INSTANTIATE_SLICES(TYPE)                                     \
    template ImageSlice<TYPE>::ImageSlice(Image<TYPE> const &image); \
    INSTANTIATE_SLICE_OP_SYM(TYPE, +);                               \
    INSTANTIATE_SLICE_OP_ASYM(TYPE, -);                              \
    INSTANTIATE_SLICE_OP_SYM(TYPE, *);                               \
    INSTANTIATE_SLICE_OP_ASYM(TYPE, /);                              \
    INSTANTIATE_SLICE_OPEQ(TYPE, +=);                                \
    INSTANTIATE_SLICE_OPEQ(TYPE, -=);                                \
    INSTANTIATE_SLICE_OPEQ(TYPE, *=);                                \
    INSTANTIATE_SLICE_OPEQ(TYPE, /=)

INSTANTIATE_SLICES(double);
INSTANTIATE_SLICES(float);
/// @endcond
}  // namespace image
}  // namespace afw
}  // namespace lsst
