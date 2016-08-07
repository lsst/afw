// -*- LSST-C++ -*-

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

#if !defined(LSST_AFW_IMAGE_SLICE_H)
#define LSST_AFW_IMAGE_SLICE_H
/**
 * @file ImageSlice.h
 * @brief Define a single column or row of an Image
 * @ingroup afw
 * @author Steve Bickerton
 *
 * The purpose of this class is to make it possible to add/sub/mult/div an Image by a row or column.
 *    We overload operators + - * / to do this.  The original motivation was to facilitate subtraction
 *    of an overscan region.
 *
 */

#include "lsst/afw/image/Image.h"

namespace lsst {
namespace afw {
namespace image {



/**
 * @class ImageSlice
 * @brief A class to specify a slice of an image
 * @ingroup afw
 *
 */
template<typename PixelT>
class ImageSlice : public Image<PixelT> {
public:
    enum ImageSliceType {ROW, COLUMN};

    explicit ImageSlice(Image<PixelT> const &img);
    ~ImageSlice() {}
    ImageSliceType getImageSliceType() const { return _sliceType; }

private:
    ImageSliceType _sliceType;
};



namespace details {

/*
 * These structs allow the operate() function to be templated over operation.
 * The operate() function handles the loop over pixels and the chosen operator will be used.
 *    The templates allow the compiler to do this efficiently.
 *
 */

template<typename PixelT>
struct Plus     { PixelT operator()(PixelT const imgPix, PixelT const slcPix) { return imgPix + slcPix; } };
template<typename PixelT>
struct Minus    { PixelT operator()(PixelT const imgPix, PixelT const slcPix) { return imgPix - slcPix; } };
template<typename PixelT>
struct Mult     { PixelT operator()(PixelT const imgPix, PixelT const slcPix) { return imgPix * slcPix; } };
template<typename PixelT>
struct Div      { PixelT operator()(PixelT const imgPix, PixelT const slcPix) { return imgPix / slcPix; } };


/**
 * @brief A function to loop over pixels and perform the requested operation
 *
 */
template<typename OperatorT, typename PixelT>
void operate(Image<PixelT> &img, ImageSlice<PixelT> const &slc,
             typename ImageSlice<PixelT>::ImageSliceType sliceType) {

    OperatorT op;

    if (sliceType == ImageSlice<PixelT>::ROW) {
        for (int y = 0; y < img.getHeight(); ++y) {
            typename ImageSlice<PixelT>::x_iterator pSlc = slc.row_begin(0);
            for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
                 pImg != end; ++pImg, ++pSlc) {
                *pImg = op(*pImg, *pSlc);
            }
        }
    } else if (sliceType == ImageSlice<PixelT>::COLUMN) {

        typename ImageSlice<PixelT>::y_iterator pSlc = slc.col_begin(0);
        for (int y = 0; y < img.getHeight(); ++y, ++pSlc) {
            for (typename Image<PixelT>::x_iterator pImg = img.row_begin(y), end = img.row_end(y);
                 pImg != end; ++pImg) {
                *pImg = op(*pImg, *pSlc);
            }
        }
    }

}
}


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// overload +
template<typename PixelT>
typename Image<PixelT>::Ptr operator+(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

template<typename PixelT>
typename Image<PixelT>::Ptr operator+(ImageSlice<PixelT> const &slc, Image<PixelT> const &img);

template<typename PixelT>
void operator+=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// -----------------------------------------------------------------
// overload -
template<typename PixelT>
typename Image<PixelT>::Ptr operator-(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

template<typename PixelT>
void operator-=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// ******************************************************************
// overload *
template<typename PixelT>
typename Image<PixelT>::Ptr operator*(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

template<typename PixelT>
typename Image<PixelT>::Ptr operator*(ImageSlice<PixelT> const &slc, Image<PixelT> const &img);

template<typename PixelT>
void operator*=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// ///////////////////////////////////////////////////////////////////
// overload /
template<typename PixelT>
typename Image<PixelT>::Ptr operator/(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

template<typename PixelT>
void operator/=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


}}}

#endif
