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
/*
 * Define a single column or row of an Image
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
 * A class to specify a slice of an image
 */
template<typename PixelT>
class ImageSlice : public Image<PixelT> {
public:
    enum ImageSliceType {ROW, COLUMN};

    /**
     * Constructor for ImageSlice
     *
     * @param img The image to represent as a slice.
     */
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
 * A function to loop over pixels and perform the requested operation
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
/**
 * Overload operator+()
 *
 * We require two of these, one for image+slice (this one) and one for slice+image (next one down)
 *
 * @param img The Image
 * @param slc The ImageSlice
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator+(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

/**
 * Overload operator+()
 *
 * @param slc The ImageSlice
 * @param img The Image
 *
 * We require two of these, one for image+slice (previous one) and one for slice+image (this)
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator+(ImageSlice<PixelT> const &slc, Image<PixelT> const &img);

/**
 * Overload operator+=()
 *
 * We'll only allow 'image += slice'.  It doesn't make sense to add an image to a slice.
 *
 * @param[in, out] img The Image
 * @param[in] slc The ImageSlice
 */
template<typename PixelT>
void operator+=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// -----------------------------------------------------------------
// overload -
/**
 * Overload operator-()
 *
 * We'll only allow 'image - slice', as 'slice - image' doesn't make sense.
 *
 * @param img The Image
 * @param slc The ImageSlice
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator-(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

/**
 * Overload operator-=()
 *
 * Only 'image -= slice' is defined.  'slice -= image' wouldn't make sense.
 *
 * @param[in, out] img The Image
 * @param[in] slc The ImageSlice
 */
template<typename PixelT>
void operator-=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// ******************************************************************
// overload *
/**
 * Overload operator*()
 *
 * We'll define both 'image*slice' (this one) and 'slice*image' (next one down).
 *
 * @param img The Image
 * @param slc The ImageSlice
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator*(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

/**
 * Overload operator*()
 *
 * We'll define both 'image*slice' (this one) and 'slice*image' (next one down).
 *
 * @param slc The Image
 * @param img The ImageSlice
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator*(ImageSlice<PixelT> const &slc, Image<PixelT> const &img);

/**
 * Overload operator*=()
 *
 * Only 'image *= slice' is defined, as 'slice *= image' doesn't make sense.
 *
 * @param[in, out] img The Image
 * @param[in] slc The ImageSlice
 */
template<typename PixelT>
void operator*=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


// ///////////////////////////////////////////////////////////////////
// overload /
/**
 * Overload operator/()
 *
 * Only 'image / slice' is defined, as 'slice / image' doesn't make sense.
 *
 * @param img The Image
 * @param slc The ImageSlice
 */
template<typename PixelT>
typename Image<PixelT>::Ptr operator/(Image<PixelT> const &img, ImageSlice<PixelT> const &slc);

/**
 * Overload operator/=()
 *
 * Only 'image /= slice' is defined, as 'slice /= image' doesn't make sense.
 *
 * @param[in, out] img The Image
 * @param[in] slc The ImageSlice
 */
template<typename PixelT>
void operator/=(Image<PixelT> &img, ImageSlice<PixelT> const &slc);


}}}

#endif
