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
 * Support for 2-D images
 *
 * This file contains the basic 2-d image support for LSST
 */
#ifndef LSST_AFW_IMAGE_IMAGEBASE_H
#define LSST_AFW_IMAGE_IMAGEBASE_H

#include <string>
#include <utility>
#include <functional>

#include <memory>

#include "lsst/geom.h"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math/Function.h"
#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"

namespace lsst {
namespace afw {

namespace fits {
class Fits;
class MemFileManager;
struct ImageWriteOptions;
}

namespace image {
namespace detail {
//
// Traits for image types
//
/// Base %image tag
struct basic_tag {};
/// tag for an Image
struct Image_tag : public basic_tag {};
/// traits class for image categories
template <typename ImageT>
struct image_traits {
    typedef typename ImageT::image_category image_category;
};
//
std::string const wcsNameForXY0 = "A";  // the name of the WCS to use to save (X0, Y0) to FITS files; e.g. "A"
}

/// A class used to request that array accesses be checked
class CheckIndices {
public:
    explicit CheckIndices(bool check = true) : _check(check) {}
    operator bool() const { return _check; }

private:
    bool _check;
};

/// metafunction to extract reference type from PixelT
template <typename PixelT>
struct Reference {
    typedef typename boost::gil::channel_traits<PixelT>::reference type;  ///< reference type
};
/// metafunction to extract const reference type from PixelT
template <typename PixelT>
struct ConstReference {
    typedef typename boost::gil::channel_traits<PixelT>::const_reference type;  ///< const reference type
};

enum ImageOrigin { PARENT, LOCAL };

/// The base class for all %image classed (Image, Mask, MaskedImage, ...)
//
// You are not expected to use this class directly in your own code; use one of the
// specialised subclasses
//
template <typename PixelT>
class ImageBase : public lsst::daf::base::Persistable, public lsst::daf::base::Citizen {
private:
    typedef typename lsst::afw::image::detail::types_traits<PixelT>::view_t _view_t;
    typedef typename lsst::afw::image::detail::types_traits<PixelT>::const_view_t _const_view_t;

    typedef ndarray::Manager Manager;

public:
    typedef detail::basic_tag image_category;  ///< trait class to identify type of %image

    /// A single Pixel of the same type as those in the ImageBase
    typedef PixelT SinglePixel;
    /// A pixel in this ImageBase
    typedef PixelT Pixel;
    /// A Reference to a PixelT
    typedef typename Reference<PixelT>::type PixelReference;
    /// A ConstReference to a PixelT
    typedef typename ConstReference<PixelT>::type PixelConstReference;
    /// An xy_locator
    typedef typename _view_t::xy_locator xy_locator;
    /// A const_xy_locator
    typedef typename _view_t::xy_locator::const_t const_xy_locator;
    /// An STL compliant iterator
    typedef typename _view_t::iterator iterator;
    /// An STL compliant const iterator
    typedef typename _const_view_t::iterator const_iterator;
    /// An STL compliant reverse iterator
    typedef typename _view_t::reverse_iterator reverse_iterator;
    /// An STL compliant const reverse iterator
    typedef typename _const_view_t::reverse_iterator const_reverse_iterator;
    /// An iterator for traversing the pixels in a row
    typedef typename _view_t::x_iterator x_iterator;
    /** A fast STL compliant iterator for contiguous images
     * N.b. The order of pixel access is undefined
     */
    typedef x_iterator fast_iterator;
    /// An iterator for traversing the pixels in a row, created from an xy_locator
    typedef typename _view_t::x_iterator xy_x_iterator;
    /// A const iterator for traversing the pixels in a row
    typedef typename _const_view_t::x_iterator const_x_iterator;
    /// An iterator for traversing the pixels in a column
    typedef typename _view_t::y_iterator y_iterator;
    /// An iterator for traversing the pixels in a row, created from an xy_locator
    typedef typename _view_t::y_iterator xy_y_iterator;
    /// A const iterator for traversing the pixels in a column
    typedef typename _const_view_t::y_iterator const_y_iterator;
    /// A mutable ndarray representation of the image
    typedef typename ndarray::Array<PixelT, 2, 1> Array;
    /// An immutable ndarray representation of the image
    typedef typename ndarray::Array<PixelT const, 2, 1> ConstArray;

    template <typename OtherPixelT>
    friend class ImageBase;  // needed by generalised copy constructors

    /// Convert a type to our SinglePixel type
    template <typename SinglePixelT>
    static SinglePixel PixelCast(SinglePixelT rhs) {
        return SinglePixel(rhs);
    }
    //
    // DecoratedImage needs enough access to ImageBase to read data from disk; we might be able to design
    // around this
    //
    template <typename>
    friend class DecoratedImage;
    template <typename, typename, typename>
    friend class MaskedImage;
    /**
     * Allocator Constructor
     *
     * allocate a new image with the specified dimensions.
     * Sets origin at (0,0)
     */
    explicit ImageBase(const lsst::geom::Extent2I& dimensions = lsst::geom::Extent2I());
    /**
     * Allocator Constructor
     *
     * allocate a new image with the specified dimensions and origin
     */
    explicit ImageBase(const lsst::geom::Box2I& bbox);
    /**
     * Copy constructor.
     *
     * @param src Right-hand-side %image
     * @param deep If false, new ImageBase shares storage with `src`; if true make a new, standalone,
     * ImageBase
     *
     * @note Unless `deep` is `true`, the new %image will share the old %image's pixels;
     * this may not be what you want.  See also assign(rhs) to copy pixels between Image%s
     */
    ImageBase(const ImageBase& src, const bool deep = false);
    ImageBase(ImageBase&& src);
    /**
     * Copy constructor to make a copy of part of an %image.
     *
     * The bbox ignores X0/Y0 if origin == LOCAL, and uses it if origin == PARENT.
     *
     * @param src Right-hand-side %image
     * @param bbox Specify desired region
     * @param origin Specify the coordinate system of the bbox
     * @param deep If false, new ImageBase shares storage with `src`; if true make a new, standalone,
     * ImageBase
     *
     * @note Unless `deep` is `true`, the new %image will share the old %image's pixels;
     * this is probably what you want
     */
    explicit ImageBase(const ImageBase& src, const lsst::geom::Box2I& bbox, const ImageOrigin origin = PARENT,
                       const bool deep = false);
    /**
     * generalised copy constructor
     *
     * defined here in the header so that the compiler can instantiate N(N-1) conversions between N
     * ImageBase types.
     */
    template <typename OtherPixelT>
    ImageBase(const ImageBase<OtherPixelT>& rhs, const bool deep) : lsst::daf::base::Citizen(typeid(this)) {
        if (!deep) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              "Only deep copies are permitted for ImageBases with different pixel types");
        }

        ImageBase<PixelT> tmp(rhs.getBBox());
        copy_and_convert_pixels(rhs._gilView, tmp._gilView);  // from boost::gil

        using std::swap;               // See Meyers, Effective C++, Item 25
        ImageBase<PixelT>::swap(tmp);  // See Meyers, Effective C++, Items 11 and 43
    }

    /**
     *  Construction from ndarray::Array and NumPy.
     *
     *  @note ndarray and NumPy indexes are ordered (y,x), but Image indices are ordered (x,y).
     *
     *  Unless deep is true, the new image will share memory with the array if the the
     *  dimension is contiguous in memory.  If the last dimension is not contiguous, the array
     *  will be deep-copied in Python, but the constructor will fail to compile in pure C++.
     */
    explicit ImageBase(Array const& array, bool deep = false, lsst::geom::Point2I const& xy0 = lsst::geom::Point2I());

    virtual ~ImageBase() = default;
    /** Shallow assignment operator.
     *
     * @note that this has the effect of making the lhs share pixels with the rhs which may
     * not be what you intended;  to copy the pixels, use assign(rhs)
     *
     * @note this behaviour is required to make the swig interface work, otherwise I'd
     * declare this function private
     */
    ImageBase& operator=(const ImageBase& rhs);
    ImageBase& operator=(ImageBase&& rhs);
    /// Set the %image's pixels to rhs
    ImageBase& operator=(const PixelT rhs);
    /**
     * Set the lhs's %pixel values to equal the rhs's
     *
     * @deprecated use assign(rhs) instead
     */
    ImageBase& operator<<=(const ImageBase& rhs);

    /**
     * Copy pixels from another image to a specified subregion of this image.
     *
     * @param[in] rhs  source image whose pixels are to be copied into this image (the destination)
     * @param[in] bbox  subregion of this image to set; if empty (the default) then all pixels are set
     * @param[in] origin  origin of bbox: if PARENT then the lower left pixel of this image is at xy0
     *                    if LOCAL then the lower left pixel of this image is at 0,0
     *
     * @throws lsst::pex::exceptions::LengthError if the dimensions of rhs and the specified subregion of
     * this image do not match.
     */
    void assign(ImageBase const& rhs, lsst::geom::Box2I const& bbox = lsst::geom::Box2I(), ImageOrigin origin = PARENT);
    //
    // Operators etc.
    //
    /// Return a reference to the pixel `(x, y)`
    PixelReference operator()(int x, int y);
    /// Return a reference to the pixel `(x, y)` with bounds checking
    PixelReference operator()(int x, int y, CheckIndices const&);
    /// Return a const reference to the pixel `(x, y)`
    PixelConstReference operator()(int x, int y) const;
    /// Return a const reference to the pixel `(x, y)` with bounds checking
    PixelConstReference operator()(int x, int y, CheckIndices const&) const;

    PixelConstReference get0(int x, int y) const { return operator()(x - getX0(), y - getY0()); }
    PixelConstReference get0(int x, int y, CheckIndices const& check) const {
        return operator()(x - getX0(), y - getY0(), check);
    }
    void set0(int x, int y, const PixelT v) { operator()(x - getX0(), y - getY0()) = v; }
    void set0(int x, int y, const PixelT v, CheckIndices const& check) {
        operator()(x - getX0(), y - getY0(), check) = v;
    }

    /// Return the number of columns in the %image
    int getWidth() const { return _gilView.width(); }
    /// Return the number of rows in the %image
    int getHeight() const { return _gilView.height(); }
    /**
     * Return the %image's column-origin
     *
     * This will usually be 0 except for images created using the
     * `ImageBase(fileName, hdu, BBox, mode)` ctor or `ImageBase(ImageBase, BBox)` cctor
     * The origin can be reset with `setXY0`
     */
    int getX0() const { return _origin.getX(); }
    /**
     * Return the %image's row-origin
     *
     * This will usually be 0 except for images created using the
     * `ImageBase(fileName, hdu, BBox, mode)` ctor or `ImageBase(ImageBase, BBox)` cctor
     * The origin can be reset with `setXY0`
     */
    int getY0() const { return _origin.getY(); }

    /**
     * Return the %image's origin
     *
     * This will usually be (0, 0) except for images created using the
     * `ImageBase(fileName, hdu, BBox, mode)` ctor or `ImageBase(ImageBase, BBox)` cctor
     * The origin can be reset with `setXY0`
     */
    lsst::geom::Point2I getXY0() const { return _origin; }

    /**
     * Convert image position to index (nearest integer and fractional parts)
     *
     * @returns std::pair(nearest integer index, fractional part)
     */
    std::pair<int, double> positionToIndex(
            double const pos,                ///< image position
            lsst::afw::image::xOrY const xy  ///< Is this a column or row coordinate?
            ) const {
        double const fullIndex = pos - PixelZeroPos - (xy == X ? getX0() : getY0());
        int const roundedIndex = static_cast<int>(fullIndex + 0.5);
        double const residual = fullIndex - roundedIndex;
        return std::pair<int, double>(roundedIndex, residual);
    }

    /**
     * Convert image index to image position
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
     *
     * @returns image position
     */
    inline double indexToPosition(double ind,                      ///< image index
                                  lsst::afw::image::xOrY const xy  ///< Is this a column or row coordinate?
                                  ) const {
        return ind + PixelZeroPos + (xy == X ? getX0() : getY0());
    }

    /// Return the %image's size;  useful for passing to constructors
    lsst::geom::Extent2I getDimensions() const { return lsst::geom::Extent2I(getWidth(), getHeight()); }

    void swap(ImageBase& rhs);

    Array getArray();
    ConstArray getArray() const;
    //
    // Iterators and Locators
    //
    /** Return an STL compliant iterator to the start of the %image
     *
     * Note that this isn't especially efficient; see @link imageIterators@endlink for
     * a discussion
     */
    iterator begin() const;
    /// Return an STL compliant iterator to the end of the %image
    iterator end() const;
    /// Return an STL compliant reverse iterator to the start of the %image
    reverse_iterator rbegin() const;
    /// Return an STL compliant reverse iterator to the end of the %image
    reverse_iterator rend() const;
    /// Return an STL compliant iterator at the point `(x, y)`
    iterator at(int x, int y) const;

    /** Return a fast STL compliant iterator to the start of the %image which must be contiguous
     *
     * @param contiguous Pixels are contiguous (must be true)
     *
     * @throws lsst::pex::exceptions::RuntimeError Argument `contiguous` is false, or the pixels are not in
     * fact contiguous
     */
    fast_iterator begin(bool contiguous) const;
    /** Return a fast STL compliant iterator to the end of the %image which must be contiguous
     *
     * @param contiguous Pixels are contiguous (must be true)
     *
     * @throws lsst::pex::exceptions::RuntimeError Argument `contiguous` is false, or the pixels are not in
     * fact contiguous
     */
    fast_iterator end(bool contiguous) const;

    /** Return an `x_iterator` to the start of the `y`'th row
     *
     * Incrementing an `x_iterator` moves it across the row
     */
    x_iterator row_begin(int y) const { return _gilView.row_begin(y); }

    /// Return an `x_iterator` to the end of the `y`'th row
    x_iterator row_end(int y) const { return _gilView.row_end(y); }

    /// Return an `x_iterator` to the point `(x, y)` in the %image
    x_iterator x_at(int x, int y) const { return _gilView.x_at(x, y); }

    /** Return an `y_iterator` to the start of the `y`'th row
     *
     * Incrementing an `y_iterator` moves it up the column
     */
    y_iterator col_begin(int x) const { return _gilView.col_begin(x); }

    /// Return an `y_iterator` to the start of the `y`'th row
    y_iterator col_end(int x) const { return _gilView.col_end(x); }

    /// Return an `y_iterator` to the point `(x, y)` in the %image
    y_iterator y_at(int x, int y) const { return _gilView.y_at(x, y); }

    /** Return an `xy_locator` at the point `(x, y)` in the %image
     *
     * Locators may be used to access a patch in an image
     */
    xy_locator xy_at(int x, int y) const { return xy_locator(_gilView.xy_at(x, y)); }
    /**
     * Set the ImageBase's origin
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * @note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(lsst::geom::Point2I const origin) { _origin = origin; }
    /**
     * Set the ImageBase's origin
     *
     * The origin is usually set by the constructor, so you shouldn't need this function
     *
     * @note There are use cases (e.g. memory overlays) that may want to set these values, but
     * don't do so unless you are an Expert.
     */
    void setXY0(int const x0, int const y0) { setXY0(lsst::geom::Point2I(x0, y0)); }

    lsst::geom::Box2I getBBox(ImageOrigin origin = PARENT) const {
        if (origin == PARENT) {
            return lsst::geom::Box2I(_origin, getDimensions());
        } else
            return lsst::geom::Box2I(lsst::geom::Point2I(0, 0), getDimensions());
    }

private:
    lsst::geom::Point2I _origin;
    Manager::Ptr _manager;
    _view_t _gilView;

    // oring of ImageBase in some larger image as returned to and manipulated
    // by the user

protected:
    static _view_t _allocateView(lsst::geom::Extent2I const& dimensions, Manager::Ptr& manager);
    static _view_t _makeSubView(lsst::geom::Extent2I const& dimensions, lsst::geom::Extent2I const& offset,
                                const _view_t& view);

    _view_t _getRawView() const { return _gilView; }

    inline bool isContiguous() const { return begin() + getWidth() * getHeight() == end(); }
};

template <typename PixelT>
void swap(ImageBase<PixelT>& a, ImageBase<PixelT>& b);


// Inline template definitions

template <typename PixelT>
typename ImageBase<PixelT>::Array ImageBase<PixelT>::getArray() {
    int rowStride = reinterpret_cast<PixelT*>(row_begin(1)) - reinterpret_cast<PixelT*>(row_begin(0));
    return ndarray::external(reinterpret_cast<PixelT*>(row_begin(0)),
                             ndarray::makeVector(getHeight(), getWidth()), ndarray::makeVector(rowStride, 1),
                             this->_manager);
}

template <typename PixelT>
typename ImageBase<PixelT>::ConstArray ImageBase<PixelT>::getArray() const {
    int rowStride = reinterpret_cast<PixelT*>(row_begin(1)) - reinterpret_cast<PixelT*>(row_begin(0));
    return ndarray::external(reinterpret_cast<PixelT*>(row_begin(0)),
                             ndarray::makeVector(getHeight(), getWidth()), ndarray::makeVector(rowStride, 1),
                             this->_manager);
}


}
}
}  // lsst::afw::image

#endif
