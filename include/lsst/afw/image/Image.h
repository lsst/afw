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
 
/**
 * \file
 * \brief Support for 2-D images
 *
 * This file contains the basic 2-d image support for LSST
 */
#ifndef LSST_AFW_IMAGE_IMAGE_H
#define LSST_AFW_IMAGE_IMAGE_H

#include <list>
#include <map>
#include <string>
#include <utility>
#include <functional>

#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math/Function.h"
#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"

namespace lsst { namespace afw {

namespace formatters {
    template <typename PixelT> class ImageFormatter;
    template <typename PixelT> class DecoratedImageFormatter;
}

namespace fits {
class Fits;
class MemFileManager;
}

namespace image {
    namespace detail {
        //
        // Traits for image types
        //
        /// Base %image tag
        struct basic_tag { };
        /// tag for an Image
        struct Image_tag : public basic_tag { };
        /// traits class for image categories
        template<typename ImageT>
        struct image_traits {
            typedef typename ImageT::image_category image_category;
        };
        //
        std::string const wcsNameForXY0 = "A"; // the name of the WCS to use to save (X0, Y0) to FITS files; e.g. "A"
    }

    /*********************************************************************************************************/
    /// A class used to request that array accesses be checked
    class CheckIndices {
    public:
        explicit CheckIndices(bool check=true) : _check(check) {}
        operator bool() const { return _check; }
    private:
        bool _check;
    };

    /*********************************************************************************************************/
    /// \brief metafunction to extract reference type from PixelT
    template<typename PixelT>
    struct Reference {
        typedef typename boost::gil::channel_traits<PixelT>::reference type; ///< reference type
    };
    /// \brief metafunction to extract const reference type from PixelT    
    template<typename PixelT>
    struct ConstReference {
        typedef typename boost::gil::channel_traits<PixelT>::const_reference type; ///< const reference type
    };
    
    enum ImageOrigin {PARENT, LOCAL};

    /// \brief The base class for all %image classed (Image, Mask, MaskedImage, ...)
    //
    // You are not expected to use this class directly in your own code; use one of the
    // specialised subclasses
    //
    template<typename PixelT>
    class ImageBase : public lsst::daf::base::Persistable,
                      public lsst::daf::base::Citizen {
    private:
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::view_t _view_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::const_view_t _const_view_t;


        typedef ndarray::Manager Manager;
    public:        

        typedef boost::shared_ptr<ImageBase<PixelT> > Ptr; ///< A shared_ptr to an ImageBase
        typedef boost::shared_ptr<const ImageBase<PixelT> > ConstPtr; ///< A shared_ptr to a const ImageBase

        typedef detail::basic_tag image_category; ///< trait class to identify type of %image

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
        /// A fast STL compliant iterator for contiguous images
        /// N.b. The order of pixel access is undefined
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

        template<typename OtherPixelT> friend class ImageBase; // needed by generalised copy constructors
        //
        /// \brief Convert a type to our SinglePixel type
        //
        template<typename SinglePixelT>
        static SinglePixel PixelCast(SinglePixelT rhs) {
            return SinglePixel(rhs);
        }
        //
        // DecoratedImage needs enough access to ImageBase to read data from disk; we might be able to design around this
        //
        template<typename> friend class DecoratedImage;
        template<typename, typename, typename> friend class MaskedImage;
        explicit ImageBase(const geom::Extent2I  & dimensions=geom::Extent2I());
        explicit ImageBase(const geom::Box2I &bbox);
        ImageBase(const ImageBase& src, const bool deep=false);
        explicit ImageBase(const ImageBase& src, const geom::Box2I& bbox,
                           const ImageOrigin origin=LOCAL, const bool deep=false);
        /// generalised copy constructor
        ///
        /// defined here in the header so that the compiler can instantiate N(N-1) conversions between N
        /// ImageBase types.
        template<typename OtherPixelT>
        ImageBase(const ImageBase<OtherPixelT>& rhs, const bool deep) :
            lsst::daf::base::Citizen(typeid(this)) {
            if (!deep) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                    "Only deep copies are permitted for ImageBases with different pixel types");
            }

            ImageBase<PixelT> tmp(rhs.getBBox(PARENT));
            copy_and_convert_pixels(rhs._gilView, tmp._gilView); // from boost::gil

            using std::swap;                           // See Meyers, Effective C++, Item 25
            ImageBase<PixelT>::swap(tmp);                  // See Meyers, Effective C++, Items 11 and 43
        }

        explicit ImageBase(
            Array const & array, bool deep = false, geom::Point2I const & xy0 = geom::Point2I()
        );

        virtual ~ImageBase() { }
        ImageBase& operator=(const ImageBase& rhs);
        ImageBase& operator=(const PixelT rhs);
        void operator<<=(const ImageBase& rhs);
        //
        // Operators etc.
        //
        PixelReference operator()(int x, int y);
        PixelReference operator()(int x, int y, CheckIndices const&);
        PixelConstReference operator()(int x, int y) const;
        PixelConstReference operator()(int x, int y, CheckIndices const&) const;

        PixelConstReference get0(int x, int y) const {
            return operator()(x-getX0(), y-getY0());
        }
        PixelConstReference get0(int x, int y, CheckIndices const& check) const {
            return operator()(x-getX0(), y-getY0(), check);
        }
        void set0(int x, int y, const PixelT v) {
            operator()(x-getX0(), y-getY0()) = v;
        }
        void set0(int x, int y, const PixelT v, CheckIndices const& check) {
            operator()(x-getX0(), y-getY0(), check) = v;
        }

        /// Return the number of columns in the %image
        int getWidth() const { return _gilView.width(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _gilView.height(); }
        /**
         * Return the %image's column-origin
         *
         * This will usually be 0 except for images created using the
         * <tt>ImageBase(fileName, hdu, BBox, mode)</tt> ctor or <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with \c setXY0
         */
        int getX0() const { return _origin.getX(); }
        /**
         * Return the %image's row-origin
         *
         * This will usually be 0 except for images created using the
         * <tt>ImageBase(fileName, hdu, BBox, mode)</tt> ctor or <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with \c setXY0
         */
        int getY0() const { return _origin.getY(); }

        /**
         * Return the %image's origin
         *
         * This will usually be (0, 0) except for images created using the
         * <tt>ImageBase(fileName, hdu, BBox, mode)</tt> ctor or <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with \c setXY0
         */
        geom::Point2I getXY0() const { return _origin; }
        
        /**
         * @brief Convert image position to index (nearest integer and fractional parts)
         *
         * @return std::pair(nearest integer index, fractional part)
         */
        std::pair<int, double> positionToIndex(
                double const pos, ///< image position
                lsst::afw::image::xOrY const xy ///< Is this a column or row coordinate?
        ) const {
            double const fullIndex = pos - PixelZeroPos - (xy == X ? getX0() : getY0());
            int const roundedIndex = static_cast<int>(fullIndex + 0.5);
            double const residual = fullIndex - roundedIndex;
            return std::pair<int, double>(roundedIndex, residual);
        }

        /**
         * @brief Convert image index to image position
         *
         * The LSST indexing convention is:
         * * the index of the bottom left pixel is 0,0
         * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
         *
         * @return image position
         */
        inline double indexToPosition(
                double ind, ///< image index
                lsst::afw::image::xOrY const xy ///< Is this a column or row coordinate?
        ) const {
            return ind + PixelZeroPos + (xy == X ? getX0() : getY0());
        }
        
        /// Return the %image's size;  useful for passing to constructors
        geom::Extent2I getDimensions() const { return geom::Extent2I(getWidth(), getHeight()); }
        
        void swap(ImageBase &rhs);

        Array getArray();
        ConstArray getArray() const;
        //
        // Iterators and Locators
        //
        iterator begin() const;
        iterator end() const;
        reverse_iterator rbegin() const;
        reverse_iterator rend() const;
        iterator at(int x, int y) const;

        fast_iterator begin(bool) const;
        fast_iterator end(bool) const;

        /// Return an \c x_iterator to the start of the \c y'th row
        ///
        /// Incrementing an \c x_iterator moves it across the row
        x_iterator row_begin(int y) const {
            return _gilView.row_begin(y);
        }

        /// Return an \c x_iterator to the end of the \c y'th row
        x_iterator row_end(int y) const {
            return _gilView.row_end(y);
        }

        /// Return an \c x_iterator to the point <tt>(x, y)</tt> in the %image
        x_iterator x_at(int x, int y) const { return _gilView.x_at(x, y); }

        /// Return an \c y_iterator to the start of the \c y'th row
        ///
        /// Incrementing an \c y_iterator moves it up the column
        y_iterator col_begin(int x) const {
            return _gilView.col_begin(x);
        }
        
        /// Return an \c y_iterator to the end of the \c y'th row
        y_iterator col_end(int x) const {
            return _gilView.col_end(x);
        }

        /// Return an \c y_iterator to the point <tt>(x, y)</tt> in the %image
        y_iterator y_at(int x, int y) const {
            return _gilView.y_at(x, y);
        }

        /// Return an \c xy_locator at the point <tt>(x, y)</tt> in the %image
        ///
        /// Locators may be used to access a patch in an image
        xy_locator xy_at(int x, int y) const {
            return xy_locator(_gilView.xy_at(x, y));
        }
        /**
         * Set the ImageBase's origin
         *
         * The origin is usually set by the constructor, so you shouldn't need this function
         *
         * \note There are use cases (e.g. memory overlays) that may want to set these values, but
         * don't do so unless you are an Expert.
         */
        void setXY0(geom::Point2I const origin) {
            _origin=origin;
        }
        /**
         * Set the ImageBase's origin
         *
         * The origin is usually set by the constructor, so you shouldn't need this function
         *
         * \note There are use cases (e.g. memory overlays) that may want to set these values, but
         * don't do so unless you are an Expert.
         */
        void setXY0(int const x0, int const y0) {
            setXY0(geom::Point2I(x0,y0));
        }

        geom::Box2I getBBox(ImageOrigin origin=LOCAL) const {
            if (origin == PARENT) {
                return geom::Box2I(_origin, getDimensions());
            }
            else return geom::Box2I(geom::Point2I(0,0), getDimensions());
        }
    private:
        geom::Point2I _origin;
        Manager::Ptr _manager;
        _view_t _gilView;

        //oring of ImageBase in some larger image as returned to and manipulated
        //by the user


    protected:
#if !defined(SWIG)
        static _view_t _allocateView(geom::Extent2I const & dimensions, Manager::Ptr & manager);
        static _view_t _makeSubView(
            geom::Extent2I const & dimensions, 
            geom::Extent2I const & offset, 
            const _view_t & view
        );

        _view_t _getRawView() const { return _gilView; }

#endif
        inline bool isContiguous() const {
            return begin()+getWidth()*getHeight() == end();
        }    
    };

    template<typename PixelT>
    void swap(ImageBase<PixelT>& a, ImageBase<PixelT>& b);

    /************************************************************************************************************/
    /// A class to represent a 2-dimensional array of pixels
    template<typename PixelT>
    class Image : public ImageBase<PixelT> {
    public:
        template<typename, typename, typename> friend class MaskedImage;
        typedef boost::shared_ptr<Image<PixelT> > Ptr;
        typedef boost::shared_ptr<const Image<PixelT> > ConstPtr;

        typedef detail::Image_tag image_category;

#if !defined(SWIG)
        /// A templated class to return this classes' type (present in Image/Mask/MaskedImage)
        template<typename ImagePT=PixelT>
        struct ImageTypeFactory {
            /// Return the desired type
            typedef Image<ImagePT> type;
        };
#endif
        template<typename OtherPixelT> friend class Image; // needed by generalised copy constructors
        
        explicit Image(unsigned int width, unsigned int height, PixelT initialValue=0);
        explicit Image(geom::Extent2I const & dimensions=geom::Extent2I(), PixelT initialValue=0);
        explicit Image(geom::Box2I const & bbox, PixelT initialValue=0);

        explicit Image(Image const & rhs, geom::Box2I const & bbox, ImageOrigin const origin=LOCAL, 
                       const bool deep=false);
        Image(const Image& rhs, const bool deep=false);

        /**
         *  @brief Construct an Image by reading a regular FITS file.
         *
         *  @param[in]      fileName    File to read.
         *  @param[in]      hdu         HDU to read, 1-indexed (i.e. 1=Primary HDU).  The special value
         *                              of 0 reads the Primary HDU unless it is empty, in which case it
         *                              reads the first extension HDU.
         *  @param[in,out]  metadata    Metadata read from the header (may be null).
         *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
         *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
         *                              should take into account the xy0 saved with the image.
         */
        explicit Image(
            std::string const & fileName, int hdu=0,
            PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
            geom::Box2I const & bbox=geom::Box2I(), 
            ImageOrigin origin=LOCAL
        );

        /**
         *  @brief Construct an Image by reading a FITS image in memory.
         *
         *  @param[in]      manager     An object that manages the memory buffer to read.
         *  @param[in]      hdu         HDU to read, 1-indexed (i.e. 1=Primary HDU).  The special value
         *                              of 0 reads the Primary HDU unless it is empty, in which case it
         *                              reads the first extension HDU.
         *  @param[in,out]  metadata    Metadata read from the header (may be null).
         *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
         *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
         *                              should take into account the xy0 saved with the image.
         */
        explicit Image(
            fits::MemFileManager & manager, int hdu=0,
            PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
            geom::Box2I const & bbox=geom::Box2I(), 
            ImageOrigin origin=LOCAL
        );

        /**
         *  @brief Construct an Image from an already-open FITS object.
         *
         *  @param[in]      fitsfile    A FITS object to read from, already at the desired HDU.
         *  @param[in,out]  metadata    Metadata read from the header (may be null).
         *  @param[in]      bbox        If non-empty, read only the pixels within the bounding box.
         *  @param[in]      origin      Coordinate system of the bounding box; if PARENT, the bounding box
         *                              should take into account the xy0 saved with the image.
         */
        explicit Image(
            fits::Fits & fitsfile,
            PTR(lsst::daf::base::PropertySet) metadata=PTR(lsst::daf::base::PropertySet)(),
            geom::Box2I const & bbox=geom::Box2I(), 
            ImageOrigin origin=LOCAL
        );

        // generalised copy constructor
        template<typename OtherPixelT>
        Image(Image<OtherPixelT> const& rhs, const bool deep) :
            image::ImageBase<PixelT>(rhs, deep) {}

        explicit Image(ndarray::Array<PixelT,2,1> const & array, bool deep = false,
                       geom::Point2I const & xy0 = geom::Point2I()) :
            image::ImageBase<PixelT>(array, deep, xy0) {}

        virtual ~Image() { }
        //
        // Assignment operators are not inherited
        //
        Image& operator=(const PixelT rhs);
        Image& operator=(const Image& rhs);

        /**
         *  @brief Write an image to a regular FITS file.
         *
         *  @param[in] fileName      Name of the file to write.
         *  @param[in] metadata      Additional values to write to the header (may be null).
         *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
         */
        void writeFits(
            std::string const& fileName,
            CONST_PTR(lsst::daf::base::PropertySet) metadata = CONST_PTR(lsst::daf::base::PropertySet)(),
            std::string const& mode="w"
        ) const;

        /**
         *  @brief Write an image to a FITS RAM file.
         *
         *  @param[in] manager       Manager object for the memory block to write to.
         *  @param[in] metadata      Additional values to write to the header (may be null).
         *  @param[in] mode          "w"=Create a new file; "a"=Append a new HDU.
         */
        void writeFits(
            fits::MemFileManager & manager,
            CONST_PTR(lsst::daf::base::PropertySet) metadata = CONST_PTR(lsst::daf::base::PropertySet)(),
            std::string const& mode="w"
        ) const;

        /**
         *  @brief Write an image to an open FITS file object.
         *
         *  @param[in] fitsfile      A FITS file already open to the desired HDU.
         *  @param[in] metadata      Additional values to write to the header (may be null).
         */
        void writeFits(
            fits::Fits & fitsfile,
            CONST_PTR(lsst::daf::base::PropertySet) metadata = CONST_PTR(lsst::daf::base::PropertySet)()
        ) const;

        /**
         *  @brief Read an Image from a regular FITS file.
         *
         *  @param[in] filename    Name of the file to read.
         *  @param[in] hdu         Number of the "header-data unit" to read (where 1 is the Primary HDU).
         *                         The default value of 0 is interpreted as "the first HDU with NAXIS != 0".
         */
        static Image readFits(std::string const & filename, int hdu=0) {
            return Image<PixelT>(filename, hdu);
        }

        /**
         *  @brief Read an Image from a FITS RAM file.
         *
         *  @param[in] manager     Object that manages the memory to be read.
         *  @param[in] hdu         Number of the "header-data unit" to read (where 1 is the Primary HDU).
         *                         The default value of 0 is interpreted as "the first HDU with NAXIS != 0".
         */
        static Image readFits(fits::MemFileManager & manager, int hdu=0) {
            return Image<PixelT>(manager, hdu);
        }

        void swap(Image &rhs);
        //
        // Operators etc.
        //
        void operator+=(PixelT const rhs);
        virtual void operator+=(Image<PixelT>const & rhs);
        void operator+=(lsst::afw::math::Function2<double> const& function);
        void scaledPlus(double const c, Image<PixelT>const & rhs);
        void operator-=(PixelT const rhs);
        void operator-=(Image<PixelT> const& rhs);
        void operator-=(lsst::afw::math::Function2<double> const& function);
        void scaledMinus(double const c, Image<PixelT>const & rhs);
        void operator*=(PixelT const rhs);
        void operator*=(Image<PixelT> const& rhs);
        void scaledMultiplies(double const c, Image<PixelT>const & rhs);
        void operator/=(PixelT const rhs);
        void operator/=(Image<PixelT> const& rhs);
        void scaledDivides(double const c, Image<PixelT>const & rhs);

        // In-place per-pixel sqrt().  Useful when handling variance planes.
        void sqrt();
    protected:
        using ImageBase<PixelT>::_getRawView;
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::ImageFormatter<PixelT>)
    };
    
    template<typename LhsPixelT, typename RhsPixelT>
    void operator+=(Image<LhsPixelT> &lhs, Image<RhsPixelT> const& rhs);
    template<typename LhsPixelT, typename RhsPixelT>
    void operator-=(Image<LhsPixelT> &lhs, Image<RhsPixelT> const& rhs);
    template<typename LhsPixelT, typename RhsPixelT>
    void operator*=(Image<LhsPixelT> &lhs, Image<RhsPixelT> const& rhs);
    template<typename LhsPixelT, typename RhsPixelT>
    void operator/=(Image<LhsPixelT> &lhs, Image<RhsPixelT> const& rhs);

    template<typename PixelT>
    void swap(Image<PixelT>& a, Image<PixelT>& b);

/************************************************************************************************************/
    /**
     * \brief A container for an Image and its associated metadata
     */
    template<typename PixelT>
    class DecoratedImage : public lsst::daf::base::Persistable,
                           public lsst::daf::base::Citizen {
    public:
        /// shared_ptr to a DecoratedImage
        typedef boost::shared_ptr<DecoratedImage> Ptr;
        /// shared_ptr to a const DecoratedImage
        typedef boost::shared_ptr<const DecoratedImage> ConstPtr;
        /// shared_ptr to the Image
        typedef PTR(Image<PixelT>) ImagePtr;
        /// shared_ptr to the Image as const
        typedef CONST_PTR(Image<PixelT>) ImageConstPtr;

        explicit DecoratedImage(const geom::Extent2I & dimensions=geom::Extent2I());
        explicit DecoratedImage(const geom::Box2I & bbox);
        explicit DecoratedImage(PTR(Image<PixelT>) rhs);
        DecoratedImage(DecoratedImage const& rhs, const bool deep=false);
        explicit DecoratedImage(
            std::string const& fileName, 
            const int hdu=0, 
            geom::Box2I const& bbox=geom::Box2I(), 
            ImageOrigin const origin = LOCAL
        );

        DecoratedImage& operator=(const DecoratedImage& image);

        PTR(lsst::daf::base::PropertySet) getMetadata() const { return _metadata; }
        void setMetadata(PTR(lsst::daf::base::PropertySet) metadata) { _metadata = metadata; }

        /// Return the number of columns in the %image
        int getWidth() const { return _image->getWidth(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _image->getHeight(); }
        
        /// Return the %image's column-origin
        int getX0() const { return _image->getX0(); }
        /// Return the %image's row-origin
        int getY0() const { return _image->getY0(); }

        /// Return the %image's size;  useful for passing to constructors
        const geom::Extent2I getDimensions() const { return _image->getDimensions(); }

        void swap(DecoratedImage &rhs);
        
        void writeFits(
            std::string const& fileName,
            CONST_PTR(lsst::daf::base::PropertySet) metadata = CONST_PTR(lsst::daf::base::PropertySet)(),
            std::string const& mode="w"
        ) const;

        /// Return a shared_ptr to the DecoratedImage's Image
        ImagePtr      getImage()       { return _image; }
        /// Return a shared_ptr to the DecoratedImage's Image as const
        ImageConstPtr getImage() const { return _image; }

        /**
         * Return the DecoratedImage's gain
         * \note This is mostly just a place holder for other properties that we might
         * want to associate with a DecoratedImage
         */
        double getGain() const { return _gain; }
        /// Set the DecoratedImage's gain
        void setGain(double gain) { _gain = gain; }
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::DecoratedImageFormatter<PixelT>)
        PTR(Image<PixelT>) _image;
        PTR(lsst::daf::base::PropertySet) _metadata;
        
        double _gain;

        void init();
    };

    template<typename PixelT>
    void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b);

}}}  // lsst::afw::image

#endif
