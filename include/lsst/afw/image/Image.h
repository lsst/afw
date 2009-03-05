// -*- lsst-c++ -*-
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

#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"

namespace lsst { namespace afw {

namespace formatters {
    template <typename PixelT> class ImageFormatter;
}

namespace image {
    namespace detail {
        //
        // Traits for image types
        //
        /// Base %image tag
        struct basic_tag { };
        /// tag for an Image
        struct Image_tag : basic_tag { };
        /// traits class for image categories
        template<typename ImageT>
        struct image_traits {
            typedef typename ImageT::image_category image_category;
        };
    }

    /************************************************************************************************************/
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
    /// \brief The base class for all %image classed (Image, Mask, MaskedImage, ...)
    //
    // You are not expected to use this class directly in your own code; use one of the
    // specialised subclasses
    //
    template<typename PixelT>
    class ImageBase : public lsst::daf::base::Persistable,
                      public lsst::daf::data::LsstBase {
    private:
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::image_t _image_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::view_t _view_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::const_view_t _const_view_t;

        typedef typename boost::shared_ptr<_image_t> _image_t_Ptr;
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
        explicit ImageBase(const int width=0, const int height=0);
        explicit ImageBase(const std::pair<int, int> dimensions);
        ImageBase(const ImageBase& src, const bool deep=false);
        explicit ImageBase(const ImageBase& src, const BBox& bbox, const bool deep=false);
        /// generalised copy constructor; defined here in the header so that the compiler can instantiate
        /// N(N-1)/2 conversions between N ImageBase types.
        template<typename OtherPixelT>
        ImageBase(const ImageBase<OtherPixelT>& rhs, const bool deep) :
            lsst::daf::data::LsstBase(typeid(this)) {
            if (!deep) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                    "Only deep copies are permitted for ImageBases with different pixel types");
            }

            ImageBase<PixelT> tmp(rhs.getDimensions());
            copy_and_convert_pixels(rhs._gilView, tmp._gilView); // from boost::gil
            tmp._ix0 = rhs._ix0;
            tmp._iy0 = rhs._iy0;
            tmp._x0 = rhs._x0;
            tmp._y0 = rhs._y0;

            using std::swap;                           // See Meyers, Effective C++, Item 25
            ImageBase<PixelT>::swap(tmp);                  // See Meyers, Effective C++, Items 11 and 43
        }

        virtual ~ImageBase() { }
        ImageBase& operator=(const ImageBase& rhs);
        ImageBase& operator=(const PixelT rhs);
        void operator<<=(const ImageBase& rhs);
        //
        // Operators etc.
        //
        PixelReference operator()(int x, int y);
        PixelConstReference operator()(int x, int y) const;

        /// Return the number of columns in the %image
        int getWidth() const { return _gilView.width(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _gilView.height(); }
        /**
         * Return the %image's column-origin
         *
         * This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with \c setXY0
         */
        int getX0() const { return _x0; }
        /**
         * Return the %image's row-origin
         *
         * This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
         * The origin can be reset with \c setXY0
         */
        int getY0() const { return _y0; }
        /// Return the %image's size;  useful for passing to constructors
        const std::pair<int, int> getDimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }
        
        void swap(ImageBase &rhs);
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

        x_iterator row_begin(int y) const;
        x_iterator row_end(int y) const;
        x_iterator x_at(int x, int y) const;

        y_iterator col_begin(int x) const;
        y_iterator col_end(int x) const;
        y_iterator y_at(int x, int y) const;

        xy_locator xy_at(int x, int y) const;
        /**
         * Set the ImageBase's origin
         *
         * The origin is usually set by the constructor, so you shouldn't need this function
         *
         * \note There are use cases (e.g. memory overlays) that may want to set these values, but
         * don't do so unless you are an Expert.
         */
        void setXY0(PointI const origin) {
            _x0 = origin.getX();
            _y0 = origin.getY();
        }

    private:
        _image_t_Ptr _gilImage;
        _view_t _gilView;
        //
        int _ix0;                       // origin of ImageBase in some larger image (0 if not a subImageBase)
        int _iy0;                       // do not lie about this!  You may lie about _[xy]0, but be careful

        int _x0;                        // origin of ImageBase in some larger image (0 if not a subImageBase)
        int _y0;                        // as returned to and manipulated by the user
        //
        // Provide functions that minimise the temptation to get at the variables directly
        //
    protected:
#if !defined(SWIG)
        _image_t_Ptr _getRawImagePtr() { return _gilImage; }
        _view_t _getRawView() const { return _gilView; }
        void _setRawView() {
            _gilView = flipped_up_down_view(view(*_gilImage));
        }
#endif
    };

    template<typename PixelT>
    void swap(ImageBase<PixelT>& a, ImageBase<PixelT>& b);

    /************************************************************************************************************/
    /// A class to represent a 2-dimensional array of pixels
    template<typename PixelT>
    class Image : public ImageBase<PixelT> {
    private:
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::image_t _image_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::view_t _view_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::const_view_t _const_view_t;

        typedef typename boost::shared_ptr<_image_t> _image_t_Ptr;
    public:
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
        
        explicit Image(const int width=0, int const height=0);
        explicit Image(const int width, int const height, PixelT initialValue);
        explicit Image(const std::pair<int, int> dimensions);
        explicit Image(const std::pair<int, int> dimensions, PixelT initialValue);
        Image(const Image& rhs, const bool deep=false);
        explicit Image(const Image& rhs, const BBox& bbox, const bool deep=false);
        explicit Image(std::string const& fileName, const int hdu=0,
                       lsst::daf::base::PropertySet::Ptr metadata=lsst::daf::base::PropertySet::Ptr(),
                       BBox const& bbox=BBox());

        // generalised copy constructor
        template<typename OtherPixelT>
        Image(Image<OtherPixelT> const& rhs, const bool deep) :
            image::ImageBase<PixelT>(rhs, deep) {}

        virtual ~Image() { }
        //
        // Assignment operators are not inherited
        //
        Image& operator=(const PixelT rhs);
        Image& operator=(const Image& rhs);

        //void readFits(std::string const& fileName, ...); // replaced by constructor
        void writeFits(std::string const& fileName,
            lsst::daf::base::PropertySet::Ptr metadata=lsst::daf::base::PropertySet::Ptr()) const;

        void swap(Image &rhs);
        //
        // Operators etc.
        //
        void operator+=(PixelT const rhs);
        void operator+=(Image<PixelT>const & rhs);
        void scaledPlus(double const c, Image<PixelT>const & rhs);
        void operator-=(PixelT const rhs);
        void operator-=(Image<PixelT> const& rhs);
        void scaledMinus(double const c, Image<PixelT>const & rhs);
        void operator*=(PixelT const rhs);
        void operator*=(Image<PixelT> const& rhs);
        void scaledMultiplies(double const c, Image<PixelT>const & rhs);
        void operator/=(PixelT const rhs);
        void operator/=(Image<PixelT> const& rhs);
        void scaledDivides(double const c, Image<PixelT>const & rhs);
    protected:
        using ImageBase<PixelT>::_getRawView;
    private:
        LSST_PERSIST_FORMATTER(lsst::afw::formatters::ImageFormatter<PixelT>);
    };
    
    template<typename PixelT>
    void swap(Image<PixelT>& a, Image<PixelT>& b);
    
    /************************************************************************************************************/
    /**
     * \brief A container for an Image and its associated metadata
     */
    template<typename PixelT>
    class DecoratedImage : public lsst::daf::base::Persistable,
                           public lsst::daf::data::LsstBase {
    public:
        /// shared_ptr to a DecoratedImage
        typedef boost::shared_ptr<DecoratedImage> Ptr;
        /// shared_ptr to a const DecoratedImage
        typedef boost::shared_ptr<const DecoratedImage> ConstPtr;
        /// shared_ptr to the Image
        typedef typename Image<PixelT>::Ptr ImagePtr;
        /// shared_ptr to the Image as const
        typedef typename Image<PixelT>::ConstPtr ImageConstPtr;

        explicit DecoratedImage(const int width=0, const int height=0);
        explicit DecoratedImage(const std::pair<int, int> dimensions);
        explicit DecoratedImage(typename Image<PixelT>::Ptr rhs);
        DecoratedImage(DecoratedImage const& rhs, const bool deep=false);
        explicit DecoratedImage(std::string const& fileName, const int hdu=0);

        DecoratedImage& operator=(const DecoratedImage& image);

        /// Return the number of columns in the %image
        int getWidth() const { return _image->getWidth(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _image->getHeight(); }
        
        /// Return the %image's column-origin
        int getX0() const { return _image->getX0(); }
        /// Return the %image's row-origin
        int getY0() const { return _image->getY0(); }

        /// Return the %image's size;  useful for passing to constructors
        const std::pair<int, int> getDimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

        void swap(DecoratedImage &rhs);
        
        //void readFits(std::string const& fileName, ...); // replaced by constructor
        void writeFits(std::string const& fileName,
            lsst::daf::base::PropertySet::Ptr metadata=lsst::daf::base::PropertySet::Ptr()) const;
        
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
        typename Image<PixelT>::Ptr _image;
        double _gain;

        void init();
    };

    template<typename PixelT>
    void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b);
}}}  // lsst::afw::image

#endif
