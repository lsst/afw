// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class Image
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

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
        struct basic_tag { };
        struct image_tag : basic_tag { };

        template<typename ImageT>
        struct image_traits {
            typedef typename ImageT::image_category image_category;
        };
    }

    /************************************************************************************************************/
    /// \brief A type like std::pair<int, int>, but in lsst::afw::image thus permitting Koenig lookup
    //
    // We want to be able to call operator+= in the global namespace, but define it in lsst::afw::image.
    // To make this possible, at least one of its arguments must be in lsst::afw::image, so we define
    // this type to make the argument lookup ("Koenig Lookup") work smoothly
    //
    struct pair2I : std::pair<int, int> {
        explicit pair2I(int first, int second) : std::pair<int, int>(first, second) {}
    };
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

#if !defined(SWIG)
        /// A single pixel of this type (useful when working with MaskedImage%s)
        typedef PixelT SinglePixel;
        /// A reference to a Pixel (useful when working with MaskedImage%s)
        struct Pixel {
            typedef PixelT type;
        };
#endif
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
        /// A const iterator for traversing the pixels in a row
        typedef typename _const_view_t::x_iterator const_x_iterator;
        /// An iterator for traversing the pixels in a column
        typedef typename _view_t::y_iterator y_iterator;
        /// A const iterator for traversing the pixels in a column
        typedef typename _const_view_t::y_iterator const_y_iterator;
        /** \brief advance an \c xy_locator by \c off
         *
         * Allow users to use pair2I (basically a \c std::pair<int,int>) to manipulate xy_locator%s.  They're
         * declared here for reasons similar to Meyer's item 46 --- we want to make
         * sure that they're instantiated along with the class
         *
         * We don't actually usually use \c std::pair<int,int> but our own struct in namespace lsst::afw::image
         * so as to enable Koenig lookup
         */
        friend xy_locator& operator+=(xy_locator& loc, pair2I const& off) {
            return (loc += boost::gil::point2<std::ptrdiff_t>(off.first, off.second));
        }
        /// \brief advance a const \c xy_locator by \c off
        friend const_xy_locator& operator+=(const_xy_locator& loc, pair2I const& off) {
            return (loc += boost::gil::point2<std::ptrdiff_t>(off.first, off.second));
        }

        /// \brief retreat a \c xy_locator by \c off
        friend xy_locator& operator-=(xy_locator& loc, pair2I const& off) {
            return (loc -= boost::gil::point2<std::ptrdiff_t>(off.first, off.second));
        }
        /// \brief retreat a const \c xy_locator by \c off
        friend const_xy_locator& operator-=(const_xy_locator& loc, pair2I const& off) {
            return (loc -= boost::gil::point2<std::ptrdiff_t>(off.first, off.second));
        }
        //
        template<typename OtherPixelT> friend class ImageBase; // needed by generalised copy constructors
        //
        /// \brief Convert a type to our SinglePixel type
        //
        template<typename SinglePixelT>
        static SinglePixel PixelCast(SinglePixelT rhs) {
            return SinglePixel(rhs);
        }
        //
        // DecoratedImage needs enough access to ImageBase to read data from disk;  we might be able to design around this.
        //
        template<typename> friend class DecoratedImage;
        template<typename, typename, typename> friend class MaskedImage;
        /// Create an uninitialised ImageBase of the specified size
        explicit ImageBase(const int width=0, const int height=0);
        /// Create an uninitialised ImageBase of the specified size
        /// \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
        /// which may be conveniently used to make objects of an appropriate size
        explicit ImageBase(const std::pair<int, int> dimensions);
        /// Copy constructor.
        /// \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
        /// this may not be what you want.  See also operator<< to copy pixels between Image%s
        ImageBase(const ImageBase& src, const bool deep=false);
        /// Copy constructor to make a copy of part of an %image.
        /// \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
        /// this is probably what you want 
        ImageBase(const ImageBase& src, const BBox& bbox, const bool deep=false);
        /// generalised copy constructor; defined here in the header so that the compiler can instantiate
        /// N(N-1)/2 conversions between N ImageBase types.
        template<typename OtherPixelT>
        ImageBase(const ImageBase<OtherPixelT>& rhs, const bool deep) :
            lsst::daf::data::LsstBase(typeid(this)) {
            if (!deep) {
                throw lsst::pex::exceptions::InvalidParameter("Only deep copies are permitted for ImageBases "
                                                              "with different pixel types");
            }

            ImageBase<PixelT> tmp(rhs.dimensions());
            copy_and_convert_pixels(rhs._gilView, tmp._gilView); // from boost::gil
            tmp._x0 = rhs._x0;
            tmp._y0 = rhs._y0;

            using std::swap;                           // See Meyers, Effective C++, Item 25
            ImageBase<PixelT>::swap(tmp);                  // See Meyers, Effective C++, Items 11 and 43
        }

        virtual ~ImageBase() { }
        /// Assignment operator.
        /// \note that this has the effect of making the lhs share pixels with the rhs which may
        /// not be what you intended;  to copy the pixels, use \c operator<<
        /// \note this behaviour is required to make the swig interface work, otherwise I'd
        /// declare this function private
        ImageBase& operator=(const ImageBase& rhs);
        /// Set the %image's pixels to rhs
        ImageBase& operator=(const PixelT rhs);
        /// Set the lhs's %pixel values to equal the rhs's
        void operator<<=(const ImageBase& rhs);
        //
        // Operators etc.
        //
        /// Return a reference to the pixel <tt>(x, y)</tt>
        PixelReference operator()(int x, int y);
        /// Return a const reference to the pixel <tt>(x, y)</tt>
        PixelConstReference operator()(int x, int y) const;

        /// Return the number of columns in the %image
        int getWidth() const { return _gilView.width(); }
        /// Return the number of rows in the %image
        int getHeight() const { return _gilView.height(); }
        /// Return the %image's column-origin
        ///
        /// This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
        /// The origin can be reset with setXY0
        int getX0() const { return _x0; }
        /// Return the %image's row-origin
        ///
        /// This will usually be 0 except for images created using the <tt>ImageBase(ImageBase, BBox)</tt> cctor
        /// The origin can be reset with setXY0
        int getY0() const { return _y0; }
        /// Return the %image's size;  useful for passing to constructors
        const std::pair<int, int> dimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }
        
        void swap(ImageBase &rhs);
        //
        // Iterators and Locators
        //
        /// Return an STL compliant iterator to the start of the %image
        ///
        /// Note that this isn't especially efficient; see \link secPixelAccessTutorial\endlink for
        /// a discussion
        iterator begin() const;
        /// Return an STL compliant iterator to the end of the %image
        iterator end() const;
        /// Return an STL compliant reverse iterator to the start of the %image
        reverse_iterator rbegin() const;
        /// Return an STL compliant reverse iterator to the end of the %image
        reverse_iterator rend() const;
        /// Return an STL compliant iterator at the point <tt>(x, y)</tt>
        iterator at(int x, int y) const;

        /// Return an \c x_iterator to the start of the \c y'th row
        ///
        /// Incrementing an \c x_iterator moves it across the row
        x_iterator row_begin(int y) const;
        /// Return an \c x_iterator to the end of the \c y'th row
        x_iterator row_end(int y) const;
        /// Return an \c x_iterator to the point <tt>(x, y)</tt> in the %image
        x_iterator x_at(int x, int y) const;

        /// Return an \c y_iterator to the start of the \c y'th row
        ///
        /// Incrementing an \c y_iterator moves it up the column
        y_iterator col_begin(int x) const;
        /// Return an \c y_iterator to the end of the \c y'th row
        y_iterator col_end(int x) const;
        /// Return an \c y_iterator to the point <tt>(x, y)</tt> in the %image
        y_iterator y_at(int x, int y) const;

        /// Return an \c xy_locator at the point <tt>(x, y)</tt> in the %image
        ///
        /// Locators may be used to access a patch in an image
        xy_locator xy_at(int x, int y) const;

        //
        // There are use cases (e.g. memory overlays) that may want to set these values, but
        // don't do so unless you are an Expert.
        //
        void setXY0(PointI const origin) {
            _x0 = origin.getX();
            _y0 = origin.getY();
        }

    private:
        //LSST_PERSIST_FORMATTER(lsst::afw::formatters::ImageBaseFormatter<PixelT>);

        _image_t_Ptr _gilImage;
        _view_t _gilView;
        //
        int _x0;                      // origin of ImageBase in some larger image (0 if not a subImageBase)
        int _y0;
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

        typedef detail::image_tag image_category;

#if !defined(SWIG)
        template<typename ImagePT=PixelT>
        struct ImageTypeFactory {
            typedef Image<ImagePT> type;
        };
#endif
        template<typename OtherPixelT> friend class Image; // needed by generalised copy constructors
        
        explicit Image(const int width=0, const int height=0);
        explicit Image(const std::pair<int, int> dimensions);
        Image(const Image& rhs, const bool deep=false);
        Image(const Image& rhs, const BBox& bbox, const bool deep=false);
        Image(std::string const& fileName, const int hdu = 0,
#if 1                                   // Old name for boost::shared_ptrs
              typename lsst::daf::base::DataProperty::PtrType
              metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
              typename lsst::daf::base::DataProperty::Ptr
              metadata=lsst::daf::base::DataProperty::Ptr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
             );

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
#if 1                                   // Old name for boost::shared_ptrs
                       typename lsst::daf::base::DataProperty::PtrType
                       metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
                       typename lsst::daf::base::DataProperty::ConstPtr
                       metadata=lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
                      ) const;
        //
        // Operators etc.
        //
        void operator+=(PixelT const rhs);
        void operator+=(Image<PixelT>const & rhs);
        void operator-=(PixelT const rhs);
        void operator-=(Image<PixelT> const& rhs);
        void operator*=(PixelT const rhs);
        void operator*=(Image<PixelT> const& rhs);
        void operator/=(PixelT const rhs);
        void operator/=(Image<PixelT> const& rhs);
    protected:
        using ImageBase<PixelT>::_getRawView;
    private:
        //LSST_PERSIST_FORMATTER(lsst::afw::formatters::ImageFormatter<PixelT>);
    };
    
    template<typename PixelT>
    void swap(Image<PixelT>& a, Image<PixelT>& b);
    
    /************************************************************************************************************/
    
    template<typename PixelT>
    class DecoratedImage {
    public:
        typedef boost::shared_ptr<DecoratedImage> Ptr;
        typedef boost::shared_ptr<const DecoratedImage> ConstPtr;
        typedef typename Image<PixelT>::Ptr ImagePtr;
        typedef typename Image<PixelT>::ConstPtr ImageConstPtr;

        explicit DecoratedImage(const int width=0, const int height=0);
        explicit DecoratedImage(const std::pair<int, int> dimensions);
        explicit DecoratedImage(typename Image<PixelT>::Ptr rhs);
        DecoratedImage(DecoratedImage const& rhs, const bool deep=false);
        explicit DecoratedImage(std::string const& fileName, const int hdu=0);

        DecoratedImage& operator=(const DecoratedImage& image);

#if 0                                   // use compiler-generated dtor. N.b. not virtual; this isn't a base class
        ~DecoratedImage();
#endif
        
        int getWidth() const { return _image->getWidth(); }
        int getHeight() const { return _image->getHeight(); }
        
        int getX0() const { return _image->getX0(); }
        int getY0() const { return _image->getY0(); }

        const std::pair<int, int> dimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

        void swap(DecoratedImage &rhs);
        
        //void readFits(std::string const& fileName, ...); // replaced by constructor
        void writeFits(std::string const& fileName,
#if 1                                   // Old name for boost::shared_ptrs
                       typename lsst::daf::base::DataProperty::PtrType
                       metadata=lsst::daf::base::DataProperty::PtrType(static_cast<lsst::daf::base::DataProperty *>(0))
#else
                       typename lsst::daf::base::DataProperty::ConstPtr
                       metadata=lsst::daf::base::DataProperty::ConstPtr(static_cast<lsst::daf::base::DataProperty *>(0))
#endif
                      ) const;
        
        ImagePtr      getImage()       { return _image; }
        ImageConstPtr getImage() const { return _image; }
        
#if 1                                   // Old name for boost::shared_ptrs
        lsst::daf::base::DataProperty::PtrType getMetaData() const { return _metaData; }
#else
        lsst::daf::base::DataProperty::Ptr      getMetaData()       { return _metaData; }
        lsst::daf::base::DataProperty::ConstPtr getMetaData() const { return _metaData; }
#endif

        double getGain() const { return _gain; }
        void setGain(double gain) { _gain = gain; }
    private:
        typename Image<PixelT>::Ptr _image;
        lsst::daf::base::DataProperty::PtrType _metaData;
        double _gain;

        void init();
    };

    template<typename PixelT>
    void swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b);
}}}  // lsst::afw::image

#endif
