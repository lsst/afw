// -*- lsst-c++ -*-
///////////////////////////////////////////////////////////
//  Image.h
//  Implementation of the Class Image
//  Created on:      09-Feb-2007 15:57:46
//  Original author: Tim Axelrod
///////////////////////////////////////////////////////////

#ifndef LSST_IMAGE_IMAGE_H
#define LSST_IMAGE_IMAGE_H

#include <list>
#include <map>
#include <string>
#include <utility>

#include "boost/mpl/bool.hpp"
#include "boost/shared_ptr.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "lsst/daf/base.h"
#include "lsst/daf/data/LsstBase.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"

namespace lsst {
namespace afw {

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

    template<typename T>
    class Point {
    public:
        Point(T val=0) : _x(val), _y(val) {}
        Point(T x, T y) : _x(x), _y(y) {}
        Point(T const xy[2]) : _x(xy[0]), _y(xy[1]) {}

        T getX() const { return _x; }
        T getY() const { return _y; }
        void setX(T x) { _x = x; }
        void setY(T y) { _y = y; }

        bool operator==(const Point& rhs) const { return (_x == rhs._x && _y == rhs._y); }
        bool operator!=(const Point& rhs) const { return !(*this == rhs); }
        
        Point operator+(const Point& rhs) const { return Point(_x + rhs._x, _y + rhs._y); }
        Point operator-(const Point& rhs) const { return Point(_x - rhs._x, _y - rhs._y); }
#if !defined(SWIG)
        T const& operator[](int const i) const {
            switch (i) {
              case 0: return _x;
              case 1: return _y;
              default: throw lsst::pex::exceptions::OutOfRange(boost::format("Index i == %d must be 0 or 1") % i);
            }
        }
        T& operator[](int const i) {
            return const_cast<T&>((static_cast<const Point&>(*this))[i]); // Meyers, Effective C++, Item 3
        }
#endif
    private:
        T _x, _y;
    };
    typedef Point<double> PointD;
    typedef Point<int> PointI;
    
    class Bbox : private std::pair<PointI, PointI > {
    public:
        Bbox(PointI llc, int width, int height) :
            std::pair<PointI, PointI>(llc, PointI(width, height)) {}
        Bbox(PointI llc, PointI urc) :
            std::pair<PointI, PointI>(llc, urc - llc + 1) {}
        int getX0() const { return first.getX(); }
        int getY0() const { return first.getY(); }
        int getX1() const { return first.getX() + second.getX() - 1; }
        int getY1() const { return first.getY() + second.getY() - 1; }
        int getWidth() const { return second.getX(); }
        int getHeight() const { return second.getY(); }

        bool operator==(const Bbox& rhs) const {
            return
                getX0() == rhs.getX0() && getY0() == rhs.getY0() &&
                getWidth() == rhs.getWidth() && getHeight() == rhs.getHeight();
        }
        bool operator!=(const Bbox& rhs) const {
            return !operator==(rhs);
        }
    };

    /************************************************************************************************************/

    template<typename PixelT>
    struct Reference {
        typedef typename boost::gil::channel_traits<PixelT>::reference type;
    };
    
    template<typename PixelT>
    struct ConstReference {
        typedef typename boost::gil::channel_traits<PixelT>::const_reference type;
    };
    
    template<typename PixelT>
    class ImageBase : public lsst::daf::base::Persistable,
                      public lsst::daf::data::LsstBase {
    private:
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::image_t _image_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::view_t _view_t;
        typedef typename lsst::afw::image::detail::types_traits<PixelT>::const_view_t _const_view_t;

        typedef typename boost::shared_ptr<_image_t> _image_t_Ptr;
    public:
        typedef boost::shared_ptr<ImageBase<PixelT> > Ptr;
        typedef boost::shared_ptr<const ImageBase<PixelT> > ConstPtr;

        typedef detail::basic_tag image_category;

#if !defined(SWIG)
        typedef PixelT PixelConstant;

        struct Pixel {
            typedef PixelT type;
            typedef PixelConstant Constant;
        };
#endif

        typedef typename Reference<PixelT>::type PixelReference;
        typedef typename ConstReference<PixelT>::type PixelConstReference;
        
        typedef typename _view_t::xy_locator xy_locator;
        typedef typename _view_t::xy_locator::const_t const_xy_locator;
        typedef typename _view_t::iterator iterator;
        typedef typename _const_view_t::iterator const_iterator;
        typedef typename _view_t::reverse_iterator reverse_iterator;
        typedef typename _const_view_t::reverse_iterator const_reverse_iterator;
        typedef typename _view_t::x_iterator x_iterator;
        typedef typename _const_view_t::x_iterator const_x_iterator;
        typedef typename _view_t::y_iterator y_iterator;
        typedef typename _const_view_t::y_iterator const_y_iterator;

        template<typename OtherPixelT> friend class ImageBase; // needed by generalised copy constructors
        //
        /// @brief Convert a type to our PixelConstant type
        //
        template<typename PixelConstantT>
        static PixelConstant PixelCast(PixelConstantT rhs) {
            return PixelConstant(rhs);
        }
        //
        // DecoratedImage needs enough access to ImageBase to read data from disk;  we might be able to design around this.
        //
        template<typename> friend class DecoratedImage;
        template<typename, typename, typename> friend class MaskedImage;

        explicit ImageBase(const int width=0, const int height=0);
        explicit ImageBase(const std::pair<int, int> dimensions);
        ImageBase(const ImageBase& src, const bool deep=false);
        ImageBase(const ImageBase& src, const Bbox& bbox, const bool deep=false);
        // generalised copy constructor; defined here in the header so that the compiler can instantiate
        // N(N-1)/2 conversions between N ImageBase types.
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

        ImageBase& operator=(const ImageBase& rhs);
        ImageBase& operator=(const PixelT rhs);
        void operator<<=(const ImageBase& rhs);
        //
        // Operators etc.
        //
        PixelReference operator()(int x, int y);
        PixelConstReference operator()(int x, int y) const;

        int getWidth() const { return _gilView.width(); }
        int getHeight() const { return _gilView.height(); }
        
        int getX0() const { return _x0; }
        int getY0() const { return _y0; }

        const std::pair<int, int> dimensions() const { return std::pair<int, int>(getWidth(), getHeight()); }

        void swap(ImageBase &rhs);
        //
        // Iterators and Locators
        //
        iterator begin() const;
        iterator end() const;
        reverse_iterator rbegin() const;
        reverse_iterator rend() const;
        iterator at(int x, int y) const;

        x_iterator x_at(int x, int y) const;
        x_iterator row_begin(int y) const;
        x_iterator row_end(int y) const;

        y_iterator y_at(int x, int y) const;
        y_iterator col_begin(int x) const;
        y_iterator col_end(int x) const;

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
        Image(const Image& rhs, const Bbox& bbox, const bool deep=false);
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

        DecoratedImage& operator=(const DecoratedImage<PixelT>& image);

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

#endif // LSST_IMAGE_IMAGE_H
