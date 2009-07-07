/**
 * \file
 * \brief Implementation for ImageBase and Image
 */
#include <iostream>
#include "boost/mpl/vector.hpp"
#include "boost/lambda/lambda.hpp"
#include "boost/format.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/gil/gil_all.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace image = lsst::afw::image;

/// Create an uninitialised ImageBase of the specified size
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(int const width, int const height) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(new _image_t(width, height)),
    _gilView(flipped_up_down_view(view(*_gilImage))),
    _ix0(0), _iy0(0), _x0(0), _y0(0)
{
}

/**
 * Create an uninitialised ImageBase of the specified size
 *
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(std::pair<int, int> const dimensions) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(new _image_t(dimensions.first, dimensions.second)),
    _gilView(flipped_up_down_view(view(*_gilImage))),
    _ix0(0), _iy0(0), _x0(0), _y0(0)
{
}

/**
 * Copy constructor.
 *
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this may not be what you want.  See also operator<<=() to copy pixels between Image%s
 */
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(ImageBase const& rhs, ///< Right-hand-side %image
                                    bool const deep       ///< If false, new ImageBase shares storage with rhs; if true
                                                          ///< make a new, standalone, ImageBase
                                   ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(rhs._gilImage), // don't copy the pixels
    _gilView(subimage_view(flipped_up_down_view(view(*_gilImage)),
                           rhs._ix0, rhs._iy0, rhs.getWidth(), rhs.getHeight())),
    _ix0(rhs._ix0), _iy0(rhs._iy0), _x0(rhs._x0), _y0(rhs._y0)
{
    if (deep) {
        ImageBase tmp(getDimensions());
        tmp <<= *this;                  // now copy the pixels
        swap(tmp);
    }
}

/**
 * Copy constructor to make a copy of part of an %image.
 *
 * The BBox is in the @em pixel coordinate system, that is, it ignores X0/Y0
 *
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this is probably what you want 
 */
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(ImageBase const& rhs, ///< Right-hand-side %image
                                    BBox const& bbox,     ///< Specify desired region
                                    bool const deep       ///< If false, new ImageBase shares storage with rhs; if true
                                                          ///< make a new, standalone, ImageBase
                                   ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(rhs._gilImage), // boost::shared_ptr, so don't copy the pixels
    _gilView(subimage_view(rhs._gilView,
                           bbox.getX0(), bbox.getY0(), bbox.getWidth(), bbox.getHeight())),
    _ix0(rhs._ix0 + bbox.getX0()), _iy0(rhs._iy0 + bbox.getY0()),
    _x0(rhs._x0 + bbox.getX0()), _y0(rhs._y0 + bbox.getY0())
{
    if (_ix0 < 0 || _iy0 < 0 || _ix0 + getWidth() > _gilImage->width() || _iy0 + getHeight() > _gilImage->height()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("BBox (%d,%d) %dx%d doesn't fit in image") %
                              bbox.getX0() % bbox.getY0() % bbox.getWidth() % bbox.getHeight()).str());
    }

    if (deep) {
        ImageBase tmp(getDimensions());
        tmp <<= *this;                  // now copy the pixels
        swap(tmp);
    }
}

/// Assignment operator.
///
/// \note that this has the effect of making the lhs share pixels with the rhs which may
/// not be what you intended;  to copy the pixels, use operator<<
///
/// \note this behaviour is required to make the swig interface work, otherwise I'd
/// declare this function private
template<typename PixelT>
image::ImageBase<PixelT>& image::ImageBase<PixelT>::operator=(ImageBase const& rhs) {
    ImageBase tmp(rhs);
    swap(tmp);                          // See Meyers, Effective C++, Item 11
    
    return *this;
}

/// Set the lhs's %pixel values to equal the rhs's
template<typename PixelT>
void image::ImageBase<PixelT>::operator<<=(ImageBase const& rhs) {
    if (getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Dimension mismatch: %dx%d v. %dx%d") %
                              getWidth() % getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    copy_pixels(rhs._gilView, _gilView);
}

/// Return a reference to the pixel <tt>(x, y)</tt>
template<typename PixelT>
typename image::ImageBase<PixelT>::PixelReference image::ImageBase<PixelT>::operator()(int x, int y) {
    return const_cast<typename image::ImageBase<PixelT>::PixelReference>(
	static_cast<typename image::ImageBase<PixelT>::PixelConstReference>(_gilView(x, y)[0])
                       );
}

/// Return a const reference to the pixel <tt>(x, y)</tt>
template<typename PixelT>
typename image::ImageBase<PixelT>::PixelConstReference
	image::ImageBase<PixelT>::operator()(int x, int y) const {
    return _gilView(x, y)[0];
}

template<typename PixelT>
void image::ImageBase<PixelT>::swap(ImageBase &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25
    
    swap(_gilImage, rhs._gilImage);   // just swapping the pointers
    swap(_gilView, rhs._gilView);
    swap(_ix0, rhs._ix0);
    swap(_iy0, rhs._iy0);
    swap(_x0, rhs._x0);
    swap(_y0, rhs._y0);
}

template<typename PixelT>
void image::swap(ImageBase<PixelT>& a, ImageBase<PixelT>& b) {
    a.swap(b);
}
//
// Iterators
//
/// Return an STL compliant iterator to the start of the %image
///
/// Note that this isn't especially efficient; see \link imageIterators\endlink for
/// a discussion
template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::begin() const {
    return _gilView.begin();
}

/// Return an STL compliant iterator to the end of the %image
template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::end() const {
    return _gilView.end();
}

/// Return an STL compliant reverse iterator to the start of the %image
template<typename PixelT>
typename image::ImageBase<PixelT>::reverse_iterator image::ImageBase<PixelT>::rbegin() const {
    return _gilView.rbegin();
}

/// Return an STL compliant reverse iterator to the end of the %image
template<typename PixelT>
typename image::ImageBase<PixelT>::reverse_iterator image::ImageBase<PixelT>::rend() const {
    return _gilView.rend();
}

/// Return an STL compliant iterator at the point <tt>(x, y)</tt>
template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::at(int x, int y) const {
    return _gilView.at(x, y);
}

/// Return a fast STL compliant iterator to the start of the %image which must be contiguous
/// Note that this goes through the image backwards (hence rbegin/rend)
///
/// \exception lsst::pex::exceptions::Runtime
/// Argument \a contiguous is false, or the pixels are not in fact contiguous
template<typename PixelT>
typename image::ImageBase<PixelT>::fast_iterator image::ImageBase<PixelT>::begin(
		bool contiguous         ///< Pixels are contiguous (must be true)
                                                                                ) const {
    if (!contiguous) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Only contiguous == true makes sense");
    }
    if (row_begin(getHeight() - 1) + getWidth()*getHeight() != row_end(0)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Image's pixels are not contiguous");
    }

    return row_begin(getHeight() - 1);
}

/// Return a fast STL compliant iterator to the end of the %image which must be contiguous
/// Note that this goes through the image backwards (hence rbegin/rend)
///
/// \exception lsst::pex::exceptions::Runtime
/// Argument \a contiguous is false, or the pixels are not in fact contiguous
template<typename PixelT>
typename image::ImageBase<PixelT>::fast_iterator image::ImageBase<PixelT>::end(
		bool contiguous         ///< Pixels are contiguous (must be true)
                                                                              ) const {
    if (!contiguous) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Only contiguous == true makes sense"); 
    }
    if (row_begin(getHeight() - 1) + getWidth()*getHeight() != row_end(0)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Image's pixels are not contiguous");
    }

    return row_end(0);
}

/// Return an \c xy_locator at the point <tt>(x, y)</tt> in the %image
///
/// Locators may be used to access a patch in an image
template<typename PixelT>
typename image::ImageBase<PixelT>::xy_locator image::ImageBase<PixelT>::xy_at(int x, int y) const {
    return xy_locator(_gilView.xy_at(x, y));
}

/// Return an \c x_iterator to the start of the \c y'th row
///
/// Incrementing an \c x_iterator moves it across the row
template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::row_begin(int y) const {
    return _gilView.row_begin(y);
}

/// Return an \c x_iterator to the end of the \c y'th row
template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::row_end(int y) const {
    return _gilView.row_end(y);
}

/// Return an \c x_iterator to the point <tt>(x, y)</tt> in the %image
template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::x_at(int x, int y) const {
    return _gilView.x_at(x, y);
}

/// Return an \c y_iterator to the start of the \c y'th row
///
/// Incrementing an \c y_iterator moves it up the column
template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::col_begin(int x) const {
    return _gilView.col_begin(x);
}

/// Return an \c y_iterator to the end of the \c y'th row
template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::col_end(int x) const {
    return _gilView.col_end(x);
}

/// Return an \c y_iterator to the point <tt>(x, y)</tt> in the %image
template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::y_at(int x, int y) const {
    return _gilView.y_at(x, y);
}

/************************************************************************************************************/
/// Set the %image's pixels to rhs
template<typename PixelT>
image::ImageBase<PixelT>& image::ImageBase<PixelT>::operator=(PixelT const rhs) {
    fill_pixels(_gilView, rhs);

    return *this;
}

/************************************************************************************************************/
//
// On to Image itself.  ctors, cctors, and operator=
//
/**
 * Create an uninitialised Image of the specified size 
 *
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
image::Image<PixelT>::Image(int const width, int const height) :
    image::ImageBase<PixelT>(width, height) {}

/**
 * Create an initialised Image of the specified size 
 *
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
image::Image<PixelT>::Image(int const width, ///< Number of columns
                            int const height, ///< Number of rows
                            PixelT initialValue ///< Initial value
                           ) :
    image::ImageBase<PixelT>(width, height) {
    *this = initialValue;
}

/**
 * Create an initialised Image of the specified size
 */
template<typename PixelT>
image::Image<PixelT>::Image(std::pair<int, int> const dimensions ///< (width, height) of the desired Image
                           ) :
    image::ImageBase<PixelT>(dimensions) {}

/**
 * Create an uninitialized Image of the specified size
 */
template<typename PixelT>
image::Image<PixelT>::Image(std::pair<int, int> const dimensions, ///< (width, height) of the desired Image
                            PixelT initialValue ///< Initial value
                           ) :
    image::ImageBase<PixelT>(dimensions) {
    *this = initialValue;
}

/**
 * Copy constructor.
 *
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this may not be what you want.  See also operator<<=() to copy pixels between Image%s
 */
template<typename PixelT>
image::Image<PixelT>::Image(Image const& rhs, ///< Right-hand-side Image
                            bool const deep       ///< If false, new Image shares storage with rhs; if true
                                                  ///< make a new, standalone, ImageBase
                           ) :
    image::ImageBase<PixelT>(rhs, deep) {}

/**
 * Copy constructor to make a copy of part of an Image.
 *
 * The BBox is in the @em pixel coordinate system, that is, it ignores X0/Y0
 *
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this is probably what you want 
 */
template<typename PixelT>
image::Image<PixelT>::Image(Image const& rhs,             ///< Right-hand-side Image
                            BBox const& bbox,             ///< Specify desired region
                            bool const deep               ///< If false, new ImageBase shares storage with rhs; if true
                                                          ///< make a new, standalone, ImageBase
                           ) :
    image::ImageBase<PixelT>(rhs, bbox, deep) {}

/// Set the %image's pixels to rhs
template<typename PixelT>
image::Image<PixelT>& image::Image<PixelT>::operator=(PixelT const rhs) {
    this->ImageBase<PixelT>::operator=(rhs);

    return *this;
}

/// Assignment operator.
///
/// \note that this has the effect of making the lhs share pixels with the rhs which may
/// not be what you intended;  to copy the pixels, use operator<<=
///
/// \note this behaviour is required to make the swig interface work, otherwise I'd
/// declare this function private
template<typename PixelT>
image::Image<PixelT>& image::Image<PixelT>::operator=(Image const& rhs) {
    this->ImageBase<PixelT>::operator=(rhs);
    
    return *this;
}

/************************************************************************************************************/
/**
 * Construct an Image from a FITS file
 *
 * @note We use FITS numbering, so the first HDU is HDU 1, not 0 (although we're nice and interpret 0 meaning
 * the first HDU, i.e. HDU 1).  I.e. if you have a PDU, the numbering is thus [PDU, HDU2, HDU3, ...]
 */
template<typename PixelT>
image::Image<PixelT>::Image(std::string const& fileName, ///< File to read
                            int const hdu,               ///< Desired HDU
                            lsst::daf::base::PropertySet::Ptr metadata, ///< file metadata (may point to NULL)
                            BBox const& bbox                            ///< Only read these pixels
                           ) :
    image::ImageBase<PixelT>() {

    typedef boost::mpl::vector<
        lsst::afw::image::detail::types_traits<unsigned char>::image_t,
        lsst::afw::image::detail::types_traits<unsigned short>::image_t,
        lsst::afw::image::detail::types_traits<short>::image_t,
        lsst::afw::image::detail::types_traits<int>::image_t,
        lsst::afw::image::detail::types_traits<float>::image_t,
        lsst::afw::image::detail::types_traits<double>::image_t
    > fits_img_types;

    if (!boost::filesystem::exists(fileName)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::NotFoundException,
                          (boost::format("File %s doesn't exist") % fileName).str());
    }

    if (!metadata) {
        metadata = lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet);
    }

    if (!image::fits_read_image<fits_img_types>(fileName, *this->_getRawImagePtr(), metadata, hdu, bbox)) {
        throw LSST_EXCEPT(image::FitsException, (boost::format("Failed to read %s HDU %d") % fileName % hdu).str());
    }
    this->_setRawView();

    if (bbox) {
        this->setXY0(bbox.getLLC());
    }
    /*
     * We will interpret one of the header WCSs as providing the (X0, Y0) values
     */
    this->setXY0(this->getXY0() + image::detail::getImageXY0FromMetadata(image::detail::wcsNameForXY0, metadata.get()));
}

/**
 * Write an Image to the specified file
 */
template<typename PixelT>
void image::Image<PixelT>::writeFits(
	std::string const& fileName,    ///< File to write
        lsst::daf::base::PropertySet::Ptr metadata //!< metadata to write to header; or NULL
                                    ) const {
    using lsst::daf::base::PropertySet;

    PropertySet::Ptr wcsAMetadata = image::detail::createTrivialWcsAsPropertySet(image::detail::wcsNameForXY0,
                                                                                 this->getX0(), this->getY0());

    if (metadata) {
        metadata = metadata->deepCopy();
        metadata->combine(wcsAMetadata);
    } else {
        metadata = wcsAMetadata;
    }

    image::fits_write_view(fileName, _getRawView(), metadata);
}

/************************************************************************************************************/

template<typename PixelT>
void image::Image<PixelT>::swap(Image &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25
    ImageBase<PixelT>::swap(rhs);
    ;                                   // no private variables to swap
}

template<typename PixelT>
void image::swap(Image<PixelT>& a, Image<PixelT>& b) {
    a.swap(b);
}

/************************************************************************************************************/
//
// N.b. We could use the STL, but I find boost::lambda clearer, and more easily extended
// to e.g. setting random numbers
//    transform_pixels(_gilView, _gilView, lambda::ret<PixelT>(lambda::_1 + rhs));
// is equivalent to
//    transform_pixels(_gilView, _gilView, std::bind2nd(std::plus<PixelT>(), rhs));
//
namespace bl = boost::lambda;

/// Add scalar rhs to lhs
template<typename PixelT>
void image::Image<PixelT>::operator+=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 + rhs));
}

/// Add Image rhs to lhs
template<typename PixelT>
void image::Image<PixelT>::operator+=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 + bl::_2));
}

/// Add Image c*rhs to lhs
template<typename PixelT>
void image::Image<PixelT>::scaledPlus(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 + bl::ret<PixelT>(c*bl::_2)));
}

/// Subtract scalar rhs from lhs
template<typename PixelT>
void image::Image<PixelT>::operator-=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 - rhs));
}

/// Subtract Image rhs from lhs
template<typename PixelT>
void image::Image<PixelT>::operator-=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 - bl::_2));
}

/// Subtract Image c*rhs from lhs
template<typename PixelT>
void image::Image<PixelT>::scaledMinus(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 - bl::ret<PixelT>(c*bl::_2)));
}

/// Multiply lhs by scalar rhs
template<typename PixelT>
void image::Image<PixelT>::operator*=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 * rhs));
}

/// Multiply lhs by Image rhs (i.e. %pixel-by-%pixel multiplication)
template<typename PixelT>
void image::Image<PixelT>::operator*=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 * bl::_2));
}

/// Multiply lhs by Image c*rhs (i.e. %pixel-by-%pixel multiplication)
template<typename PixelT>
void image::Image<PixelT>::scaledMultiplies(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 * bl::ret<PixelT>(c*bl::_2)));
}

/// Divide lhs by scalar rhs
///
/// \note Floating point types implement this by multiplying by the 1/rhs
template<typename PixelT>
void image::Image<PixelT>::operator/=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 / rhs));
}
//
// Specialize float and double for efficiency
//
namespace lsst { namespace afw { namespace image {
template<>
void Image<double>::operator/=(double const rhs) {
    double const irhs = 1/rhs;
    *this *= irhs;
}

template<>
void Image<float>::operator/=(float const rhs) {
    float const irhs = 1/rhs;
    *this *= irhs;
}
}}}

/// Divide lhs by Image rhs (i.e. %pixel-by-%pixel division)
template<typename PixelT>
void image::Image<PixelT>::operator/=(Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 / bl::_2));
}

/// Divide lhs by Image c*rhs (i.e. %pixel-by-%pixel division)
template<typename PixelT>
void image::Image<PixelT>::scaledDivides(double const c, Image<PixelT> const& rhs) {
    if (this->getDimensions() != rhs.getDimensions()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Images are of different size, %dx%d v %dx%d") %
                           this->getWidth() % this->getHeight() % rhs.getWidth() % rhs.getHeight()).str());
    }
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), bl::ret<PixelT>(bl::_1 / bl::ret<PixelT>(c*bl::_2)));
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(T) \
   template class image::ImageBase<T>; \
   template class image::Image<T>;

INSTANTIATE(boost::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
