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
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"

namespace image = lsst::afw::image;

/// Create an uninitialised ImageBase of the specified size
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(int const width, int const height) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(new _image_t(width, height)),
    _gilView(flipped_up_down_view(view(*_gilImage))),
    _x0(0), _y0(0)
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
    _x0(0), _y0(0)
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
                                    bool const deep       ///< If true, new ImageBase shares storage with rhs; if false
                                                          ///< make a new, standalone, ImageBase
                                   ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(rhs._gilImage), // don't copy the pixels
    _gilView(subimage_view(flipped_up_down_view(view(*_gilImage)),
                            rhs._x0, rhs._y0, rhs.getWidth(), rhs.getHeight())),
    _x0(rhs._x0),
    _y0(rhs._y0)
{
    if (deep) {
        ImageBase tmp(getDimensions());
        tmp <<= *this;                  // now copy the pixels
        swap(tmp);
    }
}

/**
 * Copy constructor to make a copy of part of an %image.
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this is probably what you want 
 */
template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(ImageBase const& rhs, ///< Right-hand-side %image
                                    BBox const& bbox,     ///< Specify desired region
                                    bool const deep       ///< If true, new ImageBase shares storage with rhs; if false
                                                          ///< make a new, standalone, ImageBase
                                   ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(rhs._gilImage), // boost::shared_ptr, so don't copy the pixels
    _gilView(subimage_view(rhs._gilView,
                           bbox.getX0(), bbox.getY0(), bbox.getWidth(), bbox.getHeight())),
    _x0(rhs._x0 + bbox.getX0()), _y0(rhs._y0 + bbox.getY0())
{
    if (_x0 < 0 || _y0 < 0 || _x0 + getWidth() > _gilImage->width() || _y0 + getHeight() > _gilImage->height()) {
        throw lsst::pex::exceptions::LengthError(boost::format("BBox (%d,%d) %dx%d doesn't fit in image") %
                                                 bbox.getX0() % bbox.getY0() % bbox.getWidth() % bbox.getHeight());
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
        throw lsst::pex::exceptions::LengthError(boost::format("Dimension mismatch: %dx%d v. %dx%d") %
                                                 getWidth() % getHeight() % rhs.getWidth() % rhs.getHeight());
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
/// Note that this isn't especially efficient; see \link secPixelAccessTutorial\endlink for
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
/// Create an uninitialised Image of the specified size 
template<typename PixelT>
image::Image<PixelT>::Image(int const width, int const height) :
    image::ImageBase<PixelT>(width, height) {}

/**
 * Create an uninitialised Image of the specified size
 *
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
image::Image<PixelT>::Image(std::pair<int, int> const dimensions // (width, height) of the desired Image
                           ) :
    image::ImageBase<PixelT>(dimensions) {}

/**
 * Copy constructor.
 *
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this may not be what you want.  See also operator<<=() to copy pixels between Image%s
 */
template<typename PixelT>
image::Image<PixelT>::Image(Image const& rhs, ///< Right-hand-side Image
                            bool const deep       ///< If true, new Image shares storage with rhs; if false
                                                  ///< make a new, standalone, ImageBase
                           ) :
    image::ImageBase<PixelT>(rhs, deep) {}

/**
 * Copy constructor to make a copy of part of an Image.
 * \note Unless \c deep is \c true, the new %image will share the old %image's pixels;
 * this is probably what you want 
 */
template<typename PixelT>
image::Image<PixelT>::Image(Image const& rhs,             ///< Right-hand-side Image
                            BBox const& bbox,             ///< Specify desired region
                            bool const deep               ///< If true, new ImageBase shares storage with rhs; if false
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
 */
template<typename PixelT>
image::Image<PixelT>::Image(std::string const& fileName, ///< File to read
                            int const hdu,               ///< Desired HDU
                            lsst::daf::base::DataProperty::PtrType metaData ///< file metaData (may point to NULL)
                           ) :
    image::ImageBase<PixelT>() {

    typedef boost::mpl::vector<
        lsst::afw::image::detail::types_traits<unsigned char>::image_t,
        lsst::afw::image::detail::types_traits<unsigned short>::image_t,
        lsst::afw::image::detail::types_traits<short>::image_t,
        lsst::afw::image::detail::types_traits<int>::image_t,
        lsst::afw::image::detail::types_traits<float>::image_t // ,
        //lsst::afw::image::detail::types_traits<double>::image_t
    > fits_img_types;

    if (!boost::filesystem::exists(fileName)) {
        throw lsst::pex::exceptions::NotFound(boost::format("File %s doesn't exist") % fileName);
    }

    if (!image::fits_read_image<fits_img_types>(fileName, *this->_getRawImagePtr(), metaData)) {
        throw lsst::pex::exceptions::FitsError(boost::format("Failed to read %s HDU %d") % fileName % hdu);
    }
    this->_setRawView();
}
/**
 * Write an Image to the specified file
 */
template<typename PixelT>
void image::Image<PixelT>::writeFits(
	std::string const& fileName,    ///< File to write
#if 1                                   // Old name for boost::shared_ptrs
        typename lsst::daf::base::DataProperty::PtrType metaData //!< metadata to write to header; or NULL
#else
        typename lsst::daf::base::DataProperty::ConstPtr metaData //!< metadata to write to header; or NULL
#endif
                                    ) const {

    image::fits_write_view(fileName, _getRawView(), metaData);
}

/************************************************************************************************************/
//
// N.b. We could use the STL, but I find boost::lambda clearer, and more easily extended
// to e.g. setting random numbers
//    transform_pixels(_gilView, _gilView, lambda::ret<PixelT>(lambda::_1 + rhs));
// is equivalent to
//    transform_pixels(_gilView, _gilView, std::bind2nd(std::plus<PixelT>(), rhs));
//
using boost::lambda::ret;
using boost::lambda::_1;
using boost::lambda::_2;

/// Add scalar rhs to lhs
template<typename PixelT>
void image::Image<PixelT>::operator+=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 + rhs));
}

/// Add Image rhs to lhs
template<typename PixelT>
void image::Image<PixelT>::operator+=(Image<PixelT> const& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 + _2));
}

/// Subtract scalar rhs from lhs
template<typename PixelT>
void image::Image<PixelT>::operator-=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 - rhs));
}

/// Subtract Image rhs from lhs
template<typename PixelT>
void image::Image<PixelT>::operator-=(Image<PixelT> const& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 - _2));
}

/// Multiply lhs by scalar rhs
template<typename PixelT>
void image::Image<PixelT>::operator*=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 * rhs));
}

/// Multiply lhs by Image rhs (i.e. %pixel-by-%pixel multiplication)
template<typename PixelT>
void image::Image<PixelT>::operator*=(Image<PixelT> const& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 * _2));
}

/// Divide lhs by scalar rhs
template<typename PixelT>
void image::Image<PixelT>::operator/=(PixelT const rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 / rhs));
}

/// Divide lhs by Image rhs (i.e. %pixel-by-%pixel division)
template<typename PixelT>
void image::Image<PixelT>::operator/=(Image<PixelT> const& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 / _2));
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
