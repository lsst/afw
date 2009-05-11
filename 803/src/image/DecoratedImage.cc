/**
 * \file
 * \brief An Image with associated metadata
 */
#include <iostream>
#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"

namespace image = lsst::afw::image;

template<typename PixelT>
void image::DecoratedImage<PixelT>::init() {
    setMetadata(lsst::daf::base::PropertySet::Ptr(new lsst::daf::base::PropertySet()));
    _gain = 0;
}

/// Create an %image of the specified size
template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const int width, ///< desired number of columns
                                              const int height ///< desired number of rows
                                             ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _image(new Image<PixelT>(width, height))
{
    init();
}
/**
 * Create an %image of the specified size
 * \note Many lsst::afw::image and lsst::afw::math objects define a \c dimensions member
 * which may be conveniently used to make objects of an appropriate size
 */
template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(
        const std::pair<int, int> dimensions // (width, height) of the desired Image
                                             ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _image(new Image<PixelT>(dimensions))
{
    init();
}
/**
 * Create a DecoratedImage wrapping \p rhs
 *
 * Note that this ctor shares pixels with the rhs; it isn't a deep copy
 */
template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(typename Image<PixelT>::Ptr rhs ///< Image to go into the DecoratedImage
                                             ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _image(rhs)
{
    init();
}
/**
 * Copy constructor
 *
 * Note that the lhs will share memory with the rhs unless \p deep is true
 */
template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const DecoratedImage& src, ///< right hand side
                                              const bool deep            ///< Make deep copy?
                                             ) :
    lsst::daf::data::LsstBase(typeid(this)),
    _image(src._image), _gain(src._gain) {

    setMetadata(src.getMetadata());
    if (deep) {
        typename Image<PixelT>::Ptr tmp = typename Image<PixelT>::Ptr(new Image<PixelT>(getDimensions()));
        *tmp <<= *_image;                // now copy the pixels
        _image.swap(tmp);
    }
}
/**
 * Assignment operator
 *
 * N.b. this is a shallow assignment; use operator<<=() if you want to copy the pixels
 */
template<typename PixelT>
image::DecoratedImage<PixelT>& image::DecoratedImage<PixelT>::operator=(const DecoratedImage& src) {
    DecoratedImage tmp(src);
    swap(tmp);                          // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename PixelT>
void image::DecoratedImage<PixelT>::swap(DecoratedImage &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25
    
    swap(_image, rhs._image);           // just swapping the pointers
    swap(_gain, rhs._gain);
}

template<typename PixelT>
void image::swap(DecoratedImage<PixelT>& a, DecoratedImage<PixelT>& b) {
    a.swap(b);
}

/************************************************************************************************************/
//
// FITS code
//
#include <boost/mpl/vector.hpp>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Image.h"

#include "boost/gil/gil_all.hpp"
#include "lsst/afw/image/fits/fits_io.h"
#include "lsst/afw/image/fits/fits_io_mpl.h"
/**
 * Create a DecoratedImage from a FITS file
 */
template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const std::string& fileName, ///< File to read
                                              const int hdu                ///< The HDU to read
                                             ) :
    lsst::daf::data::LsstBase(typeid(this))
{             ///< HDU within the file
    init();
    _image = typename Image<PixelT>::Ptr(new Image<PixelT>(fileName, hdu, getMetadata()));
}

/************************************************************************************************************/
/**
 * Write a FITS file
 */
template<typename PixelT>
void image::DecoratedImage<PixelT>::writeFits(
	const std::string& fileName,                        //!< the file to write
        typename lsst::daf::base::PropertySet::Ptr metadata //!< metadata to write to header; or NULL
                                             ) const {
    image::fits_write_view(fileName, _image->_getRawView(), metadata);
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class image::DecoratedImage<boost::uint16_t>;
template class image::DecoratedImage<int>;
template class image::DecoratedImage<float>;
template class image::DecoratedImage<double>;
