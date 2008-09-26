#include <iostream>
#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/gil/Image.h"

namespace image = lsst::afw::image;

template<typename PixelT>
void image::DecoratedImage<PixelT>::init() {
    _metaData = lsst::daf::base::DataProperty::createPropertyNode("FitsMetaData");
    _gain = 0;
}

template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const int width, const int height) :
    _image(new Image<PixelT>(width, height))
{
    init();
}

template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const std::pair<int, int> dimensions) :
    _image(new Image<PixelT>(dimensions))
{
    init();
}

template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(typename Image<PixelT>::Ptr src) :
    _image(src)
{
    init();
}

template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const DecoratedImage& src,
                                              const bool deep) :
    _image(src._image), _metaData(src._metaData), _gain(src._gain) {

    if (deep) {
        typename Image<PixelT>::Ptr tmp = typename Image<PixelT>::Ptr(new Image<PixelT>(dimensions()));
        *tmp <<= *_image;                // now copy the pixels
        _image.swap(tmp);
    }
}

template<typename PixelT>
image::DecoratedImage<PixelT>& image::DecoratedImage<PixelT>::operator =(const DecoratedImage& src) {
    DecoratedImage tmp(src);
    swap(tmp);                          // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename PixelT>
void image::DecoratedImage<PixelT>::swap(DecoratedImage &rhs) {
    using std::swap;                    // See Meyers, Effective C++, Item 25
    
    swap(_image, rhs._image);           // just swapping the pointers
    swap(_metaData, rhs._metaData);
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
#include "lsst/gil/Image.h"

#include "boost/gil/gil_all.hpp"
#include "lsst/gil/fits/fits_io.h"
#include "lsst/gil/fits/fits_io_mpl.h"

template<typename PixelT>
image::DecoratedImage<PixelT>::DecoratedImage(const std::string& fileName, const int hdu) {
    init();

    typedef boost::mpl::vector<
        lsst::afw::image::details::types_traits<unsigned char>::image_t,
        lsst::afw::image::details::types_traits<unsigned short>::image_t,
        lsst::afw::image::details::types_traits<short>::image_t,
        lsst::afw::image::details::types_traits<int>::image_t,
            lsst::afw::image::details::types_traits<float>::image_t // ,
        //lsst::afw::image::details::types_traits<double>::image_t
    > fits_img_types;

    _image = typename Image<PixelT>::Ptr(new Image<PixelT>());

    if (!image::fits_read_image<fits_img_types>(fileName, *_image->_getRawImagePtr(), _metaData)) {
        throw lsst::pex::exceptions::FitsError(boost::format("Failed to read %s HDU %d") % fileName % hdu);
    }
    _image->_setRawView();
}

/************************************************************************************************************/

template<typename PixelT>
void image::DecoratedImage<PixelT>::DecoratedImage::writeFits(
	const std::string& fileName,
#if 1                                   // Old name for boost::shared_ptrs
        typename lsst::daf::base::DataProperty::PtrType metaData //!< metadata to write to header; or NULL
#else
        typename lsst::daf::base::DataProperty::ConstPtr metaData //!< metadata to write to header; or NULL
#endif
                                                        ) const {

    image::fits_write_view(fileName, _image->_getRawView(), metaData);
}

/************************************************************************************************************/
//
// Explicit instantiations
//
template class image::DecoratedImage<boost::uint16_t>;
//template class image::DecoratedImage<int>;
template class image::DecoratedImage<float>;
//template class image::DecoratedImage<double>; // not yet available as a gray64_noscale due to RHL's laziness
