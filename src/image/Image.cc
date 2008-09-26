#include <iostream>
#include "boost/lambda/lambda.hpp"
#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/gil/Image.h"

namespace image = lsst::afw::image;

template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(const int width, const int height) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(new _image_t(width, height)),
    _gilView(flipped_up_down_view(view(*_gilImage))),
    _x0(0), _y0(0)
{
}

template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(const std::pair<int, int> dimensions) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(new _image_t(dimensions.first, dimensions.second)),
    _gilView(flipped_up_down_view(view(*_gilImage))),
    _x0(0), _y0(0)
{
}

template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(const ImageBase& src,
                                    const bool deep) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(src._gilImage), // don't copy the pixels
    _gilView(subimage_view(flipped_up_down_view(view(*_gilImage)),
                            src._x0, src._y0, src.getWidth(), src.getHeight())),
    _x0(src._x0),
    _y0(src._y0)
{
    if (deep) {
        ImageBase tmp(dimensions());
        tmp <<= *this;                  // now copy the pixels
        swap(tmp);
    }
}

template<typename PixelT>
image::ImageBase<PixelT>::ImageBase(const ImageBase& src, const Bbox& bbox, const bool deep) :
    lsst::daf::data::LsstBase(typeid(this)),
    _gilImage(src._gilImage), // boost::shared_ptr, so don't copy the pixels
    _gilView(subimage_view(src._gilView,
                           bbox.getX0(), bbox.getY0(), bbox.getWidth(), bbox.getHeight())),
    _x0(src._x0 + bbox.getX0()), _y0(src._y0 + bbox.getY0())
{
    if (deep) {
        ImageBase tmp(dimensions());
        tmp <<= *this;                  // now copy the pixels
        swap(tmp);
    }
}

template<typename PixelT>
image::ImageBase<PixelT>& image::ImageBase<PixelT>::operator=(const ImageBase& src) {
    ImageBase tmp(src);
    swap(tmp);                          // See Meyers, Effective C++, Item 11
    
    return *this;
}

template<typename PixelT>
void image::ImageBase<PixelT>::operator<<=(const ImageBase& src) {
    if (dimensions() != src.dimensions()) {
        throw lsst::pex::exceptions::LengthError(boost::format("Dimension mismatch: %dx%d v. %dx%d") %
                                                 getWidth() % getHeight() % src.getWidth() % src.getHeight());
    }
    copy_pixels(src._gilView, _gilView);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::PixelReference image::ImageBase<PixelT>::operator()(int x, int y) {
    return const_cast<typename image::ImageBase<PixelT>::PixelReference>(
	static_cast<typename image::ImageBase<PixelT>::PixelConstReference>(_gilView(x, y)[0])
                       );
}

template<typename PixelT>
typename image::ImageBase<PixelT>::PixelConstReference
	image::ImageBase<PixelT>::operator()(int x, int y) const {
    return _gilView(x, y)[0];
}

template<typename PixelT>
void image::ImageBase<PixelT>::swap(ImageBase<PixelT> &rhs) {
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
template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::begin() const {
    return _gilView.begin();
}

template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::end() const {
    return _gilView.end();
}

template<typename PixelT>
typename image::ImageBase<PixelT>::reverse_iterator image::ImageBase<PixelT>::rbegin() const {
    return _gilView.rbegin();
}

template<typename PixelT>
typename image::ImageBase<PixelT>::reverse_iterator image::ImageBase<PixelT>::rend() const {
    return _gilView.rend();
}

template<typename PixelT>
typename image::ImageBase<PixelT>::iterator image::ImageBase<PixelT>::at(int x, int y) const {
    return _gilView.at(x, y);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::xy_locator image::ImageBase<PixelT>::xy_at(int x, int y) const {
    return xy_locator(_gilView.xy_at(x, y));
}

template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::x_at(int x, int y) const {
    return _gilView.x_at(x, y);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::row_begin(int y) const {
    return _gilView.row_begin(y);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::x_iterator image::ImageBase<PixelT>::row_end(int y) const {
    return _gilView.row_end(y);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::y_at(int x, int y) const {
    return _gilView.y_at(x, y);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::col_begin(int x) const {
    return _gilView.col_begin(x);
}

template<typename PixelT>
typename image::ImageBase<PixelT>::y_iterator image::ImageBase<PixelT>::col_end(int x) const {
    return _gilView.col_end(x);
}

/************************************************************************************************************/

template<typename PixelT>
image::ImageBase<PixelT>& image::ImageBase<PixelT>::operator=(const PixelT scalar) {
    fill_pixels(_gilView, scalar);

    return *this;
}

/************************************************************************************************************/
//
// On to Image itself.  ctors, cctors, and operator=
//
template<typename PixelT>
image::Image<PixelT>::Image(const int width, const int height) :
    image::ImageBase<PixelT>(width, height) {}

template<typename PixelT>
image::Image<PixelT>::Image(const std::pair<int, int> dimensions) :
    image::ImageBase<PixelT>(dimensions) {}

template<typename PixelT>
image::Image<PixelT>::Image(const Image& src,
                            const bool deep) :
    image::ImageBase<PixelT>(src, deep) {}

template<typename PixelT>
image::Image<PixelT>::Image(const Image& src, const Bbox& bbox, const bool deep) :
    image::ImageBase<PixelT>(src, bbox, deep) {}

template<typename PixelT>
image::Image<PixelT>& image::Image<PixelT>::operator=(const PixelT scalar) {
    this->ImageBase<PixelT>::operator=(scalar);

    return *this;
}

template<typename PixelT>
image::Image<PixelT>& image::Image<PixelT>::operator=(const Image& src) {
    this->ImageBase<PixelT>::operator=(src);
    
    return *this;
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

template<typename PixelT>
void image::Image<PixelT>::operator+=(const PixelT rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 + rhs));
}

template<typename PixelT>
void image::Image<PixelT>::operator+=(const Image<PixelT>& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 + _2));
}

template<typename PixelT>
void image::Image<PixelT>::operator-=(const PixelT rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 - rhs));
}

template<typename PixelT>
void image::Image<PixelT>::operator-=(const Image<PixelT>& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 - _2));
}

template<typename PixelT>
void image::Image<PixelT>::operator*=(const PixelT rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 * rhs));
}

template<typename PixelT>
void image::Image<PixelT>::operator*=(const Image<PixelT>& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 * _2));
}

template<typename PixelT>
void image::Image<PixelT>::operator/=(const PixelT rhs) {
    transform_pixels(_getRawView(), _getRawView(), ret<PixelT>(_1 / rhs));
}

template<typename PixelT>
void image::Image<PixelT>::operator/=(const Image<PixelT>& rhs) {
    transform_pixels(_getRawView(), rhs._getRawView(), _getRawView(), ret<PixelT>(_1 / _2));
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(T) \
   template class image::ImageBase<T>; \
   template class image::Image<T>; \
   //template void image::swap(Image<T>&, Image<T>&)

INSTANTIATE(boost::uint16_t);
INSTANTIATE(int);
INSTANTIATE(float);
INSTANTIATE(double);
