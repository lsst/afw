// -*- lsst-c++ -*-
/**
 * @file Stack.cc
 * @brief Provide functions to stack images
 * @ingroup stack
 * @author Steve Bickerton
 *
 */
#include <vector>
#include "boost/shared_ptr.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Slice.h"

namespace afwImage = lsst::afw::image;
namespace afwMath  = lsst::afw::math;
namespace ex       = lsst::pex::exceptions;




/**************************************************************************
 *
 * column operators
 *
 **************************************************************************/

template<typename PixelT>
afwImage::Slice<PixelT>::Slice(image::Image<PixelT> &img) : 
    afwImage::Image<PixelT>(img), _sliceType(afwImage::ROW)
{

    // verify the img is a slice (row or column)
    if (img.getWidth() != 1 && img.getHeight() != 1) {
	throw LSST_EXCEPT(ex::OutOfRangeException, "Input image must be a slice (width or height == 1)");
    }  else if (img.getWidth() == 1 && img.getHeight() == 1) {
	throw LSST_EXCEPT(ex::InvalidParameterException, 
			  "1x1 image ambiguous, Do you want a row or column?");
    } else if (img.getWidth() == 1 && img.getHeight() != 1) {
	_sliceType = afwImage::COLUMN;
    } else if (img.getHeight() == 1 && img.getWidth() != 1) {
	_sliceType = afwImage::ROW;
    }

    // what about 1xn images where a 1x1 row slice is desired?
    // what about nx1 images wehre a 1x1 column slice is desired?
}


// make a bbox slice and operate 
template<typename PixelT>
void afwImage::Slice<PixelT>::operator+=(afwImage::Image<PixelT> &img) {
    operate<Plus<PixelT> >(img, *this, _sliceType);
}
template<typename PixelT>
void afwImage::Slice<PixelT>::operator-=(afwImage::Image<PixelT> &img) {
    operate<Minus<PixelT> >(img, *this, _sliceType);
}
template<typename PixelT>
void afwImage::Slice<PixelT>::operator*=(afwImage::Image<PixelT> &img) {
    operate<Mult<PixelT> >(img, *this, _sliceType);
}
template<typename PixelT>
void afwImage::Slice<PixelT>::operator/=(afwImage::Image<PixelT> &img) {
    operate<Div<PixelT> >(img, *this, _sliceType);
}


////////////////////////////////////////////////////////////
#if 0
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator+(afwImage::Image<PixelT> &img, afwImage::Slice<PixelT> &slc) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    operate<Plus<PixelT> >(*retImg, slc, slc.getSliceType());
    return retImg;
}
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator+(afwImage::Slice<PixelT> &slc, afwImage::Image<PixelT> &img) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    operate<Plus<PixelT> >(*retImg, slc, slc.getSliceType());
    return retImg;
}
#endif

////////////////////////////////////////////////////////////
#if 0
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator+(afwImage::Image<PixelT> &img) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    afwImage::Slice<PixelT>::operator+=(*retImg);
    return retImg;
}
#endif
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator-(afwImage::Image<PixelT> &img) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    afwImage::Slice<PixelT>::operator-=(*retImg);
    return retImg;
}
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator*(afwImage::Image<PixelT> &img) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    afwImage::Slice<PixelT>::operator*=(*retImg);
    return retImg;
}
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator/(afwImage::Image<PixelT> &img) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    afwImage::Slice<PixelT>::operator/=(*retImg);
    return retImg;
}


/**
 *
 *
 */
template<typename ImageT>
typename ImageT::Ptr afwImage::sliceOperate(
					    ImageT &image,
					    ImageT &slice,
					    std::string sliceType,
					    char op,
					    bool deep
					    ) {

    typename ImageT::Ptr img(new ImageT(image, deep));
    typename afwImage::Slice<typename ImageT::Pixel> slc(slice);

    if (op == '+') {
	slc += *img;
    } else if (op == '-') {
	slc -= *img;
    } else if (op == '*') {
	slc *= *img;
    } else if (op == '/') {
	slc /= *img;
    } else {
	throw LSST_EXCEPT(ex::InvalidParameterException, "Invalid operator.  use +-*/.");
    }

    return img;
}





/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_SLICES(TYPE) \
    template afwImage::Slice<TYPE>::Slice(afwImage::Image<TYPE> &image); \
    template afwImage::Image<TYPE>::Ptr afwImage::sliceOperate(		\
							       afwImage::Image<TYPE> &image, \
							       afwImage::Image<TYPE> &slice, \
							       std::string sliceType, \
							       char op, bool deep);

INSTANTIATE_SLICES(double);
INSTANTIATE_SLICES(float);


#define INSTANTIATE_SLICE_VOID(OP_EQ, T) \
    template void afwImage::Slice<T>::operator OP_EQ(afwImage::Image<T> &rhs);

#define INSTANTIATE_SLICE_IMAGE(OP_EQ, T) \
    template afwImage::Image<T>::Ptr afwImage::Slice<T>::operator OP_EQ(afwImage::Image<T> &rhs);

#define INSTANTIATE_OVERLOAD(OP_EQ, T) \
    template afwImage::Image<T>::Ptr afwImage::Slice<T>::operator OP_EQ(afwImage::Image<T> &img, afwImage::Slice<T> &slc); \
    template afwImage::Image<T>::Ptr afwImage::Slice<T>::operator OP_EQ(afwImage::Slice<T> &slc, afwImage::Image<T> &img);


#define INSTANTIATE_SLICE_OPS(T) \
   INSTANTIATE_SLICE_VOID(+=, T);\
   INSTANTIATE_SLICE_VOID(-=, T);\
   INSTANTIATE_SLICE_IMAGE(-, T);
   //   INSTANTIATE_OVERLOAD(+, T);

INSTANTIATE_SLICE_OPS(boost::uint16_t);
INSTANTIATE_SLICE_OPS(int);
INSTANTIATE_SLICE_OPS(float);
INSTANTIATE_SLICE_OPS(double);
