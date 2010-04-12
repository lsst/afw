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
#include "lsst/afw/image/ImageSlice.h"

namespace afwImage      = lsst::afw::image;
namespace afwMath       = lsst::afw::math;
namespace ex            = lsst::pex::exceptions;


template<typename PixelT>
afwImage::ImageSlice<PixelT>::ImageSlice(
                               image::Image<PixelT> &img
                              ) : 
    afwImage::Image<PixelT>(img),
    _sliceType(ROW)
{

    // verify the img is a slice (row or column)
    if (img.getWidth() != 1 && img.getHeight() != 1) {
	throw LSST_EXCEPT(ex::OutOfRangeException, "Input image must be a slice (width or height == 1)");
    }  else if (img.getWidth() == 1 && img.getHeight() == 1) {
	throw LSST_EXCEPT(ex::InvalidParameterException, 
			  "1x1 image ambiguous, Do you want a row or column?");
    } else if (img.getWidth() == 1 && img.getHeight() != 1) {
	_sliceType = COLUMN;
    } else if (img.getHeight() == 1 && img.getWidth() != 1) {
	_sliceType = ROW;
    }

    // what about 1xn images where a 1x1 row slice is desired?
    // what about nx1 images wehre a 1x1 column slice is desired?
}




/**************************************************************************
 *
 * column operators
 *
 **************************************************************************/


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// overload +
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator+(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    *retImg += slc;
    return retImg;
}
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator+(afwImage::ImageSlice<PixelT> &slc, afwImage::Image<PixelT> &img) {
    return afwImage::operator+(img, slc);
}
template<typename PixelT>    
void afwImage::operator+=(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    afwImage::details::operate<afwImage::details::Plus<PixelT> >(img, slc, slc.getImageSliceType());
}

// -----------------------------------------------------------------
// overload -
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator-(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    *retImg -= slc;
    return retImg;
}
template<typename PixelT>    
void afwImage::operator-=(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    details::operate<details::Minus<PixelT> >(img, slc, slc.getImageSliceType());
}


// ******************************************************************
// overload *
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator*(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    *retImg *= slc;
    return retImg;
}
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator*(afwImage::ImageSlice<PixelT> &slc, afwImage::Image<PixelT> &img) {
    return afwImage::operator*(img, slc);
}
template<typename PixelT>    
void afwImage::operator*=(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    details::operate<details::Mult<PixelT> >(img, slc, slc.getImageSliceType());
}

// ///////////////////////////////////////////////////////////////////
// overload /
template<typename PixelT>    
typename afwImage::Image<PixelT>::Ptr afwImage::operator/(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    typename afwImage::Image<PixelT>::Ptr retImg(new afwImage::Image<PixelT>(img, true));
    *retImg /= slc;
    return retImg;
}
template<typename PixelT>    
void afwImage::operator/=(afwImage::Image<PixelT> &img, afwImage::ImageSlice<PixelT> &slc) {
    details::operate<details::Div<PixelT> >(img, slc, slc.getImageSliceType());
}




/*
 * Explicit Instantiations
 *
 */

#define INSTANTIATE_SLICE_OP_SYM(TYPE, OP) \
    template afwImage::Image<TYPE>::Ptr afwImage::operator OP(afwImage::Image<TYPE> &img, afwImage::ImageSlice<TYPE> &slc); \
    template afwImage::Image<TYPE>::Ptr afwImage::operator OP(afwImage::ImageSlice<TYPE> &slc, afwImage::Image<TYPE> &img)


#define INSTANTIATE_SLICE_OP_ASYM(TYPE, OP) \
    template afwImage::Image<TYPE>::Ptr afwImage::operator OP(afwImage::Image<TYPE> &img, afwImage::ImageSlice<TYPE> &slc)
    

#define INSTANTIATE_SLICE_OPEQ(TYPE, OP) \
    template void afwImage::operator OP(afwImage::Image<TYPE> &img, afwImage::ImageSlice<TYPE> &slc);



#define INSTANTIATE_SLICES(TYPE) \
    template afwImage::ImageSlice<TYPE>::ImageSlice(afwImage::Image<TYPE> &image); \
    INSTANTIATE_SLICE_OP_SYM(TYPE, +);                                  \
    INSTANTIATE_SLICE_OP_ASYM(TYPE, -);                                 \
    INSTANTIATE_SLICE_OP_SYM(TYPE, *);                                  \
    INSTANTIATE_SLICE_OP_ASYM(TYPE, /);                                 \
    INSTANTIATE_SLICE_OPEQ(TYPE, +=);                                   \
    INSTANTIATE_SLICE_OPEQ(TYPE, -=);                                   \
    INSTANTIATE_SLICE_OPEQ(TYPE, *=);                                   \
    INSTANTIATE_SLICE_OPEQ(TYPE, /=);


INSTANTIATE_SLICES(double);
INSTANTIATE_SLICES(float);

