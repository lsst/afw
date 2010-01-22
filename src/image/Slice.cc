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
afwImage::Slice<PixelT>::Slice(image::Image<PixelT> &img) : afwImage::Image<PixelT>(img) {

    // verify the img is a slice (row or column)
    if (img.getWidth() != 1 && img.getHeight() != 1) {
	throw LSST_EXCEPT(ex::OutOfRangeException, "Input image must be a slice (width or height == 1)");
    }
    // what about 1x1 images
    // what about 1xn images where a 1x1 row slice is desired?
    // what about nx1 images wehre a 1x1 column slice is desired?
}

template<typename PixelT>
void afwImage::Slice<PixelT>::operator+=(afwImage::Image<PixelT> &rhs) {
    if (this->getWidth() == 1) {
	afwImage::sliceOperate(rhs, *this, "column", '+', false);
    } else {
	afwImage::sliceOperate(rhs, *this, "row", '+', false);
    }
}

template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwImage::Slice<PixelT>::operator+(afwImage::Image<PixelT> &rhs) {
    typename afwImage::Image<PixelT>::Ptr ret(new afwImage::Image<PixelT>(rhs, true));
    afwImage::Slice<PixelT>::operator+=(*ret);
    return ret;
}

/**
 *
 *
 */
template<typename ImageT, typename SliceT>
typename ImageT::Ptr afwImage::sliceOperate(
        ImageT const &image,
	SliceT const &slice,
	std::string sliceType,
	char op,
	bool deep
							    ) {

    // make sure slice has the right dimensions
    int n, dx = 1, dy = 1;
    int i = 0, *x, *y, zero = 0;
    if (sliceType == "column")  {
	if ( slice.getWidth() != 1) {
	    throw LSST_EXCEPT(ex::InvalidParameterException, "column slice must have width 1");
	}
	if ( slice.getHeight() != image.getHeight() ) {
	    throw LSST_EXCEPT(ex::InvalidParameterException, 
			      "image and column slice must have the same height.");
	}
	n = image.getWidth();
	dy = image.getHeight();
	x = &i;
	y = &zero;
    } else if (sliceType == "row") {
	if ( slice.getHeight() != 1) {
	    throw LSST_EXCEPT(ex::InvalidParameterException, "row slice must have height 1");
	}
	if ( slice.getWidth() != image.getWidth() ) {
	    throw LSST_EXCEPT(ex::InvalidParameterException, 
			      "image and row slice must have the same width.");
	}
	n = image.getHeight();
	dx = image.getWidth();
	x = &zero;
	y = &i;
    } else {
	throw LSST_EXCEPT(ex::InvalidParameterException, "Only column and row slice types are available.");
    }

    typename ImageT::Ptr outImage(new ImageT(image, deep));

    for (i = 0; i < n; ++i) {
	
	// make a bbox slice and operate 
	afwImage::BBox bbox(afwImage::PointI(*x, *y), dx, dy);
	
	ImageT imgBox(image, bbox);
	ImageT outBox(*outImage, bbox);
	if (op == '+') {
	    outBox += slice;
	} else if (op == '-') {
	    outBox -= slice;
	} else if (op == '*') {
	    outBox *= slice;
	} else if (op == '/') {
	    outBox /= slice;
	} else {
	    throw LSST_EXCEPT(ex::InvalidParameterException, "Invalid operator.  use +-*/.");
	}
    }

    return outImage;
}





/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_SLICES(TYPE) \
    template afwImage::Slice<TYPE>::Slice(afwImage::Image<TYPE> &image); \
    template afwImage::Image<TYPE>::Ptr afwImage::sliceOperate(		\
							       afwImage::Image<TYPE> const &image, \
							       afwImage::Image<TYPE> const &slice, \
							       std::string sliceType, \
							       char op, bool deep); \
    template afwImage::Image<TYPE>::Ptr afwImage::sliceOperate(		\
							       afwImage::Image<TYPE> const &image, \
							       afwImage::Slice<TYPE> const &slice, \
							       std::string sliceType, \
							       char op, bool deep);


INSTANTIATE_SLICES(double);
INSTANTIATE_SLICES(float);


#define INSTANTIATE_SLICE_VOID(OP_EQ, T) \
    template void afwImage::Slice<T>::operator OP_EQ(afwImage::Image<T> &rhs);

#define INSTANTIATE_SLICE_IMAGE(OP_EQ, T) \
    template afwImage::Image<T>::Ptr afwImage::Slice<T>::operator OP_EQ(afwImage::Image<T> &rhs);


#define INSTANTIATE_SLICE_OPS(T) \
   INSTANTIATE_SLICE_VOID(+=, T);\
   INSTANTIATE_SLICE_IMAGE(+, T);

INSTANTIATE_SLICE_OPS(boost::uint16_t);
INSTANTIATE_SLICE_OPS(int);
INSTANTIATE_SLICE_OPS(float);
INSTANTIATE_SLICE_OPS(double);
