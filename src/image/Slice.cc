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
namespace imageDetails  = lsst::afw::image::details;


/**************************************************************************
 *
 * column operators
 *
 **************************************************************************/

template<typename PixelT>
afwImage::Slice<PixelT>::Slice(image::Image<PixelT> &img) : 
    afwImage::Image<PixelT>(img), _sliceType(imageDetails::ROW)
{

    // verify the img is a slice (row or column)
    if (img.getWidth() != 1 && img.getHeight() != 1) {
	throw LSST_EXCEPT(ex::OutOfRangeException, "Input image must be a slice (width or height == 1)");
    }  else if (img.getWidth() == 1 && img.getHeight() == 1) {
	throw LSST_EXCEPT(ex::InvalidParameterException, 
			  "1x1 image ambiguous, Do you want a row or column?");
    } else if (img.getWidth() == 1 && img.getHeight() != 1) {
	_sliceType = imageDetails::COLUMN;
    } else if (img.getHeight() == 1 && img.getWidth() != 1) {
	_sliceType = imageDetails::ROW;
    }

    // what about 1xn images where a 1x1 row slice is desired?
    // what about nx1 images wehre a 1x1 column slice is desired?
}


/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_SLICES(TYPE) \
    template afwImage::Slice<TYPE>::Slice(afwImage::Image<TYPE> &image);

INSTANTIATE_SLICES(double);
INSTANTIATE_SLICES(float);

