// -*- lsst-c++ -*-
#if !defined(LSST_AFW_MATH_STACK_H)
#define LSST_AFW_MATH_STACK_H
/**
 * @file Stack.h
 * @brief Functions to stack images
 * @ingroup stack
 */ 
#include <vector>
#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Statistics.h"

namespace lsst {
namespace afw {
namespace math {    

template<typename PixelT>
typename lsst::afw::image::Image<PixelT>::Ptr statisticsStack(
        std::vector<typename lsst::afw::image::Image<PixelT>::Ptr > &images,      ///< Images to process
        Property flags,
        StatisticsControl const& sctrl=StatisticsControl());

}}}

#endif
