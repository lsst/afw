// -*- lsst-c++ -*-
/**
 * @file Stack.cc
 * @brief Provide functions to stack images
 * @ingroup stack
 * @author Steve Bickerton
 *
 */
#include <vector>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Stack.h"
#include "lsst/afw/math/MaskedVector.h"

namespace image = lsst::afw::image;
namespace math  = lsst::afw::math;
namespace ex    = lsst::pex::exceptions;

/************************************************************************************************************/
/**
 * A function to compute some statistics of a stack
 */
template<typename PixelT>
typename image::Image<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::Image<PixelT>::Ptr > &images,      ///< Images to process
        math::Property flags,                ///< The desired quantity
        math::StatisticsControl const& sctrl ///< How to calculate the desired statistic
                                          )
{
    // create the image to be returned
    typedef image::Image<PixelT> ImageT;
    typename ImageT::Ptr imgStack(new ImageT(images[0]->getDimensions(), 0.0));

    std::vector<typename ImageT::Pixel> pixelSet(images.size()); // values from a given pixel of each image
    //math::MaskedVector<typename ImageT::Pixel> pixelSet(images.size());
    
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                pixelSet[i] = (*images[i])(x, y);
            }
            (*imgStack)(x, y) = math::makeStatistics(pixelSet, flags, sctrl).getValue();
        }
    }

    return imgStack;
}

template<typename PixelT>
typename image::MaskedImage<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::MaskedImage<PixelT>::Ptr > &images,      ///< Images to process
        math::Property flags,                ///< The desired quantity
        math::StatisticsControl const& sctrl ///< How to calculate the desired statistic
                                          )
{
    // create the image to be returned
    typedef image::MaskedImage<PixelT> ImageT;
    typename ImageT::Ptr imgStack(new ImageT(images[0]->getDimensions(), 0.0));

    //std::vector<typename ImageT::Pixel> pixelSet(images.size()); // values from a given pixel of each image
    math::MaskedVector<typename ImageT::Pixel> pixelSet(images.size());
    
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                pixelSet[i]. = (*images[i])(x, y);
                pixelSet[
            }
            (*imgStack)(x, y) = math::makeStatistics(pixelSet, flags, sctrl).getValue();
        }
    }

    return imgStack;
}





/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_STACKS(TYPE) \
    template image::Image<TYPE>::Ptr math::statisticsStack<TYPE>( \
            std::vector<image::Image<TYPE>::Ptr > &images, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl);

INSTANTIATE_STACKS(double);
INSTANTIATE_STACKS(float);
