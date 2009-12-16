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
#include "lsst/afw/math/Stack.h"
#include "lsst/afw/math/MaskedVector.h"

namespace image = lsst::afw::image;
namespace math  = lsst::afw::math;
namespace ex    = lsst::pex::exceptions;

/************************************************************************************************************/
/**
 * A function to compute some statistics of a stack
 * @relates Statistics
 */
template<typename PixelT>
typename image::Image<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::Image<PixelT>::Ptr > &images,  
        math::Property flags,               
        math::StatisticsControl const& sctrl) {

    // create the image to be returned
    typedef image::Image<PixelT> ImageT;
    typename ImageT::Ptr imgStack(new ImageT(images[0]->getDimensions(), 0.0));

    std::vector<PixelT> pixelSet(images.size()); // values from a given pixel of each image
    //math::MaskedVector<typename ImageT::Pixel> pixelSet(images.size());
    
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                pixelSet[i] = (*images[i])(x, y);
            }
            (*imgStack)(x, y) = math::makeStatistics(pixelSet, flags, sctrl).getValue(flags);
        }
    }

    return imgStack;
}

/**
 * A function to compute some statistics of a stack
 * @relates Statistics
 */
template<typename PixelT>
typename image::MaskedImage<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::MaskedImage<PixelT>::Ptr > &images,
        math::Property flags,               
        math::StatisticsControl const& sctrl ) {

    // create the image to be returned
    typedef image::MaskedImage<PixelT> ImageT;
    typename ImageT::Ptr imgStack(new ImageT(images[0]->getDimensions()));

    //std::vector<typename ImageT::Pixel> pixelSet(images.size()); // values from a given pixel of each image
    
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            image::MaskPixel msk = 0x0;
            
            math::MaskedVector<PixelT> pixelSet(images.size());
            for (unsigned int i = 0; i < images.size(); ++i) {
                image::MaskPixel mskTmp = (*images[i]->getMask())(x, y);
                (*pixelSet.getImage())(i, 0)     = (*images[i]->getImage())(x, y);
                (*pixelSet.getMask())(i, 0)      = mskTmp;
                (*pixelSet.getVariance())(i, 0)  = (*images[i]->getVariance())(x, y);
                msk |= mskTmp;
            }
            math::Statistics stat = math::makeStatistics(pixelSet, flags, sctrl);
            (*imgStack->getImage())(x, y)    = stat.getValue(flags);
            (*imgStack->getMask())(x, y)     = msk;
            (*imgStack->getVariance())(x, y) = stat.getError(flags)*stat.getError(flags);
        }
    }

    return imgStack;
}




/**
 * A function to compute some statistics of a stack
 * @relates Statistics
 */
template<typename PixelT>
typename boost::shared_ptr<std::vector<PixelT> > math::statisticsStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        math::Property flags,               
        math::StatisticsControl const& sctrl ) {

    // create the image to be returned
    typedef std::vector<PixelT> VectT;
    typename boost::shared_ptr<VectT> vecStack(new VectT(vectors[0]->size(), 0.0));

    std::vector<PixelT> pixelSet(vectors.size()); // values from a given pixel of each image

    // get the desired statistic
    for (unsigned int x = 0; x < vectors[0]->size(); ++x) {
        for (unsigned int i = 0; i != vectors.size(); ++i) {
            pixelSet[i] = (*vectors[i])[x];
        }
        (*vecStack)[x] = math::makeStatistics(pixelSet, flags, sctrl).getValue(flags);
    }

    return vecStack;
}




/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_STACKS(TYPE) \
    template image::Image<TYPE>::Ptr math::statisticsStack<TYPE>(       \
            std::vector<image::Image<TYPE>::Ptr > &images, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl);\
    template image::MaskedImage<TYPE>::Ptr math::statisticsStack<TYPE>( \
            std::vector<image::MaskedImage<TYPE>::Ptr > &images, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl);\
    template boost::shared_ptr<std::vector<TYPE> > math::statisticsStack<TYPE>( \
            std::vector<boost::shared_ptr<std::vector<TYPE> > > &vectors, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl);

INSTANTIATE_STACKS(double);
INSTANTIATE_STACKS(float);
