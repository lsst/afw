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


namespace details {
    
/*
 * A function to load values from the weight vector into the variance plane of the pixelSet.
 */
template <typename PixelT>    
void loadVariance(std::vector<PixelT> const &wvector, math::MaskedVector<PixelT> &pixelSet) {
    unsigned int j = 0;
    for (typename std::vector<PixelT>::const_iterator pVec = wvector.begin();
         pVec != wvector.end(); ++pVec) {
        (*pixelSet.getVariance())(j, 0) = static_cast<image::VariancePixel>(*pVec);
        ++j;
    }
}

/*
 * A bit counter (to make sure that only one type of statistics has been requested)
 */
int bitcount(unsigned int x)
{
    int b;
    for (b = 0; x != 0; x >>= 1) {
        if (x & 01) {
            b++;
        }
    }
    return b;
}

/*
 * Check that only one type of statistics has been requested.
 */
void checkOnlyOneFlag(unsigned int flags) {
    if (bitcount(flags) != 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Requested more than one type of statistic to make the image stack.");
                          
    }
    
}
    
}






/****************************************************************************
 *
 * stack MaskedImages
 *
 ****************************************************************************/

namespace details {
    
/*
 * A function to handle MaskedImage stacking
 */
    
template<typename PixelT, bool UseVariance>
typename image::MaskedImage<PixelT>::Ptr computeMaskedImageStack(
                             std::vector<typename image::MaskedImage<PixelT>::Ptr > const &images,
                             math::Property flags,               
                             math::StatisticsControl const& sctrl,
                             std::vector<PixelT> const &wvector
                                                          ) {

    // create the image to be returned
    typedef image::MaskedImage<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions()));

    
    // get a list of row_begin iterators
    typedef typename image::MaskedImage<PixelT>::x_iterator x_iterator;
    std::vector<x_iterator> rows;
    rows.reserve(images.size());
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (unsigned int i = 0; i < images.size(); ++i) {
            x_iterator ptr = images[i]->row_begin(y);
            if (y == 0) {
                rows.push_back(ptr);
            } else {
                rows[i] = ptr;
            }
        }
    }

    // get a list to contain a pixel from x,y for each image
    math::MaskedVector<PixelT> pixelSet(images.size());
    math::StatisticsControl sctrlTmp(sctrl);

    // if we're forcing the user variances ...
    if (UseVariance) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        details::loadVariance(wvector, pixelSet); // put them in the variance vector
    }

    // loop over x,y ... the loop over the stack to fill pixelSet
    // - get the stats on pixelSet and put the value in the output image at x,y
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (x_iterator ptr = imgStack->row_begin(y), end = imgStack->row_end(y); ptr != end; ++ptr) {
            typename math::MaskedVector<PixelT>::iterator psPtr = pixelSet.begin();
            image::MaskPixel msk(0x0);
            for (unsigned int i = 0; i < images.size(); ++i, ++psPtr) {
                image::MaskPixel mskTmp = rows[i].mask();
                psPtr.value() = rows[i].image();
                psPtr.mask()  = mskTmp;

                // if we're not using the wvector weights, use the variance plane.
                if (! UseVariance) {
                    psPtr.variance() = rows[i].variance();
                }

                msk |= mskTmp;
                
                ++rows[i];
            }
            math::Statistics stat = math::makeStatistics(pixelSet, flags, sctrlTmp);
            
            PixelT variance = stat.getError()*stat.getError();
            *ptr = typename image::MaskedImage<PixelT>::Pixel(stat.getValue(), msk, variance);
        }
    }

    return imgStack;

}
    
} // end namespace details


/**
 * @brief A function to compute some statistics of a stack of Masked Images
 * @relates Statistics
 *
 * All the work is done in the function comuteMaskedImageStack.
 * A boolean template variable has been used to allow the compiler to generate the different instantiations
 *   to handle cases when we are, or are not, dealing with the variance plane.
 */
template<typename PixelT>
typename image::MaskedImage<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::MaskedImage<PixelT>::Ptr > &images,
        math::Property flags,               
        math::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                              ) {

    details::checkOnlyOneFlag(flags);

    // if we're going to use constant weights 
    if ( wvector.size() == images.size() ) {
        return details::computeMaskedImageStack<PixelT, true>(images, flags, sctrl, wvector);
        
    // if we're weighting by the pixel variance        
    } else if ( wvector.size() == 0 ) {
        return details::computeMaskedImageStack<PixelT, false>(images, flags, sctrl, wvector);
        
    // Fail if the number weights isn't the same as the number of images to be weighted.
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Weight vector must have same length as number of MaskedImages to be stacked.");
    }
}







/****************************************************************************
 *
 * stack Images
 *
 * All the work is done in the function comuteImageStack.
 * A boolean template variable has been used to allow the compiler to generate the different instantiations
 *   to handle cases when we are, or are not, dealing with the variance plane.
 *
 ****************************************************************************/


namespace details {
/***********************************************************************************/
/*
 * A function to compute some statistics of a stack of regular images
 */
template<typename PixelT, bool UseVariance>
typename image::Image<PixelT>::Ptr computeImageStack(
        std::vector<typename image::Image<PixelT>::Ptr > &images,  
        math::Property flags,               
        math::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                        ) {

    // create the image to be returned
    typedef image::Image<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions(), 0.0));

    math::StatisticsControl sctrlTmp(sctrl);
    math::MaskedVector<typename Image::Pixel> pixelSet(images.size());

    // set the mask to be an infinite iterator
    math::MaskImposter<image::MaskPixel> msk;

    // if we're going to use contant weights
    if ( UseVariance ) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        // copy the weights in to the variance vector
        details::loadVariance(wvector, pixelSet);
    }
        
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                (*pixelSet.getImage())(i, 0) = (*images[i])(x, y);
            }
            if (UseVariance) {
                math::Statistics stat =
                    math::makeStatistics(*pixelSet.getImage(), msk, *pixelSet.getVariance(),
                                         flags, sctrlTmp);
                (*imgStack)(x, y) = stat.getValue();
            } else {
                math::Statistics stat = math::makeStatistics(pixelSet, flags, sctrlTmp);
                (*imgStack)(x, y) = stat.getValue();
            }
        }
    }

    return imgStack;
}

} // end namespace details


/**
 * @brief A function to compute some statistics of a stack of regular images
 * @relates Statistics
 */
template<typename PixelT>
typename image::Image<PixelT>::Ptr math::statisticsStack(
        std::vector<typename image::Image<PixelT>::Ptr > &images,  
        math::Property flags,               
        math::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                        ) {

    details::checkOnlyOneFlag(flags);

    // if we're going to use contant weights
    if ( wvector.size() == images.size() ) {
        return details::computeImageStack<PixelT, true>(images, flags, sctrl, wvector);
        
    } else if ( wvector.size() == 0 ) {
        return details::computeImageStack<PixelT, false>(images, flags, sctrl, wvector);

    // Fail if the number weights isn't the same as the number of images to be weighted.
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Weight vector must have same length as number of Images to be stacked.");
    }

}





/****************************************************************************
 *
 * stack VECTORS
 *
 ****************************************************************************/

namespace details {

/**********************************************************************************/
/*
 * A function to compute some statistics of a stack of vectors
 */
template<typename PixelT, bool UseVariance>
typename boost::shared_ptr<std::vector<PixelT> > computeVectorStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        math::Property flags,               
        math::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                                      ) {

    // create the image to be returned
    typedef std::vector<PixelT> Vect;
    typename boost::shared_ptr<Vect> vecStack(new Vect(vectors[0]->size(), 0.0));

    math::MaskedVector<PixelT> pixelSet(vectors.size()); // values from a given pixel of each image

    math::StatisticsControl sctrlTmp(sctrl);
    // set the mask to be an infinite iterator
    math::MaskImposter<image::MaskPixel> msk;
    
    // Use constant weights for each layer
    if ( UseVariance ) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        // copy the weights in to the variance vector
        details::loadVariance(wvector, pixelSet);
    }
    
    // collect elements from the stack into the MaskedVector to do stats
    for (unsigned int x = 0; x < vectors[0]->size(); ++x) {
        typename math::MaskedVector<PixelT>::iterator psPtr = pixelSet.begin();
        for (unsigned int i = 0; i < vectors.size(); ++i, ++psPtr) {
            psPtr.value() = (*vectors[i])[x];
        }
        if (UseVariance) {
            math::Statistics stat = math::makeStatistics(*pixelSet.getImage(), msk, *pixelSet.getVariance(),
                                                         flags, sctrlTmp);
            (*vecStack)[x] = stat.getValue(flags);
        } else {
            math::Statistics stat = math::makeStatistics(pixelSet, flags, sctrlTmp);
            (*vecStack)[x] = stat.getValue(flags);
        }
    }

    return vecStack;
}

} // end namespace details


/**
 * @brief A function to handle stacking a vector of vectors
 * @relates Statistics
 *
 * All the work is done in the function comuteVectorStack.
 * A boolean template variable has been used to allow the compiler to generate the different instantiations
 *   to handle cases when we are, or are not, dealing with the variance plane.
 */
template<typename PixelT>
typename boost::shared_ptr<std::vector<PixelT> > math::statisticsStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        math::Property flags,               
        math::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                                      ) {

    details::checkOnlyOneFlag(flags);

    // Use constant weights for each layer
    if ( wvector.size() == vectors.size() ) {
        return details::computeVectorStack<PixelT, true>(vectors, flags, sctrl, wvector);
    // Use no weights.
    } else if ( wvector.size() == 0 ) {
        return details::computeVectorStack<PixelT, false>(vectors, flags, sctrl, wvector);
    // Fail if the number weights isn't the same as the number of vectors to be weighted.
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Weight vector must have same length as number of vectors to be stacked.");
    }
}






/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_STACKS(TYPE) \
    template image::Image<TYPE>::Ptr math::statisticsStack<TYPE>(       \
            std::vector<image::Image<TYPE>::Ptr > &images, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector);                          \
    template image::MaskedImage<TYPE>::Ptr math::statisticsStack<TYPE>( \
            std::vector<image::MaskedImage<TYPE>::Ptr > &images, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector);                          \
    template boost::shared_ptr<std::vector<TYPE> > math::statisticsStack<TYPE>( \
            std::vector<boost::shared_ptr<std::vector<TYPE> > > &vectors, \
            lsst::afw::math::Property flags, \
            lsst::afw::math::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector);

INSTANTIATE_STACKS(double);
INSTANTIATE_STACKS(float);
