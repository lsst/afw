// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
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

namespace afwImage = lsst::afw::image;
namespace afwMath  = lsst::afw::math;
namespace ex    = lsst::pex::exceptions;


namespace {
    
/*
 * A function to load values from the weight vector into the variance plane of the pixelSet.
 */
template <typename PixelT>    
void loadVariance(std::vector<PixelT> const &wvector, afwMath::MaskedVector<PixelT> &pixelSet) {
    unsigned int j = 0;
    for (typename std::vector<PixelT>::const_iterator pVec = wvector.begin();
         pVec != wvector.end(); ++pVec) {
        (*pixelSet.getVariance())(j, 0) = static_cast<afwImage::VariancePixel>(*pVec);
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
    if (bitcount(flags & ~afwMath::ERRORS) != 1) {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Requested more than one type of statistic to make the image stack.");
                          
    }
    
}
    
} // end anonymous namespace






/****************************************************************************
 *
 * stack MaskedImages
 *
 ****************************************************************************/

namespace {
    
/*
 * A function to handle MaskedImage stacking
 */
    
template<typename PixelT, bool UseVariance>
typename afwImage::MaskedImage<PixelT>::Ptr computeMaskedImageStack(
                             std::vector<typename afwImage::MaskedImage<PixelT>::Ptr > const &images,
                             afwMath::Property flags,               
                             afwMath::StatisticsControl const& sctrl,
                             std::vector<PixelT> const &wvector
                                                          ) {

    // create the image to be returned
    typedef afwImage::MaskedImage<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions()));

    // get a list of row_begin iterators
    typedef typename afwImage::MaskedImage<PixelT>::x_iterator x_iterator;
    std::vector<x_iterator> rows;
    rows.reserve(images.size());

    // get a list to contain a pixel from x,y for each image
    afwMath::MaskedVector<PixelT> pixelSet(images.size());
    afwMath::StatisticsControl sctrlTmp(sctrl);

    // if we're forcing the user variances ...
    if (UseVariance) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        loadVariance(wvector, pixelSet); // put them in the variance vector
    }

    // loop over x,y ... the loop over the stack to fill pixelSet
    // - get the stats on pixelSet and put the value in the output image at x,y
    for (int y = 0; y != imgStack->getHeight(); ++y) {

        for (unsigned int i = 0; i < images.size(); ++i) {
            x_iterator ptr = images[i]->row_begin(y);
            if (y == 0) {
                rows.push_back(ptr);
            } else {
                rows[i] = ptr;
            }
        }

        for (x_iterator ptr = imgStack->row_begin(y), end = imgStack->row_end(y); ptr != end; ++ptr) {
            typename afwMath::MaskedVector<PixelT>::iterator psPtr = pixelSet.begin();
            afwImage::MaskPixel msk(0x0);
            for (unsigned int i = 0; i < images.size(); ++i, ++psPtr) {
                afwImage::MaskPixel mskTmp = rows[i].mask();
                psPtr.value() = rows[i].image();
                psPtr.mask()  = mskTmp;

                // if we're not using the wvector weights, use the variance plane.
                if (! UseVariance) {
                    psPtr.variance() = rows[i].variance();
                }

                ++rows[i];
            }
            afwMath::Statistics stat =
                afwMath::makeStatistics(pixelSet, flags | afwMath::NPOINT | afwMath::ERRORS, sctrlTmp);
            
            PixelT variance = ::pow(stat.getError(flags), 2);
            msk = stat.getOrMask();
            if (stat.getValue(afwMath::NPOINT) == 0) {
                msk = sctrlTmp.getNoGoodPixelsMask();
            }
            *ptr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(flags), msk, variance);
        }
    }

    return imgStack;

}
    
} // end anonymous namespace


/**
 * @brief A function to compute some statistics of a stack of Masked Images
 * @relates Statistics
 *
 * If none of the input images are valid for some pixel,
 * the afwMath::StatisticsControl::getNoGoodPixelsMask() bit(s) are set.
 *
 * All the work is done in the function comuteMaskedImageStack.
 * A boolean template variable has been used to allow the compiler to generate the different instantiations
 *   to handle cases when we are, or are not, dealing with the variance plane.
 */
template<typename PixelT>
typename afwImage::MaskedImage<PixelT>::Ptr afwMath::statisticsStack(
        std::vector<typename afwImage::MaskedImage<PixelT>::Ptr > &images, //!< images to process
        afwMath::Property flags,                                           //!< Desired statistic (only one!)
        afwMath::StatisticsControl const& sctrl,                           //!< Fine control over processing
        std::vector<PixelT> const &wvector                                 //!< optional weights vector
                                                              ) {

    checkOnlyOneFlag(flags);

    // if we're going to use constant weights 
    if ( wvector.size() == images.size() ) {
        return computeMaskedImageStack<PixelT, true>(images, flags, sctrl, wvector);
        
    // if we're weighting by the pixel variance        
    } else if ( wvector.size() == 0 ) {
        return computeMaskedImageStack<PixelT, false>(images, flags, sctrl, wvector);
        
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


namespace {
/***********************************************************************************/
/*
 * A function to compute some statistics of a stack of regular images
 */
template<typename PixelT, bool UseVariance>
typename afwImage::Image<PixelT>::Ptr computeImageStack(
        std::vector<typename afwImage::Image<PixelT>::Ptr > &images,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                        ) {

    // create the image to be returned
    typedef afwImage::Image<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions(), 0.0));

    afwMath::StatisticsControl sctrlTmp(sctrl);
    afwMath::MaskedVector<typename Image::Pixel> pixelSet(images.size());

    // set the mask to be an infinite iterator
    afwMath::MaskImposter<afwImage::MaskPixel> msk;

    // if we're going to use contant weights
    if ( UseVariance ) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        // copy the weights in to the variance vector
        loadVariance(wvector, pixelSet);
    }
        
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                (*pixelSet.getImage())(i, 0) = (*images[i])(x, y);
            }
            if (UseVariance) {
                afwMath::Statistics stat =
                    afwMath::makeStatistics(*pixelSet.getImage(), msk, *pixelSet.getVariance(),
                                         flags, sctrlTmp);
                (*imgStack)(x, y) = stat.getValue();
            } else {
                afwMath::Statistics stat = afwMath::makeStatistics(pixelSet, flags, sctrlTmp);
                (*imgStack)(x, y) = stat.getValue();
            }
        }
    }

    return imgStack;
}

} // end anonymous namespace


/**
 * @brief A function to compute some statistics of a stack of regular images
 * @relates Statistics
 */
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr afwMath::statisticsStack(
        std::vector<typename afwImage::Image<PixelT>::Ptr > &images,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                        ) {

    checkOnlyOneFlag(flags);

    // if we're going to use contant weights
    if ( wvector.size() == images.size() ) {
        return computeImageStack<PixelT, true>(images, flags, sctrl, wvector);
        
    } else if ( wvector.size() == 0 ) {
        return computeImageStack<PixelT, false>(images, flags, sctrl, wvector);

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

namespace {

/**********************************************************************************/
/*
 * A function to compute some statistics of a stack of vectors
 */
template<typename PixelT, bool UseVariance>
typename boost::shared_ptr<std::vector<PixelT> > computeVectorStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                                      ) {

    // create the image to be returned
    typedef std::vector<PixelT> Vect;
    typename boost::shared_ptr<Vect> vecStack(new Vect(vectors[0]->size(), 0.0));

    afwMath::MaskedVector<PixelT> pixelSet(vectors.size()); // values from a given pixel of each image

    afwMath::StatisticsControl sctrlTmp(sctrl);
    // set the mask to be an infinite iterator
    afwMath::MaskImposter<image::MaskPixel> msk;
    
    // Use constant weights for each layer
    if ( UseVariance ) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
        // copy the weights in to the variance vector
        loadVariance(wvector, pixelSet);
    }
    
    // collect elements from the stack into the MaskedVector to do stats
    for (unsigned int x = 0; x < vectors[0]->size(); ++x) {
        typename afwMath::MaskedVector<PixelT>::iterator psPtr = pixelSet.begin();
        for (unsigned int i = 0; i < vectors.size(); ++i, ++psPtr) {
            psPtr.value() = (*vectors[i])[x];
        }
        if (UseVariance) {
            afwMath::Statistics stat = afwMath::makeStatistics(*pixelSet.getImage(), msk,
                                                               *pixelSet.getVariance(),
                                                               flags, sctrlTmp);
            (*vecStack)[x] = stat.getValue(flags);
        } else {
            afwMath::Statistics stat = afwMath::makeStatistics(pixelSet, flags, sctrlTmp);
            (*vecStack)[x] = stat.getValue(flags);
        }
    }

    return vecStack;
}

} // end anonymous namespace


/**
 * @brief A function to handle stacking a vector of vectors
 * @relates Statistics
 *
 * All the work is done in the function comuteVectorStack.
 * A boolean template variable has been used to allow the compiler to generate the different instantiations
 *   to handle cases when we are, or are not, dealing with the variance plane.
 */
template<typename PixelT>
typename boost::shared_ptr<std::vector<PixelT> > afwMath::statisticsStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        std::vector<PixelT> const &wvector
                                                                      ) {

    checkOnlyOneFlag(flags);

    // Use constant weights for each layer
    if ( wvector.size() == vectors.size() ) {
        return computeVectorStack<PixelT, true>(vectors, flags, sctrl, wvector);
    // Use no weights.
    } else if ( wvector.size() == 0 ) {
        return computeVectorStack<PixelT, false>(vectors, flags, sctrl, wvector);
    // Fail if the number weights isn't the same as the number of vectors to be weighted.
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Weight vector must have same length as number of vectors to be stacked.");
    }
}



/**************************************************************************
 *
 * XY row column stacking
 *
 **************************************************************************/



/**
 * @brief A function to collapse a maskedImage to a one column image
 * @relates Statistics
 *
 *
 */
template<typename PixelT>
typename afwImage::MaskedImage<PixelT>::Ptr afwMath::statisticsStack(
        afwImage::Image<PixelT> const &image,  
        afwMath::Property flags,               
        char dimension,
        afwMath::StatisticsControl const& sctrl
                                                                 ) {


    int x0 = image.getX0();
    int y0 = image.getY0();
    typedef afwImage::MaskedImage<PixelT> MImage;
    typename MImage::Ptr imgOut;

    // do each row or column, one at a time
    // - create a subimage with a bounding box, and get the stats and assign the value to the output image
    if (dimension == 'x') {
        imgOut = typename MImage::Ptr(new MImage(1, image.getHeight()));
        int y = y0;
        typename MImage::y_iterator oEnd = imgOut->col_end(0);
        for (typename MImage::y_iterator oPtr = imgOut->col_begin(0); oPtr != oEnd; ++oPtr, ++y) {
            afwImage::BBox bbox = afwImage::BBox(afwImage::PointI(x0, y), image.getWidth(), 1);
            afwImage::Image<PixelT> subImage(image, bbox);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }

    } else if (dimension == 'y') {
        imgOut = typename MImage::Ptr(new MImage(image.getWidth(), 1));
        int x = x0;
        typename MImage::x_iterator oEnd = imgOut->row_end(0);
        for (typename MImage::x_iterator oPtr = imgOut->row_begin(0); oPtr != oEnd; ++oPtr, ++x) {
            afwImage::BBox bbox = afwImage::BBox(afwImage::PointI(x, y0), 1, image.getHeight());
            afwImage::Image<PixelT> subImage(image, bbox);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Can only run statisticsStack in x or y for single image.");
    }

    return imgOut;
}

/**
 * @brief A function to collapse a maskedImage to a one column image
 * @relates Statistics
 *
 *
 */
template<typename PixelT>
typename afwImage::MaskedImage<PixelT>::Ptr afwMath::statisticsStack(
        afwImage::MaskedImage<PixelT> const &image,  
        afwMath::Property flags,               
        char dimension,
        afwMath::StatisticsControl const& sctrl
                                                                 ) {


    int x0 = image.getX0();
    int y0 = image.getY0();
    typedef afwImage::MaskedImage<PixelT> MImage;
    typename MImage::Ptr imgOut;

    // do each row or column, one at a time
    // - create a subimage with a bounding box, and get the stats and assign the value to the output image
    if (dimension == 'x') {
        imgOut = typename MImage::Ptr(new MImage(1, image.getHeight()));
        int y = 0;
        typename MImage::y_iterator oEnd = imgOut->col_end(0);
        for (typename MImage::y_iterator oPtr = imgOut->col_begin(0); oPtr != oEnd; ++oPtr, ++y) {
            afwImage::BBox bbox = afwImage::BBox(afwImage::PointI(x0, y), image.getWidth(), 1);
            afwImage::MaskedImage<PixelT> subImage(image, bbox);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }

    } else if (dimension == 'y') {
        imgOut = typename MImage::Ptr(new MImage(image.getWidth(), 1));
        int x = 0;
        typename MImage::x_iterator oEnd = imgOut->row_end(0);
        for (typename MImage::x_iterator oPtr = imgOut->row_begin(0); oPtr != oEnd; ++oPtr, ++x) {
            afwImage::BBox bbox = afwImage::BBox(afwImage::PointI(x, y0), 1, image.getHeight());
            afwImage::MaskedImage<PixelT> subImage(image, bbox);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException,
                          "Can only run statisticsStack in x or y for single image.");
    }

    return imgOut;
}






/*
 * Explicit Instantiations
 *
 */
#define INSTANTIATE_STACKS(TYPE) \
    template afwImage::Image<TYPE>::Ptr afwMath::statisticsStack<TYPE>( \
            std::vector<afwImage::Image<TYPE>::Ptr > &images, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector);                          \
    template afwImage::MaskedImage<TYPE>::Ptr afwMath::statisticsStack<TYPE>( \
            std::vector<afwImage::MaskedImage<TYPE>::Ptr > &images, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector);                          \
    template boost::shared_ptr<std::vector<TYPE> > afwMath::statisticsStack<TYPE>( \
            std::vector<boost::shared_ptr<std::vector<TYPE> > > &vectors, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            std::vector<TYPE> const &wvector); \
    template afwImage::MaskedImage<TYPE>::Ptr afwMath::statisticsStack( \
            afwImage::Image<TYPE> const &image, \
            afwMath::Property flags,                    \
            char dimension,                                     \
            afwMath::StatisticsControl const& sctrl); \
    template afwImage::MaskedImage<TYPE>::Ptr afwMath::statisticsStack( \
            afwImage::MaskedImage<TYPE> const &image, \
            afwMath::Property flags,                    \
            char dimension,                                     \
            afwMath::StatisticsControl const& sctrl);

INSTANTIATE_STACKS(double)
INSTANTIATE_STACKS(float)
