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
#include <cassert>
#include "boost/shared_ptr.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Stack.h"
#include "lsst/afw/math/MaskedVector.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath  = lsst::afw::math;
namespace pexExcept = lsst::pex::exceptions;


namespace {
    typedef std::vector<lsst::afw::image::VariancePixel> WeightVector; // vector of weights (yes, really)

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
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
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
    
template<typename PixelT, bool UseWeights>
typename afwImage::MaskedImage<PixelT>::Ptr computeMaskedImageStack(
                             std::vector<typename afwImage::MaskedImage<PixelT>::Ptr > const &images,
                             afwMath::Property flags,               
                             afwMath::StatisticsControl const& sctrl,
                             WeightVector const &wvector
                                                          )
{
    // create the image to be returned
    typedef afwImage::MaskedImage<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions()));

    // get a list of row_begin iterators
    typedef typename afwImage::MaskedImage<PixelT>::x_iterator x_iterator;
    std::vector<x_iterator> rows;
    rows.reserve(images.size());

    afwMath::MaskedVector<PixelT> pixelSet(images.size()); // a pixel from x,y for each image
    WeightVector weights;                                  // weights; non-const version
    //
    afwMath::StatisticsControl sctrlTmp(sctrl);

    // If we're using wvector, they're multiplicative weights
    if (UseWeights) {
        weights.assign(wvector.begin(), wvector.end());

        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
    } else {                                // use the variance plane if we weight at all
        weights.resize(images.size());

        sctrlTmp.setMultiplyWeights(false); // n.b. we don't turn on weighting, just ensure inverse-variances
    }
    assert (weights.size() == images.size());

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
            WeightVector::iterator wtPtr = weights.begin();
            for (unsigned int i = 0; i < images.size(); ++i, ++psPtr, ++wtPtr) {
                afwImage::MaskPixel mskTmp = rows[i].mask();
                psPtr.value() = rows[i].image();
                psPtr.mask()  = mskTmp;
                psPtr.variance() = rows[i].variance();

                // if we're not using the wvector weights, use the variance plane as weights
                if (!UseWeights) {      // should check if sctrl wants to use weights before loading variance
                    *wtPtr = rows[i].variance();
                }

                ++rows[i];
            }
            afwMath::Statistics stat =
                afwMath::makeStatistics(pixelSet, weights,
                                        flags | afwMath::NPOINT | afwMath::ERRORS, sctrlTmp);
            
            PixelT variance = ::pow(stat.getError(flags), 2);
            msk = stat.getOrMask();
            int const npoint = stat.getValue(afwMath::NPOINT);
            if (npoint == 0) {
                msk = sctrlTmp.getNoGoodPixelsMask();
            } else if (npoint == 1) {   // the population variance is NaN (we divided by N - 1)
                assert(lsst::utils::isnan(variance));
                int ngood = 0;          // good (based on mask checks)
                for (unsigned int i = 0; i < images.size(); ++i) {
                    x_iterator ptr = rows[i]; ptr += -1;
                    if (!(ptr.mask() & sctrl.getAndMask())) {
                        ++ngood;
                        variance = ptr.variance();
                    }
                }
#if 0
                assert(ngood == 1);     // we don't handle the case that images were clipped so ngood > npoint
#else
                if (ngood != 1) {
                    assert(ngood > 1);
                    std::cerr << "ngood = " << ngood << " complain to RHL" << std::endl;
                }
#endif
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
        WeightVector const &wvector        //!< optional weights vector
                                                              ) {
    if (images.size() == 0) {
        throw LSST_EXCEPT(pexExcept::LengthErrorException, "Please specify at least one object to stack");
    }

    checkOnlyOneFlag(flags);

    // if we're going to use constant weights 
    if ( wvector.size() == images.size() ) {
        return computeMaskedImageStack<PixelT, true>(images, flags, sctrl, wvector);
        
    // if we're weighting by the pixel variance        
    } else if ( wvector.size() == 0 ) {
        return computeMaskedImageStack<PixelT, false>(images, flags, sctrl, wvector);
        
    // Fail if the number weights isn't the same as the number of images to be weighted.
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
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
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr computeImageStack(
        std::vector<typename afwImage::Image<PixelT>::Ptr > &images,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        WeightVector const &weights
                                                        )
{
    // create the image to be returned
    typedef afwImage::Image<PixelT> Image;
    typename Image::Ptr imgStack(new Image(images[0]->getDimensions(), 0.0));

    afwMath::MaskedVector<typename Image::Pixel> pixelSet(images.size()); // a pixel from x,y for each image
    afwMath::StatisticsControl sctrlTmp(sctrl);

    // set the mask to be an infinite iterator
    afwMath::MaskImposter<afwImage::MaskPixel> msk;

    // If we're using wvector, they're multiplicative weights
    if (!weights.empty()) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
    }
        
    // get the desired statistic
    for (int y = 0; y != imgStack->getHeight(); ++y) {
        for (int x = 0; x != imgStack->getWidth(); ++x) {
            for (unsigned int i = 0; i != images.size(); ++i) {
                (*pixelSet.getImage())(i, 0) = (*images[i])(x, y);
            }
            
            (*imgStack)(x, y) = afwMath::makeStatistics(pixelSet, weights, flags, sctrlTmp).getValue();
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
        WeightVector const &wvector
                                                        ) {
    checkOnlyOneFlag(flags);

    if (!wvector.empty() && wvector.size() != images.size() ) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          str(boost::format("Weight vector has different length from number of objects to be stacked: %d v. %d") % wvector.size() % images.size()));
    }
    
    return computeImageStack<PixelT>(images, flags, sctrl, wvector);
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
template<typename PixelT>
typename boost::shared_ptr<std::vector<PixelT> > computeVectorStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        WeightVector const &wvector
                                                                      )
{
    // create the image to be returned
    typedef std::vector<PixelT> Vect;
    boost::shared_ptr<Vect> vecStack(new Vect(vectors[0]->size(), 0.0));

    afwMath::MaskedVector<PixelT> pixelSet(vectors.size()); // values from a given pixel of each image

    afwMath::StatisticsControl sctrlTmp(sctrl);
    // set the mask to be an infinite iterator
    afwMath::MaskImposter<afwImage::MaskPixel> msk;
    
    // If we're using wvector, they're multiplicative weights
    if (!wvector.empty()) {
        sctrlTmp.setWeighted(true);
        sctrlTmp.setMultiplyWeights(true);
    }
    
    // collect elements from the stack into the MaskedVector to do stats
    for (unsigned int x = 0; x < vectors[0]->size(); ++x) {
        typename afwMath::MaskedVector<PixelT>::iterator psPtr = pixelSet.begin();
        for (unsigned int i = 0; i < vectors.size(); ++i, ++psPtr) {
            psPtr.value() = (*vectors[i])[x];
        }

        (*vecStack)[x] = afwMath::makeStatistics(pixelSet, wvector, flags, sctrlTmp).getValue(flags);
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
boost::shared_ptr<std::vector<PixelT> > afwMath::statisticsStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,  
        afwMath::Property flags,               
        afwMath::StatisticsControl const& sctrl,
        WeightVector const &wvector
                                                                      )
{
    checkOnlyOneFlag(flags);

    if (!wvector.empty() && wvector.size() != vectors.size() ) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          str(boost::format("Weight vector has different length from number of objects to be stacked: %d v. %d") % wvector.size() % vectors.size()));
    }

    return computeVectorStack<PixelT>(vectors, flags, sctrl, wvector);
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
        imgOut = typename MImage::Ptr(
            new MImage(geom::Extent2I(1, image.getHeight()))
        );
        int y = y0;
        typename MImage::y_iterator oEnd = imgOut->col_end(0);
        for (typename MImage::y_iterator oPtr = imgOut->col_begin(0); oPtr != oEnd; ++oPtr, ++y) {
            geom::Box2I bbox = afwGeom::Box2I(
                geom::Point2I(x0, y), 
                geom::Extent2I(image.getWidth(), 1)
            );
            afwImage::Image<PixelT> subImage(image, bbox, afwImage::PARENT);
            Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(
                stat.getValue(), 0x0, 
                stat.getError()*stat.getError()
            );
        }

    } else if (dimension == 'y') {
        imgOut = typename MImage::Ptr(new MImage(geom::Extent2I(image.getWidth(), 1)));
        int x = x0;
        typename MImage::x_iterator oEnd = imgOut->row_end(0);
        for (typename MImage::x_iterator oPtr = imgOut->row_begin(0); oPtr != oEnd; ++oPtr, ++x) {
            geom::Box2I bbox = geom::Box2I(
                geom::Point2I(x, y0), geom::Extent2I(1, image.getHeight())
            );
            afwImage::Image<PixelT> subImage(image, bbox, afwImage::PARENT);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
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
                                                                 )
{
    int const x0 = image.getX0();
    int const y0 = image.getY0();
    typedef afwImage::MaskedImage<PixelT> MImage;
    typename MImage::Ptr imgOut;

    // do each row or column, one at a time
    // - create a subimage with a bounding box, and get the stats and assign the value to the output image
    if (dimension == 'x') {
        imgOut = typename MImage::Ptr(new MImage(geom::Extent2I(1, image.getHeight())));
        int y = 0;
        typename MImage::y_iterator oEnd = imgOut->col_end(0);
        for (typename MImage::y_iterator oPtr = imgOut->col_begin(0); oPtr != oEnd; ++oPtr, ++y) {
            afwGeom::Box2I bbox = afwGeom::Box2I(afwGeom::Point2I(x0, y), geom::Extent2I(image.getWidth(), 1));
            afwImage::MaskedImage<PixelT> subImage(image, bbox, afwImage::PARENT);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }

    } else if (dimension == 'y') {
        imgOut = typename MImage::Ptr(new MImage(geom::Extent2I(image.getWidth(), 1)));
        int x = 0;
        typename MImage::x_iterator oEnd = imgOut->row_end(0);
        for (typename MImage::x_iterator oPtr = imgOut->row_begin(0); oPtr != oEnd; ++oPtr, ++x) {
            afwGeom::Box2I bbox = afwGeom::Box2I(afwGeom::Point2I(x, y0), geom::Extent2I(1, image.getHeight()));
            afwImage::MaskedImage<PixelT> subImage(image, bbox, afwImage::PARENT);
            afwMath::Statistics stat = makeStatistics(subImage, flags | afwMath::ERRORS, sctrl);
            *oPtr = typename afwImage::MaskedImage<PixelT>::Pixel(stat.getValue(), 0x0, 
                                                                  stat.getError()*stat.getError());
        }
    } else {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          "Can only run statisticsStack in x or y for single image.");
    }

    return imgOut;
}

/************************************************************************************************************/
/*
 * Explicit Instantiations
 *
 */
/// \cond
#define INSTANTIATE_STACKS(TYPE) \
    template afwImage::Image<TYPE>::Ptr afwMath::statisticsStack<TYPE>( \
            std::vector<afwImage::Image<TYPE>::Ptr > &images, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            WeightVector const &wvector);                          \
    template afwImage::MaskedImage<TYPE>::Ptr afwMath::statisticsStack<TYPE>( \
            std::vector<afwImage::MaskedImage<TYPE>::Ptr > &images, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            WeightVector const &wvector);                          \
    template boost::shared_ptr<std::vector<TYPE> > afwMath::statisticsStack<TYPE>( \
            std::vector<boost::shared_ptr<std::vector<TYPE> > > &vectors, \
            afwMath::Property flags, \
            afwMath::StatisticsControl const& sctrl,    \
            WeightVector const &wvector);                               \
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
/// \endcond
