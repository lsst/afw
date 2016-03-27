// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
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

/********************************************************************
 *
 * z stacks
 *
 *********************************************************************/

/**
 * @brief A function to compute some statistics of a stack of Images
 */
template<typename PixelT>
typename lsst::afw::image::Image<PixelT>::Ptr statisticsStack(
        std::vector<typename lsst::afw::image::Image<PixelT>::Ptr > &images,      ///< Images to process
        Property flags, ///< statistics requested
        StatisticsControl const& sctrl=StatisticsControl(),   ///< Control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector=std::vector<lsst::afw::image::VariancePixel>(0) ///< vector containing weights
                                                             );

/**
 * @ brief compute statistical stack of Image.  Write to output image in-situ
 */
template<typename PixelT>
void statisticsStack(
    lsst::afw::image::Image<PixelT>& out, ///< Output image
    std::vector<typename lsst::afw::image::Image<PixelT>::Ptr > &images,      ///< Images to process
    Property flags, ///< statistics requested
    StatisticsControl const& sctrl=StatisticsControl(),   ///< Control structure
    std::vector<lsst::afw::image::VariancePixel> const& wvector=
        std::vector<lsst::afw::image::VariancePixel>(0) ///< vector containing weights
    );

/**
 * @brief A function to compute some statistics of a stack of MaskedImages
 */
template<typename PixelT>
typename lsst::afw::image::MaskedImage<PixelT>::Ptr statisticsStack(
        std::vector<typename lsst::afw::image::MaskedImage<PixelT>::Ptr > &images,///< MaskedImages to process
        Property flags, ///< statistics requested
        StatisticsControl const& sctrl=StatisticsControl(), ///< control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector=std::vector<lsst::afw::image::VariancePixel>(0) ///< vector containing weights
                                                                   );
    
/**
 * @ brief compute statistical stack of MaskedImage.  Write to output image in-situ
 */
template<typename PixelT>
void statisticsStack(
    lsst::afw::image::MaskedImage<PixelT>& out, ///< Output image
    std::vector<typename lsst::afw::image::MaskedImage<PixelT>::Ptr > &images,///< MaskedImages to process
    Property flags, ///< statistics requested
    StatisticsControl const& sctrl=StatisticsControl(), ///< control structure
    std::vector<lsst::afw::image::VariancePixel> const& wvector=
        std::vector<lsst::afw::image::VariancePixel>(0) ///< vector containing weights
    );


/**
 * @brief A function to compute some statistics of a stack of std::vectors
 */
template<typename PixelT>
boost::shared_ptr<std::vector<PixelT> > statisticsStack(
        std::vector<boost::shared_ptr<std::vector<PixelT> > > &vectors,      ///< Vectors to process
        Property flags,              ///< statistics requested
        StatisticsControl const& sctrl=StatisticsControl(),  ///< control structure
        std::vector<lsst::afw::image::VariancePixel> const& wvector=std::vector<lsst::afw::image::VariancePixel>(0) ///< vector containing weights
                                                                );
    


/********************************************************************
 *
 * x,y stacks
 *
 *********************************************************************/

/**
 * @brief A function to compute statistics on the rows or columns of an image
 */
template<typename PixelT>
typename lsst::afw::image::MaskedImage<PixelT>::Ptr statisticsStack(
        lsst::afw::image::Image<PixelT> const &image,  
        Property flags,               
        char dimension,
        StatisticsControl const& sctrl=StatisticsControl()
                                                                   );
/**
 * @brief A function to compute statistics on the rows or columns of an image
 */
template<typename PixelT>
typename lsst::afw::image::MaskedImage<PixelT>::Ptr statisticsStack(
        lsst::afw::image::MaskedImage<PixelT> const &image,  
        Property flags,               
        char dimension,
        StatisticsControl const& sctrl=StatisticsControl()
								    );




}}}

#endif
