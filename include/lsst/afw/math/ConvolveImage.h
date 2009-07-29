// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_CONVOLVEIMAGE_H
#define LSST_AFW_MATH_CONVOLVEIMAGE_H
/**
 * @file
 *
 * @brief Convolve and convolveAtAPoint functions for Image and Kernel
 *
 * @todo Consider adding a flag to convolve indicating which specialized version of basicConvolve was used.
 *   This would only be used for unit testing and trace messages suffice (barely), so not a high priority.
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <limits>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
        typename InImageT::const_xy_locator& inLocator,
        typename lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator& kernelLocator,
        int kWidth, int kHeight);
    
    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
        typename InImageT::const_xy_locator& inImage,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelColList,
        std::vector<lsst::afw::math::Kernel::PixelT> const& kernelRowList
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::Kernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::DeltaFunctionKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::LinearCombinationKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        lsst::afw::math::SeparableKernel const& kernel,
        bool doNormalize
    );
    
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
        OutImageT& convolvedImage,
        InImageT const& inImage,
        KernelT const& kernel,
        bool doNormalize,
        bool copyEdge = false
    );
    
    /**
     * \brief Return an edge pixel appropriate for a given Image type
     *
     * The value is quiet_NaN if that exists for the pixel type, else 0
     */
    template <typename ImageT>
    typename ImageT::SinglePixel edgePixel(
        lsst::afw::image::detail::Image_tag ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()
    ) {
        typedef typename ImageT::SinglePixel SinglePixelT;
        return SinglePixelT(
            std::numeric_limits<SinglePixelT>::has_quiet_NaN ?
                std::numeric_limits<SinglePixelT>::quiet_NaN() : 0);
    };
    
    /**
     * \brief Return an edge pixel appropriate for a given MaskedImage type
     *
     * The components are:
     * - %image = quiet_NaN if that exists for the pixel type, else 0
     * - mask = EDGE bit set
     * - variance = infinity
     */
    template <typename MaskedImageT>
    typename MaskedImageT::SinglePixel edgePixel(
        lsst::afw::image::detail::MaskedImage_tag   ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()
    ) {
        typedef typename MaskedImageT::Image::Pixel ImagePixelT;
        typedef typename MaskedImageT::Variance::Pixel VariancePixelT;
        
        return typename MaskedImageT::SinglePixel(
            std::numeric_limits<ImagePixelT>::has_quiet_NaN ?
                std::numeric_limits<ImagePixelT>::quiet_NaN() : 0,
            MaskedImageT::Mask::getPlaneBitMask("EDGE"),
            std::numeric_limits<VariancePixelT>::infinity());
    };
}}}   // lsst::afw::math

/**
 * @brief Apply convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
    typename InImageT::const_xy_locator& imageLocator, ///< locator for %image pixel that overlaps
        ///< pixel (0,0) of kernel (the origin of the kernel, not its center)
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT>::const_xy_locator &kernelLocator,
                    ///< locator for (0,0) pixel of kernel (the origin of the kernel, not its center)
    int kWidth,     ///< number of columns in kernel
    int kHeight     ///< number of rows in kernel
                                  ) {
    typename OutImageT::SinglePixel outValue = 0;
    for (int y = 0; y != kHeight; ++y) {
        for (int x = 0; x != kWidth; ++x, ++imageLocator.x(), ++kernelLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kVal = kernelLocator[0];
            if (kVal != 0) {
                outValue += *imageLocator*kVal;
            }
        }

        imageLocator  += lsst::afw::image::detail::difference_type(-kWidth, 1);
        kernelLocator += lsst::afw::image::detail::difference_type(-kWidth, 1);
    }

    imageLocator  += lsst::afw::image::detail::difference_type(0, -kHeight);
    kernelLocator += lsst::afw::image::detail::difference_type(0, -kHeight);

    return outValue;
}

/**
 * @brief Apply separable convolution kernel to an %image at one point
 *
 * @note This subroutine sacrifices convenience for performance. The user is expected to figure out
 * the kernel center and adjust the supplied pixel accessors accordingly.
 * For an example of how to do this see convolve().
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
inline typename OutImageT::SinglePixel lsst::afw::math::convolveAtAPoint(
    typename InImageT::const_xy_locator& imageLocator,
                                        ///< locator for %image pixel that overlaps (0,0) pixel of kernel(!)
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelXList,  ///< kernel column vector
    std::vector<lsst::afw::math::Kernel::PixelT> const &kernelYList   ///< kernel row vector
) {
    typedef typename std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator k_iter;

    std::vector<lsst::afw::math::Kernel::PixelT>::const_iterator kernelYIter = kernelYList.begin();

    typedef typename OutImageT::SinglePixel OutT;
    OutT outValue = 0;
    for (k_iter kernelYIter = kernelYList.begin(), end = kernelYList.end();
         kernelYIter != end; ++kernelYIter) {

        OutT outValueY = 0;
        for (k_iter kernelXIter = kernelXList.begin(), end = kernelXList.end();
             kernelXIter != end; ++kernelXIter, ++imageLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kValX = *kernelXIter;
            if (kValX != 0) {
                outValueY += *imageLocator*kValX;
            }
        }
        
        double const kValY = *kernelYIter;
        if (kValY != 0) {
            outValue += outValueY*kValY;
        }
        
        imageLocator += lsst::afw::image::detail::difference_type(-kernelXList.size(), 1);
    }
    
    imageLocator += lsst::afw::image::detail::difference_type(0, -kernelYList.size());

    return outValue;
}

#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)
