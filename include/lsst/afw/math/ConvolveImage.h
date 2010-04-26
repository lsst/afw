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

#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

    /**
     * A collection of Kernel images for special locations on a rectangular region of an image
     *
     * See the Location enum for a list of those special locations.
     *
     * This is a low-level helper class for recursive convolving with interpolation. Many of these objects
     * may be created during a convolution, and many will share kernel images. It uses shared pointers
     * to kernels and kernel images for increased speed and decreased memory usage (at the expense of safety).
     * Note that null pointers are NOT acceptable for the constructors!
     *
     * Also note that it uses lazy evaluation: images are computed when they are wanted.
     */
    class KernelImagesForRegion :
        public lsst::daf::data::LsstBase,
        public lsst::daf::base::Persistable
    {
    public:
        typedef lsst::afw::math::Kernel::ConstPtr KernelConstPtr;
        typedef lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel> Image;
        typedef Image::Ptr ImagePtr;
        typedef Image::ConstPtr ImageConstPtr;
        typedef std::vector<KernelImagesForRegion> List;
        /**
         * locations of various points in the region
         *
         * The corners posiitions are: BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
         * The "middle" positions are the middle of each side, plus the center of the region:
         *    BOTTOM, TOP, LEFT, RIGHT, CENTER
         *
         * These positions always refer to an exact pixel. If the region has an even size along an axis
         * then the middle is shifted by 1/2 pixel (in an unspecified direction) for that axis.
         */
        enum Location {
            BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT,
            BOTTOM, TOP, LEFT, RIGHT, CENTER
        };
    
        KernelImagesForRegion(
                KernelConstPtr kernelPtr,
                lsst::afw::geom::BoxI const &bbox,
                bool doNormalize);
        KernelImagesForRegion(
                KernelConstPtr kernelPtr,
                lsst::afw::geom::BoxI const &bbox,
                bool doNormalize,
                ImageConstPtr bottomLeftImagePtr,
                ImageConstPtr bottomRightImagePtr,
                ImageConstPtr topLeftImagePtr,
                ImageConstPtr topRightImagePtr);

        lsst::afw::geom::BoxI getBBox() const { return _bbox; };
        bool getDoNormalize() const { return _doNormalize; };
        ImageConstPtr getImage(Location location) const;
        KernelConstPtr getKernel() const { return _kernelPtr; };
        std::vector<KernelImagesForRegion> getSubregions() const;
        std::vector<KernelImagesForRegion> getSubregions(int nx, int ny) const;
        bool isInterpolationOk(double tolerance) const;
        static int getMinInterpSize() { return _MinInterpSize; };

    private:
        typedef std::map<Location, ImageConstPtr> ImageMap;
        typedef std::vector<Location> LocationList;

        inline void _insertImage(Location location, ImageConstPtr &imagePtr) const;
        void _interpolateImage(Image &outImage, Location location1) const;
        lsst::afw::geom::Point2I _pixelIndexFromLocation(Location) const;
        
        // static helper functions
        static lsst::afw::geom::Point2D _computeCenterFractionalPosition(lsst::afw::geom::BoxI const &bbox);
        static lsst::afw::geom::Point2I _computeCenterIndex(lsst::afw::geom::BoxI const &bbox);
        static inline int _computeNextSubregionLength(int length, int nDivisions);
        static std::vector<int> _computeSubregionLengths(int length, int nDivisions);
        
        // member variables
        KernelConstPtr _kernelPtr;
        lsst::afw::geom::BoxI _bbox;
        lsst::afw::geom::Point2D _centerFractionalPosition;  ///< fractional position of center pixel
            ///< from bottom left to top right; 0.5 if length of axis is odd, somewhat less if even
        lsst::afw::geom::Point2I _centerIndex;  ///< index of center pixel
        bool _doNormalize;
        mutable ImageMap _imageMap; ///< cache of location:kernel image;
            ///< mutable to support lazy evaluation: const methods may add entries to the cache

        static int const _MinInterpSize;
        static LocationList const _TestLocationList;   ///< locations at which to test
            ///< linear interpolation to see if it is accurate enough
    };
    
    template <typename OutImageT, typename InImageT>
    void convolveWithInterpolation(
            OutImageT &outImage,
            InImageT const &inImage,
            lsst::afw::math::Kernel const &kernel,
            bool doNormalize,
            double tolerance = 1.0e-5,
            int maxInterpolationDistance = 50);

    template <typename OutImageT, typename InImageT>
    void convolveRegionWithRecursiveInterpolation(
            OutImageT &outImage,
            InImageT const &inImage,
            KernelImagesForRegion const &region,
            double tolerance = 1.0e-5);
    
    template <typename OutImageT, typename InImageT>
    void convolveRegionWithInterpolation(
            OutImageT &outImage,
            InImageT const &inImage,
            KernelImagesForRegion const &region);

    template <typename OutImageT, typename InImageT>
    void convolveWithBruteForce(
            OutImageT &convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::Kernel const& kernel,
            bool doNormalize);
}   // detail

    template <typename OutImageT, typename InImageT>
    void scaledPlus(
            OutImageT &outImage,
            double c1,
            InImageT const &inImage1,
            double c2,
            InImageT const &inImage2);

    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
            typename InImageT::const_xy_locator inImageLocator,
            typename lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kernelLocator,
            int kWidth,
            int kHeight);
    
    template <typename OutImageT, typename InImageT>
    inline typename OutImageT::SinglePixel convolveAtAPoint(
            typename InImageT::const_xy_locator inImageLocator,
            std::vector<lsst::afw::math::Kernel::Pixel> const& kernelColList,
            std::vector<lsst::afw::math::Kernel::Pixel> const& kernelRowList);
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::Kernel const& kernel,
            bool doNormalize);
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::DeltaFunctionKernel const& kernel,
            bool doNormalize);
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::LinearCombinationKernel const& kernel,
            bool doNormalize);
    
    template <typename OutImageT, typename InImageT>
    void basicConvolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::SeparableKernel const& kernel,
            bool doNormalize);
    
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            bool doNormalize,
            bool copyEdge = false);
    
    /**
     * \brief Return an edge pixel appropriate for a given Image type
     *
     * The value is quiet_NaN if that exists for the pixel type, else 0
     */
    template <typename ImageT>
    typename ImageT::SinglePixel edgePixel(
            lsst::afw::image::detail::Image_tag
                ///< lsst::afw::image::detail::image_traits<ImageT>::image_category()
    ) {
        typedef typename ImageT::SinglePixel SinglePixelT;
        return SinglePixelT(
            std::numeric_limits<SinglePixelT>::has_quiet_NaN ?
                std::numeric_limits<SinglePixelT>::quiet_NaN() : 0);
    }
    
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
            lsst::afw::image::detail::MaskedImage_tag
            ///< lsst::afw::image::detail::image_traits<MaskedImageT>::image_category()
    ) {
        typedef typename MaskedImageT::Image::Pixel ImagePixelT;
        typedef typename MaskedImageT::Variance::Pixel VariancePixelT;
        
        return typename MaskedImageT::SinglePixel(
            std::numeric_limits<ImagePixelT>::has_quiet_NaN ?
                std::numeric_limits<ImagePixelT>::quiet_NaN() : 0,
            MaskedImageT::Mask::getPlaneBitMask("EDGE"),
            std::numeric_limits<VariancePixelT>::infinity());
    }
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
        typename InImageT::const_xy_locator inImageLocator, ///< locator for %image pixel that overlaps
            ///< pixel (0,0) of kernel (the origin of the kernel, not its center)
        lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel>::const_xy_locator kernelLocator,
                        ///< locator for (0,0) pixel of kernel (the origin of the kernel, not its center)
        int kWidth,     ///< number of columns in kernel
        int kHeight)    ///< number of rows in kernel
{
    typename OutImageT::SinglePixel outValue = 0;
    for (int y = 0; y != kHeight; ++y) {
        for (int x = 0; x != kWidth; ++x, ++inImageLocator.x(), ++kernelLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kVal = kernelLocator[0];
            if (kVal != 0) {
                outValue += *inImageLocator*kVal;
            }
        }

        inImageLocator  += lsst::afw::image::detail::difference_type(-kWidth, 1);
        kernelLocator += lsst::afw::image::detail::difference_type(-kWidth, 1);
    }

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
        typename InImageT::const_xy_locator inImageLocator,   ///< locator for %image pixel that overlaps
            ///< pixel (0,0) of kernel (the origin of the kernel, not its center)
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelXList,  ///< kernel column vector
        std::vector<lsst::afw::math::Kernel::Pixel> const &kernelYList)  ///< kernel row vector
{
    typedef typename std::vector<lsst::afw::math::Kernel::Pixel>::const_iterator k_iter;

    typedef typename OutImageT::SinglePixel OutT;
    OutT outValue = 0;
    for (k_iter kernelYIter = kernelYList.begin(), yEnd = kernelYList.end();
         kernelYIter != yEnd; ++kernelYIter) {

        OutT outValueY = 0;
        for (k_iter kernelXIter = kernelXList.begin(), xEnd = kernelXList.end();
             kernelXIter != xEnd; ++kernelXIter, ++inImageLocator.x()) {
            typename lsst::afw::math::Kernel::Pixel const kValX = *kernelXIter;
            if (kValX != 0) {
                outValueY += *inImageLocator*kValX;
            }
        }
        
        double const kValY = *kernelYIter;
        if (kValY != 0) {
            outValue += outValueY*kValY;
        }
        
        inImageLocator += lsst::afw::image::detail::difference_type(-kernelXList.size(), 1);
    }

    return outValue;
}

#endif // !defined(LSST_AFW_MATH_CONVOLVEIMAGE_H)
