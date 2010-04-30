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
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"

namespace lsst {
namespace afw {
namespace math {

    /**
     * @brief Parameters to control convolution
     *
     * @ingroup afw
     */
    class ConvolutionControl {
    public:
        ConvolutionControl(
                bool doNormalize = true,    ///< normalize the kernel to sum=1?
                bool doCopyEdge = false,    ///< copy edge pixels from source image
                    ///< instead of setting them to the standard edge pixel?
                double maxInterpolationError = 1.0e-5,  ///< maximum allowed error
                    ///< in computing the value of the kernel at any pixel by linear interpolation
                int maxInterpolationDistance = 50)  ///< maximum width or height of a region
                    ///< over which to test if interpolation works
        :
            _doNormalize(doNormalize),
            _doCopyEdge(doCopyEdge),
            _maxInterpolationError(maxInterpolationError),
            _maxInterpolationDistance(maxInterpolationDistance)
        { }
    
        bool getDoNormalize() const { return _doNormalize; }
        bool getDoCopyEdge() const { return _doCopyEdge; }
        double getMaxInterpolationError() const { return _maxInterpolationError; }
        int getMaxInterpolationDistance() const { return _maxInterpolationDistance; };
        
        void setDoNormalize(bool doNormalize) {_doNormalize = doNormalize; }
        void setDoCopyEdge(bool doCopyEdge) { _doCopyEdge = doCopyEdge; }
        void setMaxInterpolationError(double maxInterpolationError) {
            _maxInterpolationError = maxInterpolationError; }
        void setMaxInterpolationDistance(int maxInterpolationDistance) {
            _maxInterpolationDistance = maxInterpolationDistance; }
    
    private:
        bool _doNormalize;  ///< normalize the kernel to sum=1?
        bool _doCopyEdge;   ///< copy edge pixels from source image
                    ///< instead of setting them to the standard edge pixel?
        double _maxInterpolationError;  ///< maximum allowed error in computing the kernel image;
                    ///< applies to linear interpolation and perhaps other approximate methods in the future
        int _maxInterpolationDistance;  ///< maximum width or height of a region
                    ///< over which to attempt interpolation
    };

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
             * These locations always refer to the center of a pixel. Thus if the region has an even size
             * along an axis, then the middle pixel will be 1/2 pixel off from the true center along that axis
             * (in an unspecified direction).
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
            bool isInterpolationOk(double maxInterpolationError) const;
            void interpolateImage(Image &outImage, Location location) const;
            static int getMinInterpolationSize() { return _MinInterpolationSize; };
    
//        private:
            typedef std::map<Location, ImageConstPtr> ImageMap;
            typedef std::vector<Location> LocationList;
    
            inline void _insertImage(Location location, ImageConstPtr &imagePtr) const;
            
            // static helper functions
            static lsst::afw::geom::Point2D _computeCenterFractionalPosition(lsst::afw::geom::BoxI const &bbox);
            static lsst::afw::geom::Point2I _computeCenterIndex(lsst::afw::geom::BoxI const &bbox);
            static inline int _computeNextSubregionLength(int length, int nDivisions);
            static std::vector<int> _computeSubregionLengths(int length, int nDivisions);
            lsst::afw::geom::Point2I _getPixelIndex(Location location) const;
            
            // member variables
            KernelConstPtr _kernelPtr;
            lsst::afw::geom::BoxI _bbox;
            lsst::afw::geom::Point2D _centerFractionalPosition;  ///< fractional position of center pixel
                ///< from bottom left to top right; 0.5 if length of axis is odd, somewhat less if even
            lsst::afw::geom::Point2I _centerIndex;  ///< index of center pixel
            bool _doNormalize;
            mutable ImageMap _imageMap; ///< cache of location:kernel image;
                ///< mutable to support lazy evaluation: const methods may add entries to the cache
    
            static int const _MinInterpolationSize;
            static LocationList const _TestLocationList;   ///< locations at which to test
                ///< linear interpolation to see if it is accurate enough
        };
        
        template <typename OutImageT, typename InImageT>
        void convolveWithInterpolation(
                OutImageT &outImage,
                InImageT const &inImage,
                lsst::afw::math::Kernel const &kernel,
                ConvolutionControl const &convolutionControl);
    
        template <typename OutImageT, typename InImageT>
        void convolveRegionWithRecursiveInterpolation(
                OutImageT &outImage,
                InImageT const &inImage,
                KernelImagesForRegion const &region,
                double maxInterpolationError = 1.0e-5);
        
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

        template <typename OutImageT, typename InImageT>
        void basicConvolve(
                OutImageT& convolvedImage,
                InImageT const& inImage,
                lsst::afw::math::Kernel const& kernel,
                lsst::afw::math::ConvolutionControl const& convolutionControl);
        
        template <typename OutImageT, typename InImageT>
        void basicConvolve(
                OutImageT& convolvedImage,
                InImageT const& inImage,
                lsst::afw::math::DeltaFunctionKernel const& kernel,
                lsst::afw::math::ConvolutionControl const&);
        
        template <typename OutImageT, typename InImageT>
        void basicConvolve(
                OutImageT& convolvedImage,
                InImageT const& inImage,
                lsst::afw::math::LinearCombinationKernel const& kernel,
                lsst::afw::math::ConvolutionControl const& convolutionControl);
        
        template <typename OutImageT, typename InImageT>
        void basicConvolve(
                OutImageT& convolvedImage,
                InImageT const& inImage,
                lsst::afw::math::SeparableKernel const& kernel,
                lsst::afw::math::ConvolutionControl const& convolutionControl);
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
    
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            ConvolutionControl const& convolutionControl);
    
    template <typename OutImageT, typename InImageT, typename KernelT>
    void convolve(
            OutImageT& convolvedImage,
            InImageT const& inImage,
            KernelT const& kernel,
            bool doNormalize,
            bool doCopyEdge = false);
    
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

/*
 * Define inline functions
 */

/**
 * Compute length of next subregion if the region is to be divided into pieces of approximately equal
 * length, each having one pixel of overlap with its neighbors.
 *
 * @return length of next subregion
 *
 * @warning: no range checking
 */
inline int lsst::afw::math::detail::KernelImagesForRegion::_computeNextSubregionLength(
    int length,     ///< length of region
    int nDivisions) ///< number of divisions of region
{
    return static_cast<int>(std::floor(0.5 +
        (static_cast<double>(length + nDivisions - 1) / static_cast<double>(nDivisions))));
}

/**
 * Insert an image in the cache.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if image pointer is null
 * @throw lsst::pex::exceptions::InvalidParameterException if image has the wrong dimensions
 */
inline void lsst::afw::math::detail::KernelImagesForRegion::_insertImage(
        Location location,          ///< location at which to insert image
        ImageConstPtr &imagePtr)    ///< image to insert
const {
    if (imagePtr) {
        if (_kernelPtr->getDimensions() != imagePtr->getDimensions()) {
            std::ostringstream os;
            os << "image dimensions = ( " << imagePtr->getWidth() << ", " << imagePtr->getHeight()
                << ") != (" << _kernelPtr->getWidth() << ", " << _kernelPtr->getHeight()
                << ") = kernel dimensions";
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException, os.str());
        }
        _imageMap.insert(std::make_pair(location, imagePtr));
    }
}

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
