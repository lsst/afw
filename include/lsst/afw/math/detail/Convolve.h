// -*- LSST-C++ -*-
#ifndef LSST_AFW_MATH_DETAIL_CONVOLVE_H
#define LSST_AFW_MATH_DETAIL_CONVOLVE_H
/**
 * @file
 *
 * @brief Convolution support
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/ConvolveImage.h"

# define ISINSTANCE(A, B) (dynamic_cast<B const*>(&(A)) != NULL)

namespace lsst {
namespace afw {
namespace math {
namespace detail {
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

    template <typename OutImageT, typename InImageT>
    void convolveWithBruteForce(
            OutImageT &convolvedImage,
            InImageT const& inImage,
            lsst::afw::math::Kernel const& kernel,
            bool doNormalize);

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

    private:
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
}}}}   // lsst::afw::math::detail

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

#endif // !defined(LSST_AFW_MATH_DETAIL_CONVOLVE_H)
