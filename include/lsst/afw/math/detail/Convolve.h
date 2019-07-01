// -*- LSST-C++ -*-

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

#ifndef LSST_AFW_MATH_DETAIL_CONVOLVE_H
#define LSST_AFW_MATH_DETAIL_CONVOLVE_H
/*
 * Convolution support
 */
#include <memory>
#include <sstream>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/ConvolveImage.h"

#define IS_INSTANCE(A, B) (dynamic_cast<B const*>(&(A)) != NULL)

namespace lsst {
namespace afw {
namespace math {

namespace detail {
/**
 * Low-level convolution function that does not set edge pixels.
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @param[out] convolvedImage convolved %image
 * @param[in] inImage %image to convolve
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throws lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throws lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throws std::bad_alloc when allocation of CPU memory fails
 */
template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage, lsst::afw::math::Kernel const& kernel,
                   lsst::afw::math::ConvolutionControl const& convolutionControl);

/**
 * A version of basicConvolve that should be used when convolving delta function kernels
 *
 * @param[out] convolvedImage convolved %image
 * @param[in] inImage %image to convolve
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 */
template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage,
                   lsst::afw::math::DeltaFunctionKernel const& kernel,
                   lsst::afw::math::ConvolutionControl const& convolutionControl);

/**
 * A version of basicConvolve that should be used when convolving a LinearCombinationKernel
 *
 * The Algorithm:
 * - If the kernel is spatially varying and contains only DeltaFunctionKernels
 *   then convolves the input Image by each basis kernel in turn, solves the spatial model
 *   for that component and adds in the appropriate amount of the convolved %image.
 * - In all other cases uses normal convolution
 *
 * @param[out] convolvedImage convolved %image
 * @param[in] inImage %image to convolve
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throws lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throws lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throws std::bad_alloc when allocation of CPU memory fails
 *
 * @ingroup afw
 */
template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage,
                   lsst::afw::math::LinearCombinationKernel const& kernel,
                   lsst::afw::math::ConvolutionControl const& convolutionControl);

/**
 * A version of basicConvolve that should be used when convolving separable kernels
 *
 * @param[out] convolvedImage convolved %image
 * @param[in] inImage %image to convolve
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 */
template <typename OutImageT, typename InImageT>
void basicConvolve(OutImageT& convolvedImage, InImageT const& inImage,
                   lsst::afw::math::SeparableKernel const& kernel,
                   lsst::afw::math::ConvolutionControl const& convolutionControl);

/**
 * Convolve an Image or MaskedImage with a Kernel by computing the kernel image
 * at every point. (If the kernel is not spatially varying then only compute it once).
 *
 * convolvedImage must be the same size as inImage.
 * convolvedImage has a border in which the output pixels are not set. This border has size:
 * - kernel.getCtrX() along the left edge
 * - kernel.getCtrY() along the bottom edge
 * - kernel.getWidth()  - 1 - kernel.getCtrX() along the right edge
 * - kernel.getHeight() - 1 - kernel.getCtrY() along the top edge
 *
 * @param[out] convolvedImage convolved %image
 * @param[in] inImage %image to convolve
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if convolvedImage dimensions != inImage dimensions
 * @throws lsst::pex::exceptions::InvalidParameterError if inImage smaller than kernel in width or height
 * @throws lsst::pex::exceptions::InvalidParameterError if kernel width or height < 1
 * @throws std::bad_alloc when allocation of CPU memory fails
 *
 * @warning Low-level convolution function that does not set edge pixels.
 */
template <typename OutImageT, typename InImageT>
void convolveWithBruteForce(OutImageT& convolvedImage, InImageT const& inImage,
                            lsst::afw::math::Kernel const& kernel,
                            lsst::afw::math::ConvolutionControl const& convolutionControl);

// I would prefer this to be nested in KernelImagesForRegion but SWIG doesn't support that
class RowOfKernelImagesForRegion;

/**
 * A collection of Kernel images for special locations on a rectangular region of an image
 *
 * See the Location enum for a list of those special locations.
 *
 * @warning The kernel images along the top and right edges are computed one row or column past
 * the bounding box. This allows abutting KernelImagesForRegion to share corner and edge kernel images,
 * which is useful when dividing a KernelImagesForRegion into subregions.
 *
 * @warning The bounding box for the region applies to the parent image.
 *
 * This is a low-level helper class for recursive convolving with interpolation. Many of these objects
 * may be created during a convolution, and many will share kernel images. It uses shared pointers
 * to kernels and kernel images for increased speed and decreased memory usage (at the expense of safety).
 * Note that null pointers are NOT acceptable for the constructors!
 *
 * Also note that it uses lazy evaluation: images are computed when they are wanted.
 */
class KernelImagesForRegion {
public:
    typedef std::shared_ptr<lsst::afw::math::Kernel const> KernelConstPtr;
    typedef lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel> Image;
    typedef std::shared_ptr<Image> ImagePtr;
    typedef std::shared_ptr<Image const> ImageConstPtr;

    /**
     * locations of various points in the region
     *
     * RIGHT and TOP are one column/row beyond the region's bounding box.
     * Thus adjacent regions share corner images.
     *
     * The posiitions are: BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
     *
     * These locations always refer to the center of a pixel. Thus if the region has an odd size
     * along an axis (so that the span to the top and right, which are one beyond, is even),
     * the middle pixel will be 1/2 pixel off from the true center along that axis
     * (in an unspecified direction).
     */
    enum Location { BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT };

    /**
     * Construct a KernelImagesForRegion
     *
     * @param kernelPtr kernel
     * @param bbox bounding box of region of an image for which we want to compute kernel images
     *             (inclusive and relative to parent image)
     * @param xy0 xy0 of image for which we want to compute kernel images
     * @param doNormalize normalize the kernel images?
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if kernelPtr is null
     */
    KernelImagesForRegion(KernelConstPtr kernelPtr, lsst::geom::Box2I const& bbox,
                          lsst::geom::Point2I const& xy0, bool doNormalize);
    /**
     * Construct a KernelImagesForRegion with some or all corner images
     *
     * Null corner image pointers are ignored.
     *
     * @param kernelPtr kernel
     * @param bbox bounding box of region of an image for which we want to compute kernel images
     *             (inclusive and relative to parent image)
     * @param xy0 xy0 of image
     * @param doNormalize normalize the kernel images?
     * @param bottomLeftImagePtr kernel image and sum at bottom left of region
     * @param bottomRightImagePtr kernel image and sum at bottom right of region
     * @param topLeftImagePtr kernel image and sum at top left of region
     * @param topRightImagePtr kernel image and sum at top right of region
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if kernelPtr is null
     * @throws lsst::pex::exceptions::InvalidParameterError if an image has the wrong dimensions
     *
     * @warning: if any images are incorrect you will get a mess.
     */
    KernelImagesForRegion(KernelConstPtr kernelPtr, lsst::geom::Box2I const& bbox,
                          lsst::geom::Point2I const& xy0, bool doNormalize, ImagePtr bottomLeftImagePtr,
                          ImagePtr bottomRightImagePtr, ImagePtr topLeftImagePtr, ImagePtr topRightImagePtr);

    /**
     * Get the bounding box for the region
     */
    lsst::geom::Box2I getBBox() const { return _bbox; };
    /**
     * Get xy0 of the image
     */
    lsst::geom::Point2I getXY0() const { return _xy0; };
    /**
     * Get the doNormalize parameter
     */
    bool getDoNormalize() const { return _doNormalize; };
    /**
     * Return the image and sum at the specified location
     *
     * If the image has not yet been computed, it is computed at this time.
     *
     * @param location location of image
     */
    ImagePtr getImage(Location location) const;
    /**
     * Get the kernel (as a shared pointer to const)
     */
    KernelConstPtr getKernel() const { return _kernelPtr; };
    /**
     * Compute pixel index of a given location, relative to the parent image
     * (thus offset by bottom left corner of bounding box)
     *
     * @param location location for which to return pixel index
     */
    lsst::geom::Point2I getPixelIndex(Location location) const;
    /**
     * Compute next row of subregions
     *
     * For the first row call with a new RowOfKernelImagesForRegion (with the desired number of columns and
     * rows).
     * Every subequent call updates the data in the RowOfKernelImagesForRegion.
     *
     * @param[in, out] regionRow RowOfKernelImagesForRegion object
     * @returns true if a new row was computed, false if supplied RowOfKernelImagesForRegion is for the last
     * row.
     */
    bool computeNextRow(RowOfKernelImagesForRegion& regionRow) const;

    /**
     * Get the minInterpolationSize class constant
     */
    static int getMinInterpolationSize() { return _MinInterpolationSize; };

private:
    typedef std::vector<Location> LocationList;

    /**
     * Compute image at a particular location
     *
     * @throws lsst::pex::exceptions::NotFoundError if there is no pointer at that location
     */
    void _computeImage(Location location) const;
    inline void _insertImage(Location location, ImagePtr imagePtr) const;
    /**
     * Move the region up one segment
     *
     * To avoid reallocating memory for kernel images, swap the top and bottom kernel image pointers
     * and recompute the top images. Actually, only does this to the right-hande images if isFirst is false
     * since it assumes the left images were already handled.
     *
     * Intended to support computeNextRow; as such assumes that a list of adjacent regions will be moved,
     * left to right.
     *
     * @param isFirst true if the first region in a row (or the only region you are moving)
     * @param newHeight the height of the region after moving it
     */
    void _moveUp(bool isFirst, int newHeight);

    // static helper functions
    static inline int _computeNextSubregionLength(int length, int nDivisions);
    /**
     * Compute length of each subregion for a region divided into nDivisions pieces of approximately equal
     * length.
     *
     * @param length length of region
     * @param nDivisions number of divisions of region
     * @returns a list of subspan lengths
     *
     * @throws lsst::pex::exceptions::InvalidParameterError if nDivisions >= length
     */
    static std::vector<int> _computeSubregionLengths(int length, int nDivisions);

    // member variables
    KernelConstPtr _kernelPtr;
    lsst::geom::Box2I _bbox;
    lsst::geom::Point2I _xy0;
    bool _doNormalize;
    mutable std::vector<ImagePtr> _imagePtrList;

    static int const _MinInterpolationSize;
};

/**
 * A row of KernelImagesForRegion
 *
 * Intended for iterating over subregions of a KernelImagesForRegion using computeNextRow.
 */
class RowOfKernelImagesForRegion final {
public:
    typedef std::vector<std::shared_ptr<KernelImagesForRegion>> RegionList;
    typedef RegionList::iterator Iterator;
    typedef RegionList::const_iterator ConstIterator;

    /**
     * Construct a RowOfKernelImagesForRegion
     *
     * @param nx number of columns
     * @param ny number of rows
     */
    RowOfKernelImagesForRegion(int nx, int ny);
    /**
     * Return the begin iterator for the list
     */
    RegionList::const_iterator begin() const { return _regionList.begin(); };
    /**
     * Return the end iterator for the list
     */
    RegionList::const_iterator end() const { return _regionList.end(); };
    /**
     * Return the begin iterator for the list
     */
    RegionList::iterator begin() { return _regionList.begin(); };
    /**
     * Return the end iterator for the list
     */
    RegionList::iterator end() { return _regionList.end(); };
    /**
     * Return the first region in the list
     */
    std::shared_ptr<KernelImagesForRegion> front() { return _regionList.front(); };
    /**
     * Return the last region in the list
     */
    std::shared_ptr<KernelImagesForRegion> back() { return _regionList.back(); };
    int getNX() const { return _nx; };
    int getNY() const { return _ny; };
    int getYInd() const { return _yInd; };
    /**
     * get the specified region (range-checked)
     *
     * @throws std::range_error if ind out of range
     */
    std::shared_ptr<KernelImagesForRegion const> getRegion(int ind) const { return _regionList.at(ind); };
    bool hasData() const { return static_cast<bool>(_regionList[0]); };
    bool isLastRow() const { return _yInd + 1 >= _ny; };
    int incrYInd() { return ++_yInd; };

private:
    int _nx;
    int _ny;
    int _yInd;
    RegionList _regionList;
};

/**
 * Convolve an Image or MaskedImage with a spatially varying Kernel using linear interpolation.
 *
 * This is a low-level convolution function that does not set edge pixels.
 *
 * The algorithm is as follows:
 * - divide the image into regions whose size is no larger than maxInterpolationDistance
 * - for each region:
 *   - convolve it using convolveRegionWithInterpolation (which see)
 *
 * Note that this routine will also work with spatially invariant kernels, but not efficiently.
 *
 * @param[out] outImage convolved image = inImage convolved with kernel
 * @param[in] inImage input image
 * @param[in] kernel convolution kernel
 * @param[in] convolutionControl convolution control parameters
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if outImage is not the same size as inImage
 */
template <typename OutImageT, typename InImageT>
void convolveWithInterpolation(OutImageT& outImage, InImageT const& inImage,
                               lsst::afw::math::Kernel const& kernel,
                               ConvolutionControl const& convolutionControl);

/**
 * kernel images used by convolveRegionWithInterpolation
 */
struct ConvolveWithInterpolationWorkingImages final {
public:
    typedef lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel> Image;
    ConvolveWithInterpolationWorkingImages(lsst::geom::Extent2I const& dimensions)
            : leftImage(dimensions),
              rightImage(dimensions),
              leftDeltaImage(dimensions),
              rightDeltaImage(dimensions),
              deltaImage(dimensions),
              kernelImage(dimensions) {}
    Image leftImage;
    Image rightImage;
    Image leftDeltaImage;
    Image rightDeltaImage;
    Image deltaImage;
    Image kernelImage;
};

/**
 * Convolve a region of an Image or MaskedImage with a spatially varying Kernel using interpolation.
 *
 * This is a low-level convolution function that does not set edge pixels.
 *
 * @param[out] outImage convolved image = inImage convolved with kernel
 * @param[in] inImage input image
 * @param[in] region kernel image region over which to convolve
 * @param[in] workingImages working kernel images
 *
 * @warning: this is a low-level routine that performs no bounds checking.
 */
template <typename OutImageT, typename InImageT>
void convolveRegionWithInterpolation(OutImageT& outImage, InImageT const& inImage,
                                     KernelImagesForRegion const& region,
                                     ConvolveWithInterpolationWorkingImages& workingImages);

/*
 * Define inline functions
 */

/**
 * Compute length of next subregion if the region is to be divided into pieces of approximately equal length.
 *
 * @returns length of next subregion
 *
 * @warning: no range checking
 */
inline int KernelImagesForRegion::_computeNextSubregionLength(
        int length,      ///< length of region
        int nDivisions)  ///< number of divisions of region
{
    return static_cast<int>(
            std::floor(0.5 + (static_cast<double>(length) / static_cast<double>(nDivisions))));
}

/**
 * Insert an image in the cache.
 *
 * @throws lsst::pex::exceptions::InvalidParameterError if image pointer is null
 * @throws lsst::pex::exceptions::InvalidParameterError if image has the wrong dimensions
 */
inline void KernelImagesForRegion::_insertImage(Location location,  ///< location at which to insert image
                                                ImagePtr imagePtr)  ///< image to insert
        const {
    if (imagePtr) {
        if (_kernelPtr->getDimensions() != imagePtr->getDimensions()) {
            std::ostringstream os;
            os << "image dimensions = ( " << imagePtr->getWidth() << ", " << imagePtr->getHeight() << ") != ("
               << _kernelPtr->getWidth() << ", " << _kernelPtr->getHeight() << ") = kernel dimensions";
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
        }
        _imagePtrList[location] = imagePtr;
    }
}
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif  // !defined(LSST_AFW_MATH_DETAIL_CONVOLVE_H)
