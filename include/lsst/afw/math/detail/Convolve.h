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
/**
 * @file
 *
 * @brief Convolution support
 *
 * @author Russell Owen
 *
 * @ingroup afw
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
    class KernelImagesForRegion :
        public lsst::daf::base::Citizen,
        public lsst::daf::base::Persistable
    {
    public:
        typedef CONST_PTR(lsst::afw::math::Kernel) KernelConstPtr;
        typedef lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel> Image;
        typedef PTR(Image) ImagePtr;
        typedef CONST_PTR(Image) ImageConstPtr;
        typedef CONST_PTR(KernelImagesForRegion) ConstPtr;
        typedef PTR(KernelImagesForRegion) Ptr;

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
        enum Location {
            BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
        };

        KernelImagesForRegion(
                KernelConstPtr kernelPtr,
                lsst::afw::geom::Box2I const &bbox,
                lsst::afw::geom::Point2I const &xy0,
                bool doNormalize);
        KernelImagesForRegion(
                KernelConstPtr kernelPtr,
                lsst::afw::geom::Box2I const &bbox,
                lsst::afw::geom::Point2I const &xy0,
                bool doNormalize,
                ImagePtr bottomLeftImagePtr,
                ImagePtr bottomRightImagePtr,
                ImagePtr topLeftImagePtr,
                ImagePtr topRightImagePtr);

        /**
         * Get the bounding box for the region
         */
        lsst::afw::geom::Box2I getBBox() const { return _bbox; };
        /**
         * Get xy0 of the image
         */
        lsst::afw::geom::Point2I getXY0() const { return _xy0; };
        /**
         * Get the doNormalize parameter
         */
        bool getDoNormalize() const { return _doNormalize; };
        ImagePtr getImage(Location location) const;
        /**
         * Get the kernel (as a shared pointer to const)
         */
        KernelConstPtr getKernel() const { return _kernelPtr; };
        lsst::afw::geom::Point2I getPixelIndex(Location location) const;
        bool computeNextRow(RowOfKernelImagesForRegion &regionRow) const;

        /**
         * Get the minInterpolationSize class constant
         */
        static int getMinInterpolationSize() { return _MinInterpolationSize; };
    private:
        typedef std::vector<Location> LocationList;

        void _computeImage(Location location) const;
        inline void _insertImage(Location location, ImagePtr imagePtr) const;
        void _moveUp(bool isFirst, int newHeight);

        // static helper functions
        static inline int _computeNextSubregionLength(int length, int nDivisions);
        static std::vector<int> _computeSubregionLengths(int length, int nDivisions);

        // member variables
        KernelConstPtr _kernelPtr;
        lsst::afw::geom::Box2I _bbox;
        lsst::afw::geom::Point2I _xy0;
        bool _doNormalize;
        mutable std::vector<ImagePtr> _imagePtrList;

        static int const _MinInterpolationSize;
    };

    /**
     * @brief A row of KernelImagesForRegion
     *
     * Intended for iterating over subregions of a KernelImagesForRegion using computeNextRow.
     */
    class RowOfKernelImagesForRegion {
    public:
        typedef std::vector<PTR(KernelImagesForRegion)> RegionList;
        typedef RegionList::iterator Iterator;
        typedef RegionList::const_iterator ConstIterator;

        RowOfKernelImagesForRegion(int nx, int ny);
        /**
         * @brief Return the begin iterator for the list
         */
        RegionList::const_iterator begin() const { return _regionList.begin(); };
        /**
         * @brief Return the end iterator for the list
         */
        RegionList::const_iterator end() const { return _regionList.end(); };
        /**
         * @brief Return the begin iterator for the list
         */
        RegionList::iterator begin() { return _regionList.begin(); };
        /**
         * @brief Return the end iterator for the list
         */
        RegionList::iterator end() { return _regionList.end(); };
        /**
         * @brief Return the first region in the list
         */
        PTR(KernelImagesForRegion) front() { return _regionList.front(); };
        /**
         * @brief Return the last region in the list
         */
        PTR(KernelImagesForRegion) back() { return _regionList.back(); };
        int getNX() const { return _nx; };
        int getNY() const { return _ny; };
        int getYInd() const { return _yInd; };
        /**
         * @brief get the specified region (range-checked)
         *
         * @throw std::range_error if ind out of range
         */
        CONST_PTR(KernelImagesForRegion) getRegion(int ind) const { return _regionList.at(ind); };
        bool hasData() const { return static_cast<bool>(_regionList[0]); };
        bool isLastRow() const { return _yInd + 1 >= _ny; };
        int incrYInd() { return ++_yInd; };

    private:
        int _nx;
        int _ny;
        int _yInd;
        RegionList _regionList;
    };

    template <typename OutImageT, typename InImageT>
    void convolveWithInterpolation(
            OutImageT &outImage,
            InImageT const &inImage,
            lsst::afw::math::Kernel const &kernel,
            ConvolutionControl const &convolutionControl);

    /**
     * @brief kernel images used by convolveRegionWithInterpolation
     */
    struct ConvolveWithInterpolationWorkingImages {
    public:
        typedef lsst::afw::image::Image<lsst::afw::math::Kernel::Pixel> Image;
        ConvolveWithInterpolationWorkingImages(geom::Extent2I const & dimensions) :
            leftImage(dimensions),
            rightImage(dimensions),
            leftDeltaImage(dimensions),
            rightDeltaImage(dimensions),
            deltaImage(dimensions),
            kernelImage(dimensions)
        { }
        Image leftImage;
        Image rightImage;
        Image leftDeltaImage;
        Image rightDeltaImage;
        Image deltaImage;
        Image kernelImage;
    };

    template <typename OutImageT, typename InImageT>
    void convolveRegionWithInterpolation(
            OutImageT &outImage,
            InImageT const &inImage,
            KernelImagesForRegion const &region,
            ConvolveWithInterpolationWorkingImages &workingImages);
}}}}   // lsst::afw::math::detail

/*
 * Define inline functions
 */

/**
 * Compute length of next subregion if the region is to be divided into pieces of approximately equal length.
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
        (static_cast<double>(length) / static_cast<double>(nDivisions))));
}

/**
 * Insert an image in the cache.
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if image pointer is null
 * @throw lsst::pex::exceptions::InvalidParameterError if image has the wrong dimensions
 */
inline void lsst::afw::math::detail::KernelImagesForRegion::_insertImage(
        Location location,      ///< location at which to insert image
        ImagePtr imagePtr)      ///< image to insert
const {
    if (imagePtr) {
        if (_kernelPtr->getDimensions() != imagePtr->getDimensions()) {
            std::ostringstream os;
            os << "image dimensions = ( "
                << imagePtr->getWidth() << ", " << imagePtr->getHeight()
                << ") != (" << _kernelPtr->getWidth() << ", " << _kernelPtr->getHeight()
                << ") = kernel dimensions";
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, os.str());
        }
        _imagePtrList[location] = imagePtr;
    }
}

#endif // !defined(LSST_AFW_MATH_DETAIL_CONVOLVE_H)
