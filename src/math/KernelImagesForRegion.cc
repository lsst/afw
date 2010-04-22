// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definition of KernelImagesForRegion class declared in ConvolveImage.h
 *
 * @author Russell Owen
 *
 * @ingroup afw
 */
#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "boost/assign/list_of.hpp"

#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace afwMath = lsst::afw::math;
namespace mathDetail = lsst::afw::math::detail;

using boost::assign::list_of;

/**
 * Construct a KernelImagesForRegion
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if kernelPtr is null
 */
mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr kernelPtr,     ///< kernel
        lsst::afw::geom::BoxI const &bbox,  ///< bounding box of region of an image
                                            ///< for which we want to compute kernel images
                                            ///< (inclusive and relative to parent image)
        bool doNormalize)                   ///< normalize the kernel images?
:
    lsst::daf::data::LsstBase::LsstBase(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _centerFractionalPosition(_computeCenterFractionalPosition(bbox)),
    _centerIndex(_computeCenterIndex(bbox)),
    _doNormalize(doNormalize),
    _imageMap()
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernelPtr is null");
    }
}

/**
 * Construct a KernelImagesForRegion with some or all corner images
 *
 * Null corner image pointers are ignored.
 *
 * @warning: if any images are incorrect you will get a mess.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if kernelPtr is null
 * @throw lsst::pex::exceptions::InvalidParameterException if an image has the wrong dimensions
 */
mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr const kernelPtr,     ///< kernel
        lsst::afw::geom::BoxI const &bbox,  ///< bounding box of region of an image
                                            ///< for which we want to compute kernel images
                                            ///< (inclusive and relative to parent image)
        bool doNormalize,                   ///< normalize the kernel images?
        ImageConstPtr bottomLeftImagePtr,   ///< kernel image at bottom left of region
        ImageConstPtr bottomRightImagePtr,  ///< kernel image at bottom right of region
        ImageConstPtr topLeftImagePtr,      ///< kernel image at top left of region
        ImageConstPtr topRightImagePtr)     ///< kernel image at top right of region
:
    lsst::daf::data::LsstBase::LsstBase(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _centerFractionalPosition(_computeCenterFractionalPosition(bbox)),
    _centerIndex(_computeCenterIndex(bbox)),
    _doNormalize(doNormalize),
    _imageMap()
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernelPtr is null");
    }
    _insertImage(BOTTOM_LEFT, bottomLeftImagePtr);
    _insertImage(BOTTOM_RIGHT, bottomRightImagePtr);
    _insertImage(TOP_LEFT, topLeftImagePtr);
    _insertImage(TOP_RIGHT, topRightImagePtr);
}

/**
 * Return the image at the specified location
 *
 * If the image has not yet been computed, it is computed at this time.
 */
mathDetail::KernelImagesForRegion::ImageConstPtr mathDetail::KernelImagesForRegion::getImage(
        Location location)  ///< location of image
const {
    ImageMap::const_iterator imageMapIter = _imageMap.find(location);
    if (imageMapIter != _imageMap.end()) {
        return imageMapIter->second;
    }

    afwGeom::Point2I pixelIndex = _pixelIndexFromLocation(location);
    Image::Ptr kernelImagePtr(new Image(_kernelPtr->getDimensions()));
    _kernelPtr->computeImage(
        *kernelImagePtr,
        _doNormalize,
        afwImage::indexToPosition(pixelIndex.getX()),
        afwImage::indexToPosition(pixelIndex.getY()));
    _imageMap.insert(std::make_pair(location, kernelImagePtr));
    return kernelImagePtr;
}

/**
 * Divide region into 2 by 2 sub-regions of approximately equal size.
 *
 * The subregions have exactly one row or column of overlapping pixels;
 * thus the 4 regions share 5 kernel images.
 * All corner images of all subregions are computed.
 *
 * @return a list of subregions in order: bottom left, bottom right, top left, top right
 */
std::vector<mathDetail::KernelImagesForRegion>
mathDetail::KernelImagesForRegion::getSubregions() const {
    std::vector<KernelImagesForRegion> retList;
    
    retList.push_back(KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(_bbox.getMin(), _centerIndex),
        _doNormalize,
        getImage(BOTTOM_LEFT),
        getImage(BOTTOM),
        getImage(LEFT),
        getImage(CENTER)));

    retList.push_back(KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(
            afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMinY()),
            afwGeom::Point2I::make(_bbox.getMaxX(), _centerIndex.getY())),
        _doNormalize,
        getImage(BOTTOM),
        getImage(BOTTOM_RIGHT),
        getImage(CENTER),
        getImage(RIGHT)));

    retList.push_back(KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(
            afwGeom::Point2I::make(_bbox.getMinX(), _centerIndex.getY()),
            afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMaxY())),
        _doNormalize,
        getImage(LEFT),
        getImage(CENTER),
        getImage(TOP_LEFT),
        getImage(TOP)));

    retList.push_back(KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(_centerIndex, _bbox.getMax()),
        _doNormalize,
        getImage(CENTER),
        getImage(RIGHT),
        getImage(TOP),
        getImage(TOP_RIGHT)));

    return retList;
}


/**
 * Divide region into nx by ny sub-regions of approximately equal size.
 *
 * Adjacent regions share one row or column of overlapping pixels and thus share two corner kernel images.
 *
 * All kernel images shared by the returned regions are computed.
 * In particular note that the extreme corner images may not be computed
 * (unlike the no-argument version of this function) to reduce code complexity.
 *
 * @return a list of subregions in order: bottom row left to right, next row left to right...,
 * top row left to right
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if nx >= region width or ny >= region height.
 */
std::vector<mathDetail::KernelImagesForRegion>
mathDetail::KernelImagesForRegion::getSubregions(
        int nx, ///< number of x regions
        int ny) ///< number of y regions
const {
    ImageConstPtr blImagePtr(getImage(BOTTOM_LEFT));
    ImageConstPtr brImagePtr;
    ImageConstPtr tlImagePtr;
    ImageConstPtr const trImageNullPtr;
    
    typedef std::vector<int>::const_iterator IntIter;
    std::vector<int> widthList(_computeSubregionLengths(_bbox.getWidth(), nx));
    std::vector<int> heightList(_computeSubregionLengths(_bbox.getHeight(), ny));
    std::vector<KernelImagesForRegion> retList;

    afwGeom::Point2I corner(_bbox.getMin());
    for (int yInd = 0, retInd = 0; yInd < ny; ++yInd) {
        int height = heightList[yInd];
        for (int xInd = 0; xInd < nx; ++xInd, ++retInd) {
            int width = widthList[xInd];
            if (xInd > 0) {
                // there is a region to the left; get its right-hand images
                blImagePtr = retList[retInd-1].getImage(BOTTOM_RIGHT);
                tlImagePtr = retList[retInd-1].getImage(TOP_RIGHT);
            } else {
                blImagePtr.reset();
                tlImagePtr.reset();
            }
            if (yInd > 0) {
                // there is a region below; get its top images
                if (!blImagePtr) {
                    blImagePtr = retList[retInd-nx].getImage(TOP_LEFT);
                }
                brImagePtr = retList[retInd-nx].getImage(TOP_RIGHT);
            }

            retList.push_back(KernelImagesForRegion(
                _kernelPtr,
                afwGeom::BoxI(corner, afwGeom::Extent2I::make(width, height)),
                _doNormalize,
                blImagePtr,
                brImagePtr,
                tlImagePtr,
                trImageNullPtr));
            corner += afwGeom::Extent2I::make(width, 0);
        }
        corner += afwGeom::Extent2I::make(0, height);
    }
    return retList;
}

/**
 * Will linear interpolation give a sufficiently accurate kernel image?
 *
 * The algorithm is as follows:
 * - for each location in (center, bottom, left, right, top):
 *     - error image = linearly interpolated kernel image - true kernel image (obeying doNormalize)
 *     - if the absolute value of any pixel of error image > tolerance then:
 *         - interpolation is unacceptable; stop the test
 * - interpolation is acceptable
 *
 * This is not completely foolproof, but it should do if you are careful not to test too large a region
 * relative to the wiggliness of the kernel's spatial model.
 *
 * @return true if interpolation will give sufficiently accurate results, false otherwise
 */
bool mathDetail::KernelImagesForRegion::isInterpolationOk(
        double tolerance)   ///< maximum allowed error in interpolated kernel images
const {
    typedef LocationList::const_iterator LocationIter;
    typedef Image::const_x_iterator ImageXIter;
    
    std::pair<int, int> const kernelDim = _kernelPtr->getDimensions();
    Image interpImage(kernelDim);
    for (LocationIter locIter = _TestLocationList.begin(); locIter != _TestLocationList.end(); ++locIter) {
        _interpolateImage(interpImage, *locIter);
        ImageConstPtr trueImagePtr(getImage(*locIter));
        for (int row = 0; row < kernelDim.first; ++row) {
            ImageXIter interpPtr = interpImage.row_begin(row);
            ImageXIter const interpEnd = interpImage.row_end(row);
            ImageXIter truePtr = trueImagePtr->row_begin(row);
            for ( ; interpPtr != interpEnd; ++interpPtr, ++truePtr) {
                if (std::abs(*interpPtr - *truePtr) > tolerance) {
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * Insert an image in the cache.
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if image pointer is null
 * @throw lsst::pex::exceptions::InvalidParameterException if image has the wrong dimensions
 */
void mathDetail::KernelImagesForRegion::_insertImage(
        Location location,          ///< location at which to insert image
        ImageConstPtr &imagePtr)    ///< image to insert
const {
    if (imagePtr) {
        if (_kernelPtr->getDimensions() != imagePtr->getDimensions()) {
            std::ostringstream os;
            os << "image dimensions = ( " << imagePtr->getWidth() << ", " << imagePtr->getHeight()
                << ") != (" << _kernelPtr->getWidth() << ", " << _kernelPtr->getHeight()
                << ") = kernel dimensions";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
        _imageMap.insert(std::make_pair(location, imagePtr));
    }
}

/**
 * Compute the linearly interpolated image at the specified location (not a corner).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if location is a corner,
 * i.e. is not one of: BOTTOM, TOP, LEFT, RIGHT, CENTER
 * @throw lsst::pex::exceptions::InvalidParameterException if outImage is not same dimensions as kernel.
 */
void mathDetail::KernelImagesForRegion::_interpolateImage(
        Image &outImage,  ///< output image
        Location location)      ///< location at which to compute interpolated image
const {
    double fracDist;
    switch (location) {
        case BOTTOM:
            fracDist = _centerFractionalPosition.getX();
            scaledPlus(outImage, 1.0 - fracDist, *getImage(BOTTOM_LEFT),
                                       fracDist, *getImage(BOTTOM_RIGHT));
            break;

        case TOP:
            fracDist = _centerFractionalPosition.getX();
            scaledPlus(outImage, 1.0 - fracDist, *getImage(TOP_LEFT),
                                       fracDist, *getImage(TOP_RIGHT));
            break;

        case LEFT:
            fracDist = _centerFractionalPosition.getY();
            scaledPlus(outImage, 1.0 - fracDist, *getImage(BOTTOM_LEFT),
                                       fracDist, *getImage(TOP_LEFT));
            break;

        case RIGHT:
            fracDist = _centerFractionalPosition.getY();
            scaledPlus(outImage, 1.0 - fracDist, *getImage(BOTTOM_RIGHT),
                                       fracDist, *getImage(TOP_RIGHT));
            break;

        case CENTER:
            // only perform this test for the CENTER case because the images are tested by linearInterpolate
            // for the other cases
            if (outImage.getDimensions() != _kernelPtr->getDimensions()) {
                std::ostringstream os;
                os << "image dimensions = ( " << outImage.getWidth() << ", " << outImage.getHeight()
                    << ") != (" << _kernelPtr->getWidth() << ", " << _kernelPtr->getHeight()
                    << ") = kernel dimensions";
                throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
            }
            double const xFrac = _centerFractionalPosition.getX();
            double const yFrac = _centerFractionalPosition.getY();
            for (int y = 0; y != _kernelPtr->getHeight(); ++y) {
                Image::const_x_iterator const blEnd = getImage(BOTTOM_LEFT)->row_end(y);
                Image::const_x_iterator blIter = getImage(BOTTOM_LEFT)->row_begin(y);
                Image::const_x_iterator brIter = getImage(BOTTOM_RIGHT)->row_begin(y);
                Image::const_x_iterator tlIter = getImage(TOP_LEFT)->row_begin(y);
                Image::const_x_iterator trIter = getImage(TOP_RIGHT)->row_begin(y);
                Image::x_iterator outIter = outImage.row_begin(y);
                for (; blIter != blEnd; ++blIter, ++brIter, ++tlIter, ++trIter, ++outIter) {
                    *outIter = 
                          (((*blIter * (1.0 - xFrac)) + (*brIter * xFrac)) * (1.0 - yFrac))
                        + (((*tlIter * (1.0 - xFrac)) + (*trIter * xFrac)) * yFrac);
                }
            }
            break;

        default:
            std::ostringstream os;
            os << "location = " << location << " is a corner";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}


/**
 * Compute pixel index of a given location, relative to the parent image
 * (thus offset by bottom left corner of bounding box)
 */
lsst::afw::geom::Point2I mathDetail::KernelImagesForRegion::_pixelIndexFromLocation(
        Location location)  ///< location for which to return pixel index
const {
    switch (location) {
        case BOTTOM_LEFT:
            return _bbox.getMin();
            break; // paranoia
        case BOTTOM:
            return afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMinY());
            break; // paranoia
        case BOTTOM_RIGHT:
            return afwGeom::Point2I::make(_bbox.getMaxX(), _bbox.getMinY());
            break; // paranoia
        case LEFT:
            return afwGeom::Point2I::make(_bbox.getMinX(), _centerIndex.getY());
            break; // paranoia
        case CENTER:
            return _centerIndex;
            break; // paranoia
        case RIGHT:
            return afwGeom::Point2I::make(_bbox.getMaxX(), _centerIndex.getY());
            break; // paranoia
        case TOP_LEFT:
            return afwGeom::Point2I::make(_bbox.getMinX(), _bbox.getMaxY());
            break; // paranoia
        case TOP:
            return afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMaxY());
            break; // paranoia
        case TOP_RIGHT:
            return _bbox.getMax();
            break; // paranoia
        default:
            std::ostringstream os;
            os << "Bug: unhandled location = " << location;
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
}

/**
 * Compute the fractional position of the center pixel of a bounding box.
 */
afwGeom::Point2D mathDetail::KernelImagesForRegion::_computeCenterFractionalPosition(
    afwGeom::BoxI const &bbox) ///< bounding box
{
    afwGeom::Point2I ctrInd(_computeCenterIndex(bbox));
    return afwGeom::Point2D::make(
        static_cast<double>(ctrInd.getX() - bbox.getMinX()) / static_cast<double>(bbox.getWidth()),
        static_cast<double>(ctrInd.getY() - bbox.getMinY()) / static_cast<double>(bbox.getHeight())
    );
}

/**
 * Compute center index of a bounding box
 *
 * Results will match the results from _computeSubregionLengths.
 */
afwGeom::Point2I mathDetail::KernelImagesForRegion::_computeCenterIndex(
    afwGeom::BoxI const &bbox) ///< bounding box
{
    return afwGeom::Point2I::make(
        bbox.getMinX() + _computeNextSubregionLength(bbox.getWidth(), 2) - 1,
        bbox.getMinY() + _computeNextSubregionLength(bbox.getHeight(), 2) - 1
    );
}

/**
 * Compute length of next subregion if the region is to be divided into pieces of approximately equal
 * length, each having one pixel of overlap with its neighbors.
 *
 * @return length of next subregion
 *
 * @warning: no range checking
 */
inline int mathDetail::KernelImagesForRegion::_computeNextSubregionLength(
    int length,     ///< length of region
    int nDivisions) ///< number of divisions of region
{
    return static_cast<int>(std::floor(0.5 +
        (static_cast<double>(length + nDivisions - 1) / static_cast<double>(nDivisions))));
}

/**
 * Compute length of each subregion for a region divided into nDivisions pieces of approximately equal
 * length, each having one pixel of overlap with its neighbors.
 *
 * @return a list of subspan lengths
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if nDivisions >= length
 */
std::vector<int> mathDetail::KernelImagesForRegion::_computeSubregionLengths(
    int length,     ///< length of region
    int nDivisions) ///< number of divisions of region
{
    if (nDivisions >= length) {
        std::ostringstream os;
        os << "nDivisions = " << nDivisions << " >= " << length << "length";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    std::vector<int> regionLengths;
    int remLength = length;
    for (int remNDiv = nDivisions; remNDiv > 0; --remNDiv) {
        int subLength = _computeNextSubregionLength(remLength, remNDiv);
        regionLengths.push_back(subLength);
        remLength -= (subLength - 1);
    }
    return regionLengths;
}

int const mathDetail::KernelImagesForRegion::_MinInterpSize = 5;

mathDetail::KernelImagesForRegion::LocationList const mathDetail::KernelImagesForRegion::_TestLocationList =
    boost::assign::list_of(CENTER)(BOTTOM)(LEFT)(RIGHT)(TOP);
