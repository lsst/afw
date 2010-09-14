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
 
/**
 * @file
 *
 * @brief Definition of KernelImagesForRegion class declared in detail/ConvolveImage.h
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

#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;
namespace pexLog = lsst::pex::logging;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace mathDetail = lsst::afw::math::detail;

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
        lsst::afw::geom::Point2I const &xy0,    ///< xy0 of image for which we want to compute kernel images
        bool doNormalize)                   ///< normalize the kernel images?
:
    lsst::daf::data::LsstBase::LsstBase(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _xy0(xy0),
    _centerFractionalPosition(_computeCenterFractionalPosition(bbox)),
    _centerIndex(_computeCenterIndex(bbox)),
    _doNormalize(doNormalize),
    _imageMap()
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernelPtr is null");
    }
    pexLog::TTrace<6>("lsst.afw.math.convolve",
    "KernelImagesForRegion(bbox(minimum=(%d, %d), extent=(%d, %d)), xy0=(%d, %d), doNormalize=%d, images...)",
       _bbox.getMinX(), _bbox.getMinY(), _bbox.getWidth(), _bbox.getHeight(), _xy0[0], _xy0[1], _doNormalize);
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
        KernelConstPtr const kernelPtr,         ///< kernel
        lsst::afw::geom::BoxI const &bbox,      ///< bounding box of region of an image
                                                ///< for which we want to compute kernel images
                                                ///< (inclusive and relative to parent image)
        lsst::afw::geom::Point2I const &xy0,    ///< xy0 of image
        bool doNormalize,                       ///< normalize the kernel images?
        ImageSumPair bottomLeftImageSumPair,    ///< kernel image and sum at bottom left of region
        ImageSumPair bottomRightImageSumPair,   ///< kernel image and sum at bottom right of region
        ImageSumPair topLeftImageSumPair,       ///< kernel image and sum at top left of region
        ImageSumPair topRightImageSumPair)      ///< kernel image and sum at top right of region
:
    lsst::daf::data::LsstBase::LsstBase(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _xy0(xy0),
    _centerFractionalPosition(_computeCenterFractionalPosition(bbox)),
    _centerIndex(_computeCenterIndex(bbox)),
    _doNormalize(doNormalize),
    _imageMap()
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, "kernelPtr is null");
    }
    _insertImage(BOTTOM_LEFT, bottomLeftImageSumPair);
    _insertImage(BOTTOM_RIGHT, bottomRightImageSumPair);
    _insertImage(TOP_LEFT, topLeftImageSumPair);
    _insertImage(TOP_RIGHT, topRightImageSumPair);
    pexLog::TTrace<6>("lsst.afw.math.convolve",
    "KernelImagesForRegion(bbox(minimum=(%d, %d), extent=(%d, %d)), xy0=(%d, %d), doNormalize=%d, images...)",
       _bbox.getMinX(), _bbox.getMinY(), _bbox.getWidth(), _bbox.getHeight(), _xy0[0], _xy0[1], _doNormalize);
}

/**
 * Return the image and sum at the specified location
 *
 * If the image has not yet been computed, it is computed at this time.
 */
mathDetail::KernelImagesForRegion::ImageSumPair mathDetail::KernelImagesForRegion::getImageSumPair(
        Location location)  ///< location of image
const {
    ImageMap::const_iterator imageMapIter = _imageMap.find(location);
    if (imageMapIter != _imageMap.end()) {
        return imageMapIter->second;
    }

    afwGeom::Point2I pixelIndex = getPixelIndex(location);
    Image::Ptr kernelImagePtr(new Image(_kernelPtr->getDimensions()));
    double kernelSum = _kernelPtr->computeImage(
        *kernelImagePtr,
        _doNormalize,
        afwImage::indexToPosition(pixelIndex.getX() + _xy0[0]),
        afwImage::indexToPosition(pixelIndex.getY() + _xy0[1]));
    ImageSumPair imageSumPair(kernelImagePtr, kernelSum);
    _imageMap.insert(std::make_pair(location, imageSumPair));
    return imageSumPair;
}

/**
 * Compute pixel index of a given location, relative to the parent image
 * (thus offset by bottom left corner of bounding box)
 */
lsst::afw::geom::Point2I mathDetail::KernelImagesForRegion::getPixelIndex(
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
            return afwGeom::Point2I::make(_bbox.getMaxX() + 1, _bbox.getMinY());
            break; // paranoia
        case LEFT:
            return afwGeom::Point2I::make(_bbox.getMinX(), _centerIndex.getY());
            break; // paranoia
        case CENTER:
            return _centerIndex;
            break; // paranoia
        case RIGHT:
            return afwGeom::Point2I::make(_bbox.getMaxX() + 1, _centerIndex.getY());
            break; // paranoia
        case TOP_LEFT:
            return afwGeom::Point2I::make(_bbox.getMinX(), _bbox.getMaxY() + 1);
            break; // paranoia
        case TOP:
            return afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMaxY() + 1);
            break; // paranoia
        case TOP_RIGHT:
            return afwGeom::Point2I::make(_bbox.getMaxX() + 1, _bbox.getMaxY() + 1);
            break; // paranoia
        default: {
            std::ostringstream os;
            os << "Bug: unhandled location = " << location;
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
    }
}

/**
 * Divide region into 2 by 2 sub-regions of approximately equal size.
 *
 * The four subregions share 5 kernel images (bottom, left, center, right, top) from this region.
 * All corner images of all four subregions are computed.
 *
 * @return a list of subregions in order: bottom left, bottom right, top left, top right
 */
mathDetail::KernelImagesForRegion::List
mathDetail::KernelImagesForRegion::getSubregions() const {
    List retList;
    
    retList.push_back(KernelImagesForRegion::ConstPtr(new KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(_bbox.getMin(), _centerIndex - afwGeom::Extent2I(1)),
        _xy0,
        _doNormalize,
        getImageSumPair(BOTTOM_LEFT),
        getImageSumPair(BOTTOM),
        getImageSumPair(LEFT),
        getImageSumPair(CENTER))));

    retList.push_back(KernelImagesForRegion::ConstPtr(new KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(
            afwGeom::Point2I::make(_centerIndex.getX(), _bbox.getMinY()),
            afwGeom::Point2I::make(_bbox.getMaxX(), _centerIndex.getY() - 1)),
        _xy0,
        _doNormalize,
        getImageSumPair(BOTTOM),
        getImageSumPair(BOTTOM_RIGHT),
        getImageSumPair(CENTER),
        getImageSumPair(RIGHT))));

    retList.push_back(KernelImagesForRegion::ConstPtr(new KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(
            afwGeom::Point2I::make(_bbox.getMinX(), _centerIndex.getY()),
            afwGeom::Point2I::make(_centerIndex.getX() - 1, _bbox.getMaxY())),
        _xy0,
        _doNormalize,
        getImageSumPair(LEFT),
        getImageSumPair(CENTER),
        getImageSumPair(TOP_LEFT),
        getImageSumPair(TOP))));

    retList.push_back(KernelImagesForRegion::ConstPtr(new KernelImagesForRegion(
        _kernelPtr,
        afwGeom::BoxI(_centerIndex, _bbox.getMax()),
        _xy0,
        _doNormalize,
        getImageSumPair(CENTER),
        getImageSumPair(RIGHT),
        getImageSumPair(TOP),
        getImageSumPair(TOP_RIGHT))));

    return retList;
}

/**
 * @brief Compute next row of subregions
 *
 * For the first row call with a new RowOfKernelImagesForRegion (with the desired number of columns and rows).
 * Every subequent call updates the data in the RowOfKernelImagesForRegion.
 *
 * @return true if a new row was computed, false if the supplied RowOfKernelImagesForRegion is for the last row.
 */
bool mathDetail::KernelImagesForRegion::computeNextRow(
        RowOfKernelImagesForRegion &regionRow) ///< RowOfKernelImagesForRegion object
const {
    if (regionRow.isLastRow()) {
        return false;
    }

    ImageSumPair blImageSumPair;
    ImageSumPair brImageSumPair;
    ImageSumPair tlImageSumPair;
    ImageSumPair const trImageNullPtr;
    
    int yInd = regionRow.incrYInd();
    bool isFirstRow = (yInd == 0);
    
    int startY;
    if (isFirstRow) {
        startY = this->_bbox.getMinY();
    } else {
        startY = (*regionRow.begin())->getBBox().getMaxY() + 1;
    }
    afwGeom::Point2I blCorner = afwGeom::makePointI(this->_bbox.getMinX(), startY);

    int remHeight = 1 + this->_bbox.getMaxY() - startY;
    int remYDiv = regionRow.getNY() - yInd;
    int height = _computeNextSubregionLength(remHeight, remYDiv);
    
    if (isFirstRow) {
        blImageSumPair = getImageSumPair(BOTTOM_LEFT);

        int remWidth = this->_bbox.getWidth();
        int remXDiv = regionRow.getNX();
        for (RowOfKernelImagesForRegion::Iterator rgnIter = regionRow.begin(), rgnEnd = regionRow.end(); rgnIter != rgnEnd;
            ++rgnIter) {
            int width = _computeNextSubregionLength(remWidth, remXDiv);
            --remXDiv;
            remWidth -= width;
            
            KernelImagesForRegion::Ptr regionPtr(new KernelImagesForRegion(
                _kernelPtr,
                afwGeom::BoxI(blCorner, afwGeom::Extent2I::make(width, height)),
                _xy0,
                _doNormalize,
                blImageSumPair,
                brImageSumPair,
                tlImageSumPair,
                trImageNullPtr));
            *rgnIter = regionPtr;
            
            blCorner += afwGeom::Extent2I::make(width, 0);
            blImageSumPair = regionPtr->getImageSumPair(BOTTOM_RIGHT);
            tlImageSumPair = regionPtr->getImageSumPair(TOP_RIGHT);
        }
    } else {
        for (RowOfKernelImagesForRegion::Iterator rgnIter = regionRow.begin(), rgnEnd = regionRow.end(); rgnIter != rgnEnd;
            ++rgnIter) {
            
            // rgnIter points to the region just below the one we are computing;
            // it has the same width and shares two images
            int width = (**rgnIter).getBBox().getWidth();
            blImageSumPair = (**rgnIter).getImageSumPair(TOP_LEFT);
            brImageSumPair = (**rgnIter).getImageSumPair(TOP_RIGHT);
            
            KernelImagesForRegion::Ptr regionPtr(new KernelImagesForRegion(
                _kernelPtr,
                afwGeom::BoxI(blCorner, afwGeom::Extent2I::make(width, height)),
                _xy0,
                _doNormalize,
                blImageSumPair,
                brImageSumPair,
                tlImageSumPair,
                trImageNullPtr));
            *rgnIter = regionPtr;
            
            blCorner += afwGeom::Extent2I::make(width, 0);
            tlImageSumPair = regionPtr->getImageSumPair(TOP_RIGHT);
        }
    }
    return true;
}

/**
 * Divide region into nx by ny sub-regions of approximately equal size.
 *
 * Adjacent regions share two corner kernel images.
 *
 * All kernel images shared by the returned regions are computed, but others may not be.
 * In particular note that the extreme corner images may not be computed
 * (unlike the no-argument version of this function) to reduce code complexity.
 *
 * @return a list of subregions in order: bottom row left to right, next row left to right...,
 * top row left to right
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if nx >= region width or ny >= region height.
 * @throw lsst::pex::exceptions::InvalidParameterException if nx < 1 or ny < 1.
 */
mathDetail::KernelImagesForRegion::List mathDetail::KernelImagesForRegion::getSubregions(
        int nx, ///< number of x regions
        int ny) ///< number of y regions
const {
    ImageSumPair blImageSumPair(getImageSumPair(BOTTOM_LEFT));
    ImageSumPair brImageSumPair;
    ImageSumPair tlImageSumPair;
    ImageSumPair const trImageNullPtr;
    
    typedef std::vector<int>::const_iterator IntIter;
    std::vector<int> widthList(_computeSubregionLengths(_bbox.getWidth(), nx));
    std::vector<int> heightList(_computeSubregionLengths(_bbox.getHeight(), ny));
    List retList;

    afwGeom::Point2I leftCorner(_bbox.getMin());
    for (int yInd = 0, retInd = 0; yInd < ny; ++yInd) {
        int height = heightList[yInd];
        afwGeom::Point2I corner = leftCorner;
        for (int xInd = 0; xInd < nx; ++xInd, ++retInd) {
            int width = widthList[xInd];
            if (xInd > 0) {
                // there is a region to the left; get its right-hand images
                blImageSumPair = retList[retInd-1]->getImageSumPair(BOTTOM_RIGHT);
                tlImageSumPair = retList[retInd-1]->getImageSumPair(TOP_RIGHT);
            } else {
                blImageSumPair.first.reset();
                tlImageSumPair.first.reset();
            }
            if (yInd > 0) {
                // there is a region below; get its top images
                if (!blImageSumPair.first) {
                    blImageSumPair = retList[retInd-nx]->getImageSumPair(TOP_LEFT);
                }
                brImageSumPair = retList[retInd-nx]->getImageSumPair(TOP_RIGHT);
            }

            retList.push_back(KernelImagesForRegion::ConstPtr(new KernelImagesForRegion(
                _kernelPtr,
                afwGeom::BoxI(corner, afwGeom::Extent2I::make(width, height)),
                _xy0,
                _doNormalize,
                blImageSumPair,
                brImageSumPair,
                tlImageSumPair,
                trImageNullPtr)));
            corner += afwGeom::Extent2I::make(width, 0);
        }
        leftCorner += afwGeom::Extent2I::make(0, height);
    }
    return retList;
}

/**
 * Compute the linearly interpolated image at the specified location (not a corner).
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if location is a corner,
 * i.e. is not one of: BOTTOM, TOP, LEFT, RIGHT, CENTER
 * @throw lsst::pex::exceptions::InvalidParameterException if outImage is not same dimensions as kernel.
 */
void mathDetail::KernelImagesForRegion::interpolateImage(
        Image &outImage,    ///< output image
        Location location)  ///< location at which to compute interpolated image
const {
    double fracDist;
    switch (location) {
        case BOTTOM: {
            fracDist = _centerFractionalPosition.getX();
            scaledPlus(outImage, 1.0 - fracDist, *(getImageSumPair(BOTTOM_LEFT).first),
                                       fracDist, *(getImageSumPair(BOTTOM_RIGHT)).first);
            break;
        }
        case TOP: {
                fracDist = _centerFractionalPosition.getX();
                scaledPlus(outImage, 1.0 - fracDist, *(getImageSumPair(TOP_LEFT).first),
                                           fracDist, *(getImageSumPair(TOP_RIGHT)).first);
            break;
        }
        case LEFT: {
            fracDist = _centerFractionalPosition.getY();
            scaledPlus(outImage, 1.0 - fracDist, *(getImageSumPair(BOTTOM_LEFT).first),
                                       fracDist, *(getImageSumPair(TOP_LEFT)).first);
            break;
        }
        case RIGHT: {
            fracDist = _centerFractionalPosition.getY();
            scaledPlus(outImage, 1.0 - fracDist, *(getImageSumPair(BOTTOM_RIGHT).first),
                                       fracDist, *(getImageSumPair(TOP_RIGHT)).first);
            break;
        }
        case CENTER: {
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
                Image::const_x_iterator const blEnd = getImageSumPair(BOTTOM_LEFT).first->row_end(y);
                Image::const_x_iterator blIter = getImageSumPair(BOTTOM_LEFT).first->row_begin(y);
                Image::const_x_iterator brIter = getImageSumPair(BOTTOM_RIGHT).first->row_begin(y);
                Image::const_x_iterator tlIter = getImageSumPair(TOP_LEFT).first->row_begin(y);
                Image::const_x_iterator trIter = getImageSumPair(TOP_RIGHT).first->row_begin(y);
                Image::x_iterator outIter = outImage.row_begin(y);
                for (; blIter != blEnd; ++blIter, ++brIter, ++tlIter, ++trIter, ++outIter) {
                    *outIter = 
                          (((*blIter * (1.0 - xFrac)) + (*brIter * xFrac)) * (1.0 - yFrac))
                        + (((*tlIter * (1.0 - xFrac)) + (*trIter * xFrac)) * yFrac);
                }
            }
            break;
        }
        default: {
            std::ostringstream os;
            os << "location = " << location << " is a corner";
            throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
        }
    }
}

/**
 * Will linear interpolation give a sufficiently accurate kernel image?
 *
 * The algorithm is as follows:
 * - for each location in (center, bottom, left, right, top):
 *     - for each pixel of the kernel:
 *         - if abs(linearly interpolated kernel image pixel - true kernel image pixel)
 *           > maxInterpolationError * true kernel sum then:
 *              return false (interpolation is unacceptable)
 * - return true (interpolation is acceptable)
 *
 * This is not completely foolproof, but it should do if you are careful not to test too large a region
 * relative to the wiggliness of the kernel's spatial model.
 *
 * @return true if interpolation will give sufficiently accurate results, false otherwise
 */
bool mathDetail::KernelImagesForRegion::isInterpolationOk(
        double maxInterpolationError)   ///< maximum allowed error
            ///< in computing the value of the kernel at any pixel by linear interpolation
const {
    typedef LocationList::const_iterator LocationIter;
    typedef Image::const_x_iterator ImageXIter;
    
    std::pair<int, int> const kernelDim = _kernelPtr->getDimensions();
    Image interpImage(kernelDim);
    for (LocationIter locIter = _TestLocationList.begin(); locIter != _TestLocationList.end(); ++locIter) {
        interpolateImage(interpImage, *locIter);
        ImageSumPair trueImageSumPair(getImageSumPair(*locIter));
        double const maxAllowedDiff = trueImageSumPair.second * maxInterpolationError;
        for (int row = 0; row < kernelDim.second; ++row) {
            ImageXIter interpPtr = interpImage.row_begin(row);
            ImageXIter const interpEnd = interpImage.row_end(row);
            ImageXIter truePtr = trueImageSumPair.first->row_begin(row);
            for ( ; interpPtr != interpEnd; ++interpPtr, ++truePtr) {
                if (std::abs(*interpPtr - *truePtr) > maxAllowedDiff) {
                    return false;
                }
            }
        }
    }
    return true;
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
        bbox.getMinX() + _computeNextSubregionLength(bbox.getWidth(), 2),
        bbox.getMinY() + _computeNextSubregionLength(bbox.getHeight(), 2)
    );
}

/**
 * Compute length of each subregion for a region divided into nDivisions pieces of approximately equal
 * length.
 *
 * @return a list of subspan lengths
 *
 * @throw lsst::pex::exceptions::InvalidParameterException if nDivisions >= length
 */
std::vector<int> mathDetail::KernelImagesForRegion::_computeSubregionLengths(
    int length,     ///< length of region
    int nDivisions) ///< number of divisions of region
{
    if ((nDivisions > length) || (nDivisions < 1)) {
        std::ostringstream os;
        os << "nDivisions = " << nDivisions << " not in range [1, " << length << " = length]";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    }
    std::vector<int> regionLengths;
    int remLength = length;
    for (int remNDiv = nDivisions; remNDiv > 0; --remNDiv) {
        int subLength = _computeNextSubregionLength(remLength, remNDiv);
        if (subLength < 1) {
            std::ostringstream os;
            os << "Bug! _computeSubregionLengths(length=" << length << ", nDivisions=" << nDivisions <<
                ") computed sublength = " << subLength << " < 0; remLength = " << remLength;
            throw LSST_EXCEPT(pexExcept::RuntimeErrorException, os.str());
        }
        regionLengths.push_back(subLength);
        remLength -= subLength;
    }
    return regionLengths;
}

int const mathDetail::KernelImagesForRegion::_MinInterpolationSize = 10;

mathDetail::KernelImagesForRegion::LocationList const mathDetail::KernelImagesForRegion::_TestLocationList =
    boost::assign::list_of(CENTER)(BOTTOM)(LEFT)(RIGHT)(TOP);

/**
 * @brief Construct a RowOfKernelImagesForRegion
 */
mathDetail::RowOfKernelImagesForRegion::RowOfKernelImagesForRegion(
        int nx, ///< number of columns
        int ny) ///< number of rows
:
    _nx(nx),
    _ny(ny),
    _yInd(-1),
    _regionList(ny)
{
    if ((nx < 1) || (ny < 1)) {
        std::ostringstream os;
        os << "nx = " << nx << " and/or ny = " << ny << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterException, os.str());
    };
}
