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
#include <iostream>
#include <sstream>
#include <vector>

#include "boost/assign/list_of.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/ImageUtils.h"
#include "lsst/afw/math/detail/Convolve.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace mathDetail = lsst::afw::math::detail;

/**
 * Construct a KernelImagesForRegion
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if kernelPtr is null
 */
mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr kernelPtr,               ///< kernel
        lsst::afw::geom::Box2I const &bbox,      ///< bounding box of region of an image
                                                ///< for which we want to compute kernel images
                                                ///< (inclusive and relative to parent image)
        lsst::afw::geom::Point2I const &xy0,    ///< xy0 of image for which we want to compute kernel images
        bool doNormalize)                       ///< normalize the kernel images?
:
    lsst::daf::base::Citizen(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _xy0(xy0),
    _doNormalize(doNormalize),
    _imagePtrList(4)
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "kernelPtr is null");
    }
    LOGF_TRACE6("lsst.afw.math.convolve",
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
 * @throw lsst::pex::exceptions::InvalidParameterError if kernelPtr is null
 * @throw lsst::pex::exceptions::InvalidParameterError if an image has the wrong dimensions
 */
mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr const kernelPtr,         ///< kernel
        lsst::afw::geom::Box2I const &bbox,      ///< bounding box of region of an image
                                                ///< for which we want to compute kernel images
                                                ///< (inclusive and relative to parent image)
        lsst::afw::geom::Point2I const &xy0,    ///< xy0 of image
        bool doNormalize,                       ///< normalize the kernel images?
        ImagePtr bottomLeftImagePtr,            ///< kernel image and sum at bottom left of region
        ImagePtr bottomRightImagePtr,           ///< kernel image and sum at bottom right of region
        ImagePtr topLeftImagePtr,               ///< kernel image and sum at top left of region
        ImagePtr topRightImagePtr)              ///< kernel image and sum at top right of region
:
    lsst::daf::base::Citizen(typeid(this)),
    _kernelPtr(kernelPtr),
    _bbox(bbox),
    _xy0(xy0),
    _doNormalize(doNormalize),
    _imagePtrList(4)
{
    if (!_kernelPtr) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, "kernelPtr is null");
    }
    _insertImage(BOTTOM_LEFT, bottomLeftImagePtr);
    _insertImage(BOTTOM_RIGHT, bottomRightImagePtr);
    _insertImage(TOP_LEFT, topLeftImagePtr);
    _insertImage(TOP_RIGHT, topRightImagePtr);
    LOGF_TRACE6("lsst.afw.math.convolve",
    "KernelImagesForRegion(bbox(minimum=(%d, %d), extent=(%d, %d)), xy0=(%d, %d), doNormalize=%d, images...)",
       _bbox.getMinX(), _bbox.getMinY(), _bbox.getWidth(), _bbox.getHeight(), _xy0[0], _xy0[1], _doNormalize);
}

/**
 * Return the image and sum at the specified location
 *
 * If the image has not yet been computed, it is computed at this time.
 */
mathDetail::KernelImagesForRegion::ImagePtr mathDetail::KernelImagesForRegion::getImage(
        Location location)  ///< location of image
const {
    if (_imagePtrList[location]) {
        return _imagePtrList[location];
    }

    ImagePtr imagePtr(new Image(_kernelPtr->getDimensions()));
    _imagePtrList[location] = imagePtr;
    _computeImage(location);
    return imagePtr;
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
        case BOTTOM_RIGHT:
            return afwGeom::Point2I(_bbox.getMaxX() + 1, _bbox.getMinY());
            break; // paranoia
        case TOP_LEFT:
            return afwGeom::Point2I(_bbox.getMinX(), _bbox.getMaxY() + 1);
            break; // paranoia
        case TOP_RIGHT:
            return afwGeom::Point2I(_bbox.getMaxX() + 1, _bbox.getMaxY() + 1);
            break; // paranoia
        default: {
            std::ostringstream os;
            os << "Bug: unhandled location = " << location;
            throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
        }
    }
}

/**
 * @brief Compute next row of subregions
 *
 * For the first row call with a new RowOfKernelImagesForRegion (with the desired number of columns and rows).
 * Every subequent call updates the data in the RowOfKernelImagesForRegion.
 *
 * @return true if a new row was computed, false if supplied RowOfKernelImagesForRegion is for the last row.
 */
bool mathDetail::KernelImagesForRegion::computeNextRow(
        RowOfKernelImagesForRegion &regionRow) ///< RowOfKernelImagesForRegion object
const {
    if (regionRow.isLastRow()) {
        return false;
    }

    bool hasData = regionRow.hasData();
    int startY;
    if (hasData) {
        startY = regionRow.front()->getBBox().getMaxY() + 1;
    } else {
        startY = this->_bbox.getMinY();
    }

    int yInd = regionRow.incrYInd();
    int remHeight = 1 + this->_bbox.getMaxY() - startY;
    int remYDiv = regionRow.getNY() - yInd;
    int height = _computeNextSubregionLength(remHeight, remYDiv);

    if (hasData) {
        // Move each region up one segment
        bool isFirst = true;
        for (RowOfKernelImagesForRegion::Iterator rgnIter = regionRow.begin(), rgnEnd = regionRow.end();
            rgnIter != rgnEnd; ++rgnIter) {
            (*rgnIter)->_moveUp(isFirst, height);
            isFirst = false;
        }

    } else {
        ImagePtr blImagePtr = getImage(BOTTOM_LEFT);
        ImagePtr brImagePtr;
        ImagePtr tlImagePtr;
        ImagePtr const trImageNullPtr;

        afwGeom::Point2I blCorner = afwGeom::Point2I(this->_bbox.getMinX(), startY);

        int remWidth = this->_bbox.getWidth();
        int remXDiv = regionRow.getNX();
        for (RowOfKernelImagesForRegion::Iterator rgnIter = regionRow.begin(), rgnEnd = regionRow.end();
            rgnIter != rgnEnd; ++rgnIter) {
            int width = _computeNextSubregionLength(remWidth, remXDiv);
            --remXDiv;
            remWidth -= width;

            KernelImagesForRegion::Ptr regionPtr(new KernelImagesForRegion(
                _kernelPtr,
                afwGeom::Box2I(blCorner, afwGeom::Extent2I(width, height)),
                _xy0,
                _doNormalize,
                blImagePtr,
                brImagePtr,
                tlImagePtr,
                trImageNullPtr));
            *rgnIter = regionPtr;

            if (!tlImagePtr) {
                regionPtr->getImage(TOP_LEFT);
            }

            blCorner += afwGeom::Extent2I(width, 0);
            blImagePtr = regionPtr->getImage(BOTTOM_RIGHT);
            tlImagePtr = regionPtr->getImage(TOP_RIGHT);
        }
    }
    return true;
}

/**
 * Compute image at a particular location
 *
 * @throw lsst::pex::exceptions::NotFoundError if there is no pointer at that location
 */
void mathDetail::KernelImagesForRegion::_computeImage(Location location) const {
    ImagePtr imagePtr = _imagePtrList[location];
    if (!imagePtr) {
        std::ostringstream os;
        os << "Null imagePtr at location " << location;
        throw LSST_EXCEPT(pexExcept::NotFoundError, os.str());
    }

    afwGeom::Point2I pixelIndex = getPixelIndex(location);
    _kernelPtr->computeImage(
        *imagePtr,
        _doNormalize,
        afwImage::indexToPosition(pixelIndex.getX() + _xy0[0]),
        afwImage::indexToPosition(pixelIndex.getY() + _xy0[1]));
}

/**
 * Compute length of each subregion for a region divided into nDivisions pieces of approximately equal
 * length.
 *
 * @return a list of subspan lengths
 *
 * @throw lsst::pex::exceptions::InvalidParameterError if nDivisions >= length
 */
std::vector<int> mathDetail::KernelImagesForRegion::_computeSubregionLengths(
    int length,     ///< length of region
    int nDivisions) ///< number of divisions of region
{
    if ((nDivisions > length) || (nDivisions < 1)) {
        std::ostringstream os;
        os << "nDivisions = " << nDivisions << " not in range [1, " << length << " = length]";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    }
    std::vector<int> regionLengths;
    int remLength = length;
    for (int remNDiv = nDivisions; remNDiv > 0; --remNDiv) {
        int subLength = _computeNextSubregionLength(remLength, remNDiv);
        if (subLength < 1) {
            std::ostringstream os;
            os << "Bug! _computeSubregionLengths(length=" << length << ", nDivisions=" << nDivisions <<
                ") computed sublength = " << subLength << " < 0; remLength = " << remLength;
            throw LSST_EXCEPT(pexExcept::RuntimeError, os.str());
        }
        regionLengths.push_back(subLength);
        remLength -= subLength;
    }
    return regionLengths;
}

/**
 * @brief Move the region up one segment
 *
 * To avoid reallocating memory for kernel images, swap the top and bottom kernel image pointers
 * and recompute the top images. Actually, only does this to the right-hande images if isFirst is false
 * since it assumes the left images were already handled.
 *
 * Intended to support computeNextRow; as such assumes that a list of adjacent regions will be moved,
 * left to right.
 */
void mathDetail::KernelImagesForRegion::_moveUp(
        bool isFirst,   ///< true if the first region in a row (or the only region you are moving)
        int newHeight)  ///< new height of region
{
    // move bbox up (this must be done before recomputing the top kernel images)
    _bbox = afwGeom::Box2I(
        afwGeom::Point2I(_bbox.getMinX(), _bbox.getMaxY() + 1),
        afwGeom::Extent2I(_bbox.getWidth(), newHeight));

    // swap top and bottom image pointers
    _imagePtrList[BOTTOM_RIGHT].swap(_imagePtrList[TOP_RIGHT]);
    _imagePtrList[BOTTOM_LEFT].swap(_imagePtrList[TOP_LEFT]);

    // recompute top right, and if the first image also recompute top left
    _computeImage(TOP_RIGHT);
    if (isFirst) {
        _computeImage(TOP_LEFT);
    }
}


int const mathDetail::KernelImagesForRegion::_MinInterpolationSize = 10;

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
    _regionList(nx)
{
    if ((nx < 1) || (ny < 1)) {
        std::ostringstream os;
        os << "nx = " << nx << " and/or ny = " << ny << " < 1";
        throw LSST_EXCEPT(pexExcept::InvalidParameterError, os.str());
    };
}
