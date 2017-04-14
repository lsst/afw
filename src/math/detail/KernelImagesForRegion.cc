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

/*
 * Definition of KernelImagesForRegion class declared in detail/ConvolveImage.h
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

mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr kernelPtr,
        lsst::afw::geom::Box2I const &bbox,
        lsst::afw::geom::Point2I const &xy0,
        bool doNormalize)
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
    LOGL_DEBUG("TRACE5.afw.math.convolve.KernelImagesForRegion",
    "KernelImagesForRegion(bbox(minimum=(%d, %d), extent=(%d, %d)), xy0=(%d, %d), doNormalize=%d, images...)",
       _bbox.getMinX(), _bbox.getMinY(), _bbox.getWidth(), _bbox.getHeight(), _xy0[0], _xy0[1], _doNormalize);
}

mathDetail::KernelImagesForRegion::KernelImagesForRegion(
        KernelConstPtr const kernelPtr,
        lsst::afw::geom::Box2I const &bbox,
        lsst::afw::geom::Point2I const &xy0,
        bool doNormalize,
        ImagePtr bottomLeftImagePtr,
        ImagePtr bottomRightImagePtr,
        ImagePtr topLeftImagePtr,
        ImagePtr topRightImagePtr)
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
    LOGL_DEBUG("TRACE5.afw.math.convolve.KernelImagesForRegion",
    "KernelImagesForRegion(bbox(minimum=(%d, %d), extent=(%d, %d)), xy0=(%d, %d), doNormalize=%d, images...)",
       _bbox.getMinX(), _bbox.getMinY(), _bbox.getWidth(), _bbox.getHeight(), _xy0[0], _xy0[1], _doNormalize);
}

mathDetail::KernelImagesForRegion::ImagePtr mathDetail::KernelImagesForRegion::getImage(
        Location location)
const {
    if (_imagePtrList[location]) {
        return _imagePtrList[location];
    }

    ImagePtr imagePtr(new Image(_kernelPtr->getDimensions()));
    _imagePtrList[location] = imagePtr;
    _computeImage(location);
    return imagePtr;
}

lsst::afw::geom::Point2I mathDetail::KernelImagesForRegion::getPixelIndex(
        Location location)
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

bool mathDetail::KernelImagesForRegion::computeNextRow(
        RowOfKernelImagesForRegion &regionRow)
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

            std::shared_ptr<KernelImagesForRegion> regionPtr(new KernelImagesForRegion(
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

std::vector<int> mathDetail::KernelImagesForRegion::_computeSubregionLengths(
    int length,
    int nDivisions)
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

void mathDetail::KernelImagesForRegion::_moveUp(
        bool isFirst,
        int newHeight)
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

mathDetail::RowOfKernelImagesForRegion::RowOfKernelImagesForRegion(
        int nx,
        int ny)
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
