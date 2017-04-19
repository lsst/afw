// -*- lsst-c++ -*-

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
 * Image utility functions
 */
#ifndef LSST_AFW_IMAGE_IMAGEUTILS_H
#define LSST_AFW_IMAGE_IMAGEUTILS_H

#include <cmath>

namespace lsst {
namespace afw {
namespace image {
enum xOrY { X, Y };

/**
 * position of center of pixel 0
 *
 * FITS uses 1.0, SDSS uses 0.5, LSST uses 0.0
 * (http://dev.lsstcorp.org/trac/wiki/BottomLeftPixelProposalII%3A)
 */
const double PixelZeroPos = 0.0;

/**
 * Convert image index to image position
 *
 * The LSST indexing convention is:
 * * the index of the bottom left pixel is 0,0
 * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
 *
 * @returns image position
 */
inline double indexToPosition(double ind  ///< image index
                              ) {
    return ind + PixelZeroPos;
}

/**
 * Convert image position to nearest integer index
 *
 * The LSST indexing convention is:
 * * the index of the bottom left pixel is 0,0
 * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
 *
 * @returns nearest integer index
 */
inline int positionToIndex(double pos  ///< image position
                           ) {
    return static_cast<int>(std::floor(pos + 0.5 - PixelZeroPos));
}

/**
 * Convert image position to index (nearest integer and fractional parts)
 *
 * The LSST indexing convention is:
 * * the index of the bottom left pixel is 0,0
 * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
 *
 * Note: in python this is called positionToIndexAndResidual
 *
 * @returns nearest integer index
 */
inline int positionToIndex(double &residual,  ///< fractional part of index
                           double pos         ///< image position
                           ) {
    double fullIndex = pos - PixelZeroPos;
    double roundedIndex = std::floor(fullIndex + 0.5);
    residual = fullIndex - roundedIndex;
    return static_cast<int>(roundedIndex);
}
/**
 * Convert image position to index (nearest integer and fractional parts)
 *
 * @returns std::pair(nearest integer index, fractional part)
 */
inline std::pair<int, double> positionToIndex(double const pos,  ///< image position
                                              bool               ///< ignored; just to disambiguate
                                              ) {
    double residual;                                 // fractional part of index
    int const ind = positionToIndex(residual, pos);  // integral part

    return std::pair<int, double>(ind, residual);
}
}
}
}  // lsst::afw::image

#endif  // LSST_AFW_IMAGE_IMAGEUTILS_H
