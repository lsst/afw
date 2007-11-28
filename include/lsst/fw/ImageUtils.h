// -*- lsst-c++ -*-
/**
 * \file
 *
 * \brief Image utility functions
 *
 * \defgroup fw LSST framework 
 */
#ifndef LSST_IMAGEUTILS_H
#define LSST_IMAGEUTILS_H

#include <cmath>

namespace lsst {
namespace fw {
namespace image {
    const double PixelZeroPos = 0.5; ///< position of center of pixel 0
    ///< FITS uses 1.0, SDSS uses 0.5, LSST is undecided but RHL proposed 0.0

    /**
     * \brief Convert image index to image position
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
     *
     * \return image position
     */
    inline double indexToPosition(
        int ind ///< image index
    ) {
        return static_cast<double>(ind) + PixelZeroPos;
    }
    
    /**
     * \brief Convert image position to nearest integer index
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
     *
     * \return nearest integer index
     */
    inline int positionToIndex(
        double pos ///< image position
    ) {
        return static_cast<int>(std::floor(pos + 0.5 - PixelZeroPos));
    }
    
    /**
     * \brief Convert image position to index (nearest integer and fractional parts)
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is PixelZeroPos, PixelZeroPos
     *
     * Note: in python this is called positionToIndexAndResidual
     *
     * \return nearest integer index
     */
    inline int positionToIndex(
        double &residual, ///< fractional part of index
        double pos ///< image position
    ) {
        double fullIndex = pos - PixelZeroPos;
        double roundedIndex = std::floor(fullIndex + 0.5);
        residual = fullIndex - roundedIndex;
        return static_cast<int>(roundedIndex);
    }     

}}} // lsst::fw::image

#endif // LSST_IMAGEUTILS_H

