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

namespace lsst {
namespace fw {
namespace image {

    /**
     * \brief Convert image index to image position
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is 0.5, 0.5
     *
     * \return image position
     */
    inline double indexToPosition(
        int ind ///< image index
    ) {
        return static_cast<double>(ind) + 0.5;
    }
    
    /**
     * \brief Convert image index to image position
     *
     * The LSST indexing convention is:
     * * the index of the bottom left pixel is 0,0
     * * the position of the center of the bottom left pixel is 0.5, 0.5
     *
     * \return image index
     */
    inline int positionToIndex(
        double pos ///< image position
    ) {
        return static_cast<int>(pos);
    }

} // namespace image
} // namespace fw
} // namespace lsst

#endif // LSST_IMAGEUTILS_H

