/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * \file
 * \brief Definition of default types for Mask%s and Variance Image%s
 */
#if !defined(LSST_AFW_LSST_IMAGE_TYPE_H)
#define LSST_AFW_LSST_IMAGE_TYPE_H 1

#include "boost/cstdint.hpp"

namespace lsst { namespace afw { namespace image {
    typedef boost::uint16_t MaskPixel;  ///! default type for Masks and MaskedImage Masks
    typedef float VariancePixel;        ///! default type for MaskedImage variance images
}}}
#endif
