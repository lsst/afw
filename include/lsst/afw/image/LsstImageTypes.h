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
 * Definition of default types for Mask%s and Variance Image%s
 */
#if !defined(LSST_AFW_LSST_IMAGE_TYPE_H)
#define LSST_AFW_LSST_IMAGE_TYPE_H 1

#include <cstdint>

namespace lsst {
namespace afw {
namespace image {
typedef std::uint32_t MaskPixel;  ///< default type for Masks and MaskedImage Masks
typedef float VariancePixel;      ///< default type for MaskedImage variance images
}
}
}
#endif
