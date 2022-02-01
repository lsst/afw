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

//
//##====----------------                                ----------------====##/
//
//        Formatting utilities
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_UTILS_H
#define LSST_AFW_FORMATTERS_UTILS_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "ndarray.h"

namespace lsst {
namespace afw {
namespace formatters {

/**
 * Encode a std::string as a vector of uint8
 */
ndarray::Array<std::uint8_t, 1, 1> stringToBytes(std::string const& str);

/**
 * Decode a std::string from a vector of uint8 returned by stringToBytes
 */
std::string bytesToString(ndarray::Array<std::uint8_t const, 1, 1> const& bytes);

}  // namespace formatters
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_FORMATTERS_UTILS_H
