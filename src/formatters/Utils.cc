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
// Support for formatters
//
//##====----------------                                ----------------====##/

#include <cstdint>
#include <iostream>

#include "lsst/afw/formatters/Utils.h"

using std::int64_t;

namespace lsst {
namespace afw {
namespace formatters {

ndarray::Array<std::uint8_t, 1, 1> stringToBytes(std::string const& str) {
    auto nbytes = str.size() * sizeof(char) / sizeof(std::uint8_t);
    std::uint8_t const* byteCArr = reinterpret_cast<std::uint8_t const*>(str.data());
    auto shape = ndarray::makeVector(nbytes);
    auto strides = ndarray::makeVector(1);
    // Make an Array that shares memory with `str` (and does not free that memory when destroyed),
    // then return a copy; this is simpler than manually copying the data into a newly allocated array
    ndarray::Array<std::uint8_t const, 1, 1> localArray = ndarray::external(byteCArr, shape, strides);
    return ndarray::copy(localArray);
}

std::string bytesToString(ndarray::Array<std::uint8_t const, 1, 1> const& bytes) {
    auto nchars = bytes.size() * sizeof(std::uint8_t) / sizeof(char);
    char const* charCArr = reinterpret_cast<char const*>(bytes.getData());
    return std::string(charCArr, nchars);
}

}  // namespace formatters
}  // namespace afw
}  // namespace lsst
