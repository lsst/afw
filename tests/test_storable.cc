// -*- LSST-C++ -*-
/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StorableCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <sstream>

#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace typehandling {

namespace {

class Dummy : public Storable {};

}  // namespace

BOOST_AUTO_TEST_CASE(Defaults) {
    Dummy dummy;
    BOOST_CHECK_THROW(dummy.cloneStorable(), UnsupportedOperationException);

    BOOST_TEST(!dummy.isPersistable());
    BOOST_TEST(!dummy.equals(Dummy()));

    std::stringstream buffer;
    BOOST_CHECK_THROW(buffer << dummy, UnsupportedOperationException);
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
