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
#include "boost/test/unit_test.hpp"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/table/io/Persistable.h"

#include <sstream>
#include <type_traits>
BOOST_AUTO_TEST_CASE(Defaults) {
    using t= lsst::afw::geom::Transform<lsst::afw::geom::Point2Endpoint, lsst::afw::geom::Point2Endpoint>;
    ast::FrameSet f(ast::Frame(2));
    std::shared_ptr<lsst::afw::table::io::Persistable> p = std::make_shared<t>(f);
    std::cout << "p = " << p << "\n";
    std::shared_ptr<t> p2 = std::dynamic_pointer_cast<t>(p);
    std::cout << "p2 = " << p2 << "\n";
    std::cout << "is base of " << std::is_base_of_v<lsst::afw::table::io::Persistable, t> << "\n";
    std::cout << "Persistable is polymorphic: "
          << std::is_polymorphic_v<lsst::afw::table::io::Persistable> << "\n";
    std::cout << "Actual type in p: " << typeid(*p).name() << std::endl;
    BOOST_CHECK(p2 != nullptr);
}
