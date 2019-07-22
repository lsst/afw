/*
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

#define BOOST_TEST_MODULE FunctorKeysCpp

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/included/unit_test.hpp"
#pragma clang diagnostic pop

#include "lsst/utils/tests.h"

#include "lsst/geom/Angle.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/arrays.h"
#include "lsst/afw/table/Schema.h"

/*
 * Unit tests for C++-only functionality in Key.
 */
namespace lsst {
namespace afw {
namespace table {

BOOST_AUTO_TEST_CASE(ArrayKeyHash) {
    utils::assertValidHash<ArrayKey<double>>();

    Schema schema;
    Key<double> a0 = schema.addField<double>("a_0", "");
    Key<double> a1 = schema.addField<double>("a_1", "");
    Key<double> a2 = schema.addField<double>("a_2", "");

    utils::assertHashesEqual(ArrayKey<double>(), ArrayKey<double>());
    utils::assertHashesEqual(ArrayKey<double>({a0, a1, a2}), ArrayKey<double>(schema["a"]));
}

BOOST_AUTO_TEST_CASE(PointKeyHash) {
    utils::assertValidHash<Point2IKey>();
    utils::assertValidHash<Point2DKey>();

    Schema schema;
    Key<double> aX = schema.addField<double>("a_x", "");
    Key<int> bX = schema.addField<int>("b_x", "");
    Key<int> bY = schema.addField<int>("b_y", "");
    Key<double> aY = schema.addField<double>("a_y", "");

    utils::assertHashesEqual(Point2DKey(), Point2DKey());
    utils::assertHashesEqual(Point2DKey(aX, aY), Point2DKey(schema["a"]));
    utils::assertHashesEqual(Point2IKey(bX, bY), Point2IKey(schema["b"]));
}

BOOST_AUTO_TEST_CASE(BoxKeyHash) {
    utils::assertValidHash<Box2IKey>();
    utils::assertValidHash<Box2DKey>();

    Schema schema;
    Key<double> aX = schema.addField<double>("a_min_x", "");
    Key<double> bX = schema.addField<double>("a_max_x", "");
    Key<double> bY = schema.addField<double>("a_max_y", "");
    Key<double> aY = schema.addField<double>("a_min_y", "");

    utils::assertHashesEqual(Box2IKey(), Box2IKey());
    utils::assertHashesEqual(Box2DKey(Point2DKey(schema["a_min"]), Point2DKey(bX, bY)),
                             Box2DKey(schema["a"]));
}

BOOST_AUTO_TEST_CASE(CoordKeyHash) {
    utils::assertValidHash<CoordKey>();

    Schema schema;
    Key<lsst::geom::Angle> aRa = schema.addField<lsst::geom::Angle>("a_ra", "");
    Key<lsst::geom::Angle> aDec = schema.addField<lsst::geom::Angle>("a_dec", "");

    utils::assertHashesEqual(CoordKey(), CoordKey());
    utils::assertHashesEqual(CoordKey(aRa, aDec), CoordKey(schema["a"]));
    utils::assertHashesEqual(CoordKey(aRa, aDec), CoordKey(schema["a"]));
}

BOOST_AUTO_TEST_CASE(QuadrupoleKeyHash) {
    utils::assertValidHash<QuadrupoleKey>();

    Schema schema;
    Key<double> aXX = schema.addField<double>("a_xx", "");
    Key<double> aYY = schema.addField<double>("a_yy", "");
    Key<double> aXY = schema.addField<double>("a_xy", "");

    utils::assertHashesEqual(QuadrupoleKey(), QuadrupoleKey());
    utils::assertHashesEqual(QuadrupoleKey(aXX, aYY, aXY), QuadrupoleKey(schema["a"]));
}

BOOST_AUTO_TEST_CASE(EllipseKeyHash) {
    utils::assertValidHash<EllipseKey>();

    Schema schema;
    Key<double> aXX = schema.addField<double>("a_xx", "");
    Key<double> aY = schema.addField<double>("a_y", "");
    Key<double> aYY = schema.addField<double>("a_yy", "");
    Key<double> aX = schema.addField<double>("a_x", "");
    Key<double> aXY = schema.addField<double>("a_xy", "");

    utils::assertHashesEqual(EllipseKey(), EllipseKey());
    utils::assertHashesEqual(EllipseKey(QuadrupoleKey(aXX, aYY, aXY), Point2DKey(aX, aY)),
                             EllipseKey(schema["a"]));
}

BOOST_AUTO_TEST_CASE(CovarianceMatrixKeyHash) {
    using namespace std::string_literals;

    utils::assertValidHash<CovarianceMatrixKey<double, 5>>();
    utils::assertValidHash<CovarianceMatrixKey<float, Eigen::Dynamic>>();

    Schema schema;
    Key<double> a00 = schema.addField<double>("a_fooErr", "");
    Key<double> a11 = schema.addField<double>("a_barErr", "");
    Key<double> a22 = schema.addField<double>("a_cowErr", "");
    Key<double> a01 = schema.addField<double>("a_foo_bar_Cov", "");
    Key<double> a02 = schema.addField<double>("a_foo_cow_Cov", "");
    Key<double> a12 = schema.addField<double>("a_bar_cow_Cov", "");
    auto names = {"foo"s, "bar"s, "cow"s};

    utils::assertHashesEqual(CovarianceMatrixKey<float, 2>(), CovarianceMatrixKey<float, 2>());
    utils::assertHashesEqual(CovarianceMatrixKey<double, 3>({a00, a11, a22}, {a01, a02, a12}),
                             CovarianceMatrixKey<double, 3>(schema["a"], names));
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
