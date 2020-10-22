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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE FilterLabelCpp
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop

#include <string>

#include "lsst/utils/tests.h"

#include "lsst/afw/image/FilterLabel.h"

namespace lsst {
namespace afw {
namespace image {

BOOST_AUTO_TEST_CASE(Hash) {
    std::string const band = "k";
    std::string const physical = "MyScope-K";

    utils::assertValidHash<FilterLabel>();

    utils::assertHashesEqual(FilterLabel::fromBand(band), FilterLabel::fromBand(band));
    utils::assertHashesEqual(FilterLabel::fromPhysical(physical), FilterLabel::fromPhysical(physical));
    utils::assertHashesEqual(FilterLabel::fromBandPhysical(band, physical),
                             FilterLabel::fromBandPhysical(band, physical));

    // There are multiple ways to represent a label without a band/filter; these can arise
    // from e.g. depersistence. Such objects compare equal; ensure they hash equally as well.
    utils::assertHashesEqual(impl::makeTestFilterLabel(false, "", true, physical),
                             impl::makeTestFilterLabel(false, "null", true, physical));
    utils::assertHashesEqual(impl::makeTestFilterLabel(true, band, false, ""),
                             impl::makeTestFilterLabel(true, band, false, "undefined"));
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
