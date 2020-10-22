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

#include <memory>

#include "lsst/utils/hashCombine.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/FilterLabel.h"

using namespace std::string_literals;

namespace lsst {
namespace afw {
namespace image {

namespace impl {
// Hack to allow unit tests to test states that, while legal, are
// not produced by (and should not be required of) standard factories.
FilterLabel makeTestFilterLabel(bool hasBand, std::string const &band, bool hasPhysical,
                                std::string const &physical) {
    // private constructor accessible via friend
    return FilterLabel(hasBand, band, hasPhysical, physical);
}
}  // namespace impl

FilterLabel::FilterLabel(bool hasBand, std::string const &band, bool hasPhysical, std::string const &physical)
        : _hasBand(hasBand), _hasPhysical(hasPhysical), _band(band), _physical(physical) {}

FilterLabel FilterLabel::fromBandPhysical(std::string const &band, std::string const &physical) {
    return FilterLabel(true, band, true, physical);
}

FilterLabel FilterLabel::fromBand(std::string const &band) { return FilterLabel(true, band, false, ""s); }

FilterLabel FilterLabel::fromPhysical(std::string const &physical) {
    return FilterLabel(false, ""s, true, physical);
}

// defaults give the right behavior with bool-and-string implementation
FilterLabel::FilterLabel(FilterLabel const &) = default;
FilterLabel::FilterLabel(FilterLabel &&) noexcept = default;
FilterLabel &FilterLabel::operator=(FilterLabel const &) = default;
FilterLabel &FilterLabel::operator=(FilterLabel &&) noexcept = default;
FilterLabel::~FilterLabel() noexcept = default;

bool FilterLabel::hasBandLabel() const noexcept { return _hasBand; }

std::string FilterLabel::getBandLabel() const {
    // In no implementation I can think of will hasBandLabel() be an expensive test.
    if (hasBandLabel()) {
        return _band;
    } else {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "FilterLabel has no band."s);
    }
}

bool FilterLabel::hasPhysicalLabel() const noexcept { return _hasPhysical; }

std::string FilterLabel::getPhysicalLabel() const {
    // In no implementation I can think of will hasBandLabel() be an expensive test.
    if (hasPhysicalLabel()) {
        return _physical;
    } else {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "FilterLabel has no physical filter."s);
    }
}

bool FilterLabel::operator==(FilterLabel const &rhs) const noexcept {
    // Do not compare name unless _hasName for both
    if (_hasBand != rhs._hasBand) {
        return false;
    }
    if (_hasBand && _band != rhs._band) {
        return false;
    }
    if (_hasPhysical != rhs._hasPhysical) {
        return false;
    }
    if (_hasPhysical && _physical != rhs._physical) {
        return false;
    }
    return true;
}

// Storable support

std::size_t FilterLabel::hash_value() const noexcept {
    // Do not count _name unless _hasName
    // (_has=false, _name="A") and (_has=false, _name="B") compare equal, so must have same hash
    return utils::hashCombine(42, _hasBand, _hasBand ? _band : ""s, _hasPhysical,
                              _hasPhysical ? _physical : ""s);
}

/* The implementation is biased toward Python in its format, but I expect
 * the C++ calls to mostly be used for debugging rather than presentation.
 * This class is also too simple to need "long" and "short" string forms.
 */
std::string FilterLabel::toString() const {
    std::string buffer("FilterLabel(");
    bool comma = false;

    if (hasBandLabel()) {
        if (comma) buffer += ", "s;
        buffer += "band"s + "=\""s + getBandLabel() + "\""s;
        comma = true;
    }
    if (hasPhysicalLabel()) {
        if (comma) buffer += ", "s;
        buffer += "physical"s + "=\""s + getPhysicalLabel() + "\""s;
        comma = true;
    }
    buffer += ")"s;

    return buffer;
}

std::shared_ptr<typehandling::Storable> FilterLabel::cloneStorable() const {
    return std::make_shared<FilterLabel>(*this);
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
