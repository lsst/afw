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

#include <unordered_map>
#include <utility>

#include "lsst/afw/image/detail/StorableMap.h"

namespace lsst {
namespace afw {
namespace image {
namespace detail {

StorableMap::StorableMap() = default;
StorableMap::StorableMap(StorableMap const& other) = default;
StorableMap::StorableMap(StorableMap&& other) = default;
StorableMap& StorableMap::operator=(StorableMap const& other) = default;
StorableMap& StorableMap::operator=(StorableMap&& other) = default;
StorableMap::~StorableMap() noexcept = default;

StorableMap::StorableMap(std::initializer_list<value_type> init) : _contents(init){};

StorableMap::size_type StorableMap::size() const noexcept { return _contents.size(); }

bool StorableMap::empty() const noexcept { return _contents.empty(); }

StorableMap::size_type StorableMap::max_size() const noexcept { return _contents.max_size(); }

bool StorableMap::contains(std::string const& key) const { return _contents.count(key_type(key)) == 1; }

bool StorableMap::operator==(StorableMap const& other) const noexcept { return _contents == other._contents; }

void StorableMap::clear() noexcept { _contents.clear(); }

StorableMap::const_iterator StorableMap::begin() const noexcept {
    return StorableMap::const_iterator(_contents.begin());
}
StorableMap::const_iterator StorableMap::end() const noexcept {
    return StorableMap::const_iterator(_contents.end());
};

StorableMap::iterator StorableMap::begin() noexcept { return StorableMap::iterator(_contents.begin()); };
StorableMap::iterator StorableMap::end() noexcept { return StorableMap::iterator(_contents.end()); };

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst
