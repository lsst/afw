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

#ifndef LSST_AFW_IMAGE_DETAIL_STORABLEMAP_H
#define LSST_AFW_IMAGE_DETAIL_STORABLEMAP_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/typehandling/Key.h"
#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace image {
namespace detail {

/**
 * A map of Storable supporting strongly-typed access.
 *
 * A Key for the map is parameterized by both the key type `K` and a
 * corresponding value type `V`. The map is indexed uniquely by a value of type
 * `K`; no two entries in the map may have identical values of Key::getId().
 *
 * All operations are sensitive to the value type of the key: a
 * @ref contains(Key<std::shared_ptr<T>> const&) const "contains" call
 * requesting a SkyWcs labeled "value", for example, will report no such
 * object if instead there is a Psf labeled "value". At present, a StorableMap
 * does not store type information internally, instead relying on RTTI for
 * type checking.
 */
class StorableMap final {
public:
    using mapped_type = std::shared_ptr<typehandling::Storable const>;
    using key_type = typehandling::Key<std::string, mapped_type>;
    using value_type = std::pair<key_type const, mapped_type>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = value_type*;
    using const_pointer = value_type const*;

    StorableMap();
    StorableMap(StorableMap const& other);
    StorableMap(StorableMap&& other);
    StorableMap& operator=(StorableMap const& other);
    StorableMap& operator=(StorableMap&& other);
    ~StorableMap() noexcept;

    /**
     * Construct a map from an initializer list.
     *
     * @param init an initializer list of key-value pairs. The keys and values
     *             may be of any type that can be converted to
     *             `std::shared_ptr<typehandling::Storable const>`.
     *
     * @note If `init` contains any keys with the same ID, it is unspecified
     *       which will be inserted.
     *
     * @see std::map::map
     */
    StorableMap(std::initializer_list<value_type> init);

    /// Return the number of key-value pairs in the map.
    size_type size() const noexcept;

    /// Return `true` if this map contains no key-value pairs.
    bool empty() const noexcept;

    /**
     * Return the maximum number of elements the container is able to hold due
     * to system or library implementation limitations.
     *
     * @note This value typically reflects the theoretical limit on the size
     *       of the container. At runtime, the size of the container may be
     *       limited to a value smaller than max_size() by the amount of
     *       RAM available.
     */
    size_type max_size() const noexcept;

    /**
     * Return `true` if this map contains a mapping whose key has the
     * specified label.
     *
     * More formally, this method returns `true` if and only if this map
     * contains a mapping with a key `k` such that `k.getId() == key`. There
     * can be at most one such mapping.
     *
     * @param key the weakly-typed key to search for
     *
     * @return `true` if this map contains a mapping for `key`, regardless of
     *         value type.
     *
     * @exceptsafe Provides strong exception safety.
     */
    bool contains(std::string const& key) const;

    /**
     * Test for map equality.
     *
     * Two StorableMap objects are considered equal if they map the same keys
     * to the same values.
     *
     * @{
     */
    bool operator==(StorableMap const& other) const noexcept;

    bool operator!=(StorableMap const& other) const noexcept { return !(*this == other); }

    /** @} */

    /**
     * Remove all of the mappings from this map.
     *
     * After this call, the map will be empty.
     */
    void clear() noexcept;

private:
    std::unordered_map<key_type, mapped_type> _contents;
};

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif
