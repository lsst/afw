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

#ifndef LSST_AFW_TYPEHANDLING_SIMPLEGENERICMAP_H
#define LSST_AFW_TYPEHANDLING_SIMPLEGENERICMAP_H

#include <exception>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "boost/variant.hpp"

#include "lsst/afw/typehandling/GenericMap.h"

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * A GenericMap that allows insertion and deletion of arbitrary values.
 *
 * In Python, a SimpleGenericMap behaves like a `dict`.
 *
 * @tparam K the key type of the map. Must be hashable.
 *
 * @note This class offers no guarantees, such as thread-safety, beyond those
 *       provided by MutableGenericMap.
 */
template <typename K>
class SimpleGenericMap final : public MutableGenericMap<K> {
protected:
    using typename GenericMap<K>::StorableType;
    using typename GenericMap<K>::ConstValueReference;

public:
    SimpleGenericMap() = default;
    SimpleGenericMap(SimpleGenericMap const& other) = default;
    SimpleGenericMap(SimpleGenericMap&&) = default;
    SimpleGenericMap(GenericMap<K> const& other) : _storage(_convertStorage(other)) {}
    virtual ~SimpleGenericMap() noexcept = default;

    SimpleGenericMap& operator=(SimpleGenericMap const& other) = default;
    SimpleGenericMap& operator=(SimpleGenericMap&&) = default;
    SimpleGenericMap& operator=(GenericMap<K> const& other) {
        // Atomic because unordered_map is nothrow move-assignable,
        // so only _convertStorage can fail
        _storage = _convertStorage(other);
        return *this;
    }

    typename GenericMap<K>::size_type size() const noexcept override { return _storage.size(); }

    bool empty() const noexcept override { return _storage.empty(); }

    typename GenericMap<K>::size_type max_size() const noexcept override { return _storage.max_size(); }

    bool contains(K const& key) const override { return _storage.count(key) > 0; }

    std::vector<K> keys() const override {
        std::vector<K> keySnapshot;
        keySnapshot.reserve(_storage.size());
        for (auto const& pair : _storage) {
            keySnapshot.push_back(pair.first);
        }
        return keySnapshot;
    }

    void clear() noexcept override { _storage.clear(); }

protected:
    ConstValueReference unsafeLookup(K key) const override {
        try {
            return _storage.at(key);
        } catch (std::out_of_range& e) {
            std::stringstream message;
            message << "Key not found: " << key;
            std::throw_with_nested(LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str()));
        }
    }

    bool unsafeInsert(K key, StorableType&& value) override {
        return _storage.emplace(key, std::move(value)).second;
    }

    bool unsafeErase(K key) override { return _storage.erase(key); }

private:
    // StorableType is a value, so we might as well use it in the implementation
    std::unordered_map<K, StorableType> _storage;

    /**
     * Create a new back-end map that contains the same mappings as a GenericMap.
     *
     * @param map The map whose contents should be copied.
     * @return a new map with the same mappings as `map`.
     *
     * @exceptsafe Provides strong exception-safety.
     */
    static std::unordered_map<K, StorableType> _convertStorage(GenericMap<K> const& map) {
        std::unordered_map<K, StorableType> newStorage;
        map.apply([&newStorage](K const& key, auto const& value) { newStorage.emplace(key, value); });
        return newStorage;
    }
};

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
