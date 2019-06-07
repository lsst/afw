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
 * In Python, a SimpleGenericMap behaves like a `dict`. In particular, it will
 * iterate over keys in the order they were added.
 *
 * @tparam K the key type of the map. Must be hashable.
 */
template <typename K>
class SimpleGenericMap final : public MutableGenericMap<K> {
protected:
    using typename GenericMap<K>::StorableType;
    using typename GenericMap<K>::ConstValueReference;

public:
    SimpleGenericMap() = default;
    SimpleGenericMap(SimpleGenericMap const& other) = default;
    SimpleGenericMap(SimpleGenericMap&&) noexcept = default;
    /**
     * Convert another GenericMap into a SimpleGenericMap.
     *
     * This constructor will insert key-value pairs following `other`'s
     * iteration order. This may not be the order in which they were inserted
     * into the original map.
     */
    SimpleGenericMap(GenericMap<K> const& other) : _storage(_convertStorage(other)), _keyView(other.keys()) {}
    virtual ~SimpleGenericMap() noexcept = default;

    SimpleGenericMap& operator=(SimpleGenericMap const& other) {
        std::vector<K> newKeys = other._keyView;
        _storage = other._storage;
        // strong exception safety because no exceptions can occur past this point
        using std::swap;
        swap(_keyView, newKeys);
        return *this;
    }
    SimpleGenericMap& operator=(SimpleGenericMap&&) noexcept = default;
    SimpleGenericMap& operator=(GenericMap<K> const& other) {
        std::vector<K> newKeys = other.keys();
        // strong exception safety: unordered_map is nothrow move-assignable and
        // vector is nothrow swappable, so no exceptions can occur after _convertStorage returns
        _storage = _convertStorage(other);
        using std::swap;
        swap(_keyView, newKeys);
        return *this;
    }

    typename GenericMap<K>::size_type size() const noexcept override { return _storage.size(); }

    bool empty() const noexcept override { return _storage.empty(); }

    typename GenericMap<K>::size_type max_size() const noexcept override {
        return std::min(_storage.max_size(), _keyView.max_size());
    }

    bool contains(K const& key) const override { return _storage.count(key) > 0; }

    std::vector<K> const& keys() const noexcept override { return _keyView; }

    void clear() noexcept override {
        _storage.clear();
        _keyView.clear();
    }

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
        std::vector<K> newKeys = _keyView;
        newKeys.emplace_back(key);
        bool inserted = _storage.emplace(key, std::move(value)).second;
        // strong exception safety because no exceptions can occur past this point
        if (inserted) {
            // _storage did not previously include key, so the key appended to newKeys is unique
            using std::swap;
            swap(_keyView, newKeys);
        }
        return inserted;
    }

    bool unsafeErase(K key) override {
        std::vector<K> newKeys = _keyView;
        for (auto it = newKeys.cbegin(); it != newKeys.cend();) {
            if (*it == key) {
                it = newKeys.erase(it);
            } else {
                ++it;
            }
        }
        // strong exception safety because no exceptions can occur past this point
        bool erased = _storage.erase(key) > 0;
        if (erased) {
            using std::swap;
            swap(_keyView, newKeys);
        }
        return erased;
    }

private:
    // StorableType is a value, so we might as well use it in the implementation
    std::unordered_map<K, StorableType> _storage;
    std::vector<K> _keyView;
    // Class invariant: the elements of _keyView are unique
    // Class invariant: the elements of _keyView and the keys of _storage are the same
    // Class invariant: the elements of _keyView are arranged in insertion order, oldest to newest

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
