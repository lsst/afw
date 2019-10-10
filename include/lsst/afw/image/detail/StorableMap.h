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

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
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

private:
    using _Impl = std::unordered_map<key_type, mapped_type>;

public:
    // These definitions may be replaced with adapters if we need
    // StorableMap's iterators to behave differently from _Impl's.
    using const_iterator = _Impl::const_iterator;
    using iterator = _Impl::iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    template <typename V>
    using Key = lsst::afw::typehandling::Key<std::string, V>;
    using Storable = lsst::afw::typehandling::Storable;

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

    /**
     * Return a the mapped value of the element with key equal to `key`.
     *
     * @tparam T the type of the element mapped to `key`. It may be the exact
     *           type of the element, if known, or any type to which its
     *           pointers can be implicitly converted (e.g., a superclass).
     * @param key the key of the element to find
     *
     * @return a pointer to the `T` mapped to `key`, if one exists
     *
     * @throws pex::exceptions::OutOfRangeError Thrown if the map does not
     *         have a `T` with the specified key.
     * @exceptsafe Provides strong exception safety.
     */
    template <typename T>
    std::shared_ptr<T> at(Key<std::shared_ptr<T>> const& key) const {
        static_assert(std::is_base_of<Storable, T>::value,
                      "Can only retrieve pointers to subclasses of Storable.");
        static_assert(std::is_const<T>::value,
                      "Due to implementation constraints, pointers to non-const are not supported.");
        try {
            // unordered_map::at(Key<Storable>) does not do any type-checking.
            mapped_type const& pointer = _contents.at(key);

            // Null pointer stored; skip dynamic_cast because won't change result.
            if (pointer == nullptr) {
                return nullptr;
            }

            std::shared_ptr<T> typedPointer = std::dynamic_pointer_cast<T>(pointer);
            // shared_ptr can be empty without being null. dynamic_pointer_cast
            // only promises result of failed cast is empty, so test for that.
            if (typedPointer.use_count() > 0) {
                return typedPointer;
            } else {
                std::stringstream message;
                message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
                throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
            }
        } catch (std::out_of_range const&) {
            std::throw_with_nested(LSST_EXCEPT(pex::exceptions::OutOfRangeError,
                                               "No key labeled " + key.getId() + " found."));
        }
    }

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
     * Return `true` if this map contains a mapping for the specified key.
     *
     * This is equivalent to testing whether `at(key)` would succeed.
     *
     * @tparam T the type of element being tested for
     * @param key the key to search for
     *
     * @return `true` if this map contains a mapping from the specified key to
     *         a `T`.
     *
     * @exceptsafe Provides strong exception safety.
     *
     */
    template <typename T>
    bool contains(Key<std::shared_ptr<T>> const& key) const {
        static_assert(std::is_base_of<Storable, T>::value,
                      "Can only retrieve pointers to subclasses of Storable.");
        static_assert(std::is_const<T>::value,
                      "Due to implementation constraints, pointers to non-const are not supported.");
        if (_contents.count(key) > 0) {
            // unordered_map::at(Key<Storable>) does not do any type-checking.
            mapped_type const& pointer = _contents.at(key);

            // Null pointer stored; dynamic_cast will always return null.
            if (pointer == nullptr) {
                return true;
            }

            std::shared_ptr<T> typedPointer = std::dynamic_pointer_cast<T>(pointer);
            // shared_ptr can be empty without being null. dynamic_pointer_cast
            // only promises result of failed cast is empty, so test for that.
            return typedPointer.use_count() > 0;
        } else {
            return false;
        }
    }

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

    /**
     * Insert an element into the map, if the map doesn't already contain a
     * mapping with the same or a conflicting key.
     *
     * @tparam T the type of value to insert
     * @param key the key to insert
     * @param value the value to insert
     *
     * @return `true` if the insertion took place, `false` otherwise.
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @note It is possible for a key with a value type other than `T` to
     *       prevent insertion. Callers can safely assume
     *       `this->contains(key.getId())` as a postcondition, but not
     *       `this->contains(key)`.
     */
    template <typename T>
    bool insert(Key<std::shared_ptr<T>> const& key, std::shared_ptr<T> const& value) {
        static_assert(std::is_base_of<Storable, T>::value,
                      "Can only store shared pointers to subclasses of Storable.");
        static_assert(std::is_const<T>::value,
                      "Due to implementation constraints, pointers to non-const are not supported.");
        // unordered_map uses Key<shared_ptr<Storable>> internally, so
        // any key with the same ID will block emplacement.
        return _contents.emplace(key, value).second;
    }

    /**
     * Insert an element into the map, if the map doesn't already contain a
     * mapping with a conflicting key.
     *
     * @tparam T the type of value to insert (will not compile unless shared
     *           pointer to a subclass of `Storable`)
     * @param key the key to insert
     * @param value the value to insert
     *
     * @return A pair consisting of a strongly-typed key for the value and a
     *         flag that is `true` if the insertion took place and
     *         `false` otherwise.
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @warning The type of the compiler-generated key may be surprising.
     *          Callers should save the returned key if they wish to retrieve
     *          the value later.
     */
    template <typename T>
    std::pair<Key<T>, bool> insert(std::string const& key, T const& value) {
        auto strongKey = typehandling::makeKey<T>(key);
        // Construct return value in advance, so that exception from
        // copying/moving Key is atomic.
        auto result = std::make_pair(strongKey, false);
        result.second = insert(strongKey, value);
        return result;
    }

    /**
     * Remove the mapping for a key from this map, if it exists.
     *
     * @tparam T the type of value the key maps to
     * @param key the key to remove
     *
     * @return `true` if `key` was removed, `false` if it was not present.
     */
    template <typename T>
    bool erase(Key<T> const& key) noexcept {
        // unordered_map::erase(Key<Storable>) does no type checking.
        if (this->contains(key)) {
            return _contents.erase(key) > 0;
        } else {
            return false;
        }
    }

    /**
     * Return an iterator to the first element of the map.
     *
     * @return An iterator that dereferences to a value_type, i.e. a pair of
     *         `const` Key and shared pointer to `const` Storable.
     *
     * @{
     */
    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept { return begin(); }

    /** @} */

    /**
     * Return an iterator to the element past the end of the map.
     *
     * @return An iterator that dereferences to a value_type, i.e. a pair of
     *         `const` Key and shared pointer to `const` Storable.
     *
     * @{
     */
    iterator end() noexcept;
    const_iterator end() const noexcept;
    const_iterator cend() const noexcept { return end(); }

    /** @} */

private:
    _Impl _contents;
};

}  // namespace detail
}  // namespace image
}  // namespace afw
}  // namespace lsst

#endif
