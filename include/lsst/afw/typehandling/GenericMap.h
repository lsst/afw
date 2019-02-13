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

#ifndef LSST_AFW_TYPEHANDLING_GENERICMAP_H
#define LSST_AFW_TYPEHANDLING_GENERICMAP_H

#include <functional>
#include <ostream>
#include <typeinfo>
#include <type_traits>

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * Key for type-safe lookup in a GenericMap.
 *
 * @tparam K the logical type of the key (e.g., a string)
 * @tparam V the type of the value mapped to this key
 *
 * Key objects are equality-comparable, hashable, sortable, or printable if and only if `K` is comparable,
 * hashable, sortable, or printable, respectively. Key can be used in compile-time expressions if and only
 * if `K` can (in particular, `Key<std::string, V>` cannot).
 *
 * @note Objects of this type are immutable.
 */
template <typename K, typename V>
class Key final {
public:
    using KeyType = K;
    using ValueType = V;

    /**
     * Construct a new key.
     *
     * @param id
     *          the identifier of the field. For most purposes, this value is
     *          the actual key; it can be retrieved by calling getId().
     *
     * @exceptsafe Provides the same exception safety as the copy-constructor of `K`.
     *
     * @see makeKey
     */
    constexpr Key(K id) : id(id) {}

    Key(Key const&) = default;
    Key(Key&&) = default;
    Key& operator=(Key const&) = delete;
    Key& operator=(Key&&) = delete;

    /**
     * Return the identifier of this field.
     *
     * The identifier serves as the "key" for the map abstraction
     * represented by GenericMap.
     *
     * @returns the unique key defining this field
     */
    constexpr K const& getId() const noexcept { return id; }

    /**
     * Test for key equality.
     *
     * A key is considered equal to another key if and only if their getId() are equal and their value
     * types are exactly the same (including const/volatile qualifications).
     *
     * @{
     */
    constexpr bool operator==(Key<K, V> const& other) const noexcept { return this->id == other.id; }

    template <typename U>
    constexpr std::enable_if_t<!std::is_same<U, V>::value, bool> operator==(Key<K, U> const&) const noexcept {
        return false;
    }

    template <typename U>
    constexpr bool operator!=(Key<K, U> const& other) const noexcept {
        return !(*this == other);
    }

    /** @} */

    /**
     * Define sort order for Keys.
     *
     * This must be expressed as `operator<` instead of std::less because only std::less<void> supports
     * arguments of mixed types, and it cannot be specialized.
     *
     * @param other the key, possibly of a different type, to compare to
     * @return equivalent to `this->getId() < other.getId()`
     *
     * @warning this comparison operator provides a strict weak ordering so long as `K` does, but is *not*
     * consistent with equality. In particular, keys with the same value of `getId()` but different types will
     * be equivalent but not equal.
     */
    template <typename U>
    constexpr bool operator<(Key<K, U> const& other) const noexcept {
        const std::less<K> comparator;
        return comparator(this->getId(), other.getId());
    }

    /// Return a hash of this object.
    std::size_t hash_value() const noexcept { return std::hash<K>()(id); }

private:
    /** The logical key. */
    K const id;
};

/**
 * Factory function for Key, to enable type parameter inference.
 *
 * @param id the key ID to create.
 *
 * @returns a key of the desired type
 *
 * @exceptsafe Provides the same exception safety as the copy-constructor of `K`.
 *
 * @relatesalso Key
 *
 * Calling this function prevents you from having to explicitly name the key type:
 *
 *     auto key = makeKey<int>("foo");
 */
// template parameters must be reversed for inference to work correctly
template <typename V, typename K>
constexpr Key<K, V> makeKey(K const& id) {
    return Key<K, V>(id);
}

/**
 * Output operator for Key.
 *
 * The output will use C++ template notation for the key; for example, a key "foo" pointing to an `int` may
 * print as `"foo<int>"`.
 *
 * @param os the desired output stream
 * @param key the key to print
 *
 * @returns a reference to `os`
 *
 * @exceptsafe Provides basic exception safety if the output operator of `K` is exception-safe.
 *
 * @warning the type name is compiler-specific and may be mangled or unintuitive; for example, some compilers
 * say "i" instead of "int"
 *
 * @relatesalso Key
 */
template <typename K, typename V>
std::ostream& operator<<(std::ostream& os, Key<K, V> const& key) {
    static const char* typeStr = typeid(V).name();
    static const char* constStr = std::is_const<V>::value ? " const" : "";
    static const char* volatileStr = std::is_volatile<V>::value ? " volatile" : "";
    os << key.getId() << "<" << typeStr << constStr << volatileStr << ">";
    return os;
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

namespace std {
template <typename K, typename V>
struct hash<typename lsst::afw::typehandling::Key<K, V>> {
    using argument_type = typename lsst::afw::typehandling::Key<K, V>;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const noexcept { return obj.hash_value(); }
};
}  // namespace std

#endif
