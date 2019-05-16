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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ostream>
#include <memory>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "boost/core/demangle.hpp"
#include "boost/variant.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/typehandling/PolymorphicValue.h"

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
    static const std::string typeStr = boost::core::demangle(typeid(V).name());
    static const std::string constStr = std::is_const<V>::value ? " const" : "";
    static const std::string volatileStr = std::is_volatile<V>::value ? " volatile" : "";
    os << key.getId() << "<" << typeStr << constStr << volatileStr << ">";
    return os;
}

// Test for smart pointers as "any type with an element_type member"
// Second template parameter is a dummy to let us do some metaprogramming
template <typename, typename = void>
constexpr bool IS_SMART_PTR = false;
template <typename T>
constexpr bool IS_SMART_PTR<T, std::enable_if_t<std::is_object<typename T::element_type>::value>> = true;

/**
 * Interface for a heterogeneous map.
 *
 * Objects of type GenericMap cannot necessarily have keys added or removed, although mutable values can be
 * modified as usual. In Python, a GenericMap behaves like a ``collections.abc.Mapping``. See
 * MutableGenericMap for a GenericMap that must allow insertions and deletions.
 *
 * @tparam K the key type of the map.
 *
 * A Key for the map is parameterized by both the key type `K` and a corresponding value type `V`. The map
 * is indexed uniquely by a value of type `K`; no two entries in the map may have identical values of
 * Key::getId().
 *
 * All operations are sensitive to the value type of the key: a @ref contains(Key<K,T> const&) const
 * "contains" call requesting an integer labeled "value", for example, will report no such integer if instead
 * there is a string labeled "value". For Python compatibility, a GenericMap does not store type information
 * internally, instead relying on RTTI for type checking.
 *
 * All subclasses **must** guarantee, as a class invariant, that every value in the map is implicitly
 * nothrow-convertible to the type indicated by its key. For example, MutableGenericMap ensures this by
 * appropriately templating all operations that create new key-value pairs.
 *
 * A GenericMap may contain primitive types, strings, Storable, and shared pointers to Storable as
 * values. It does not support unique pointers to Storable because such pointers are read destructively. For
 * safety reasons, it may not contain references, C-style pointers, or arrays to any type. Due to
 * implementation restrictions, `const` types (particularly pointers to `const` Storable) are not
 * currently supported.
 */
// TODO: const keys should be possible in C++17 with std::variant
template <typename K>
class GenericMap {
public:
    using key_type = K;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    virtual ~GenericMap() noexcept = default;

    /**
     * Return a reference to the mapped value of the element with key equal to `key`.
     *
     * @tparam T the type of the element mapped to `key`
     * @param key the key of the element to find
     *
     * @return a reference to the `T` mapped to `key`, if one exists
     *
     * @throws pex::exceptions::OutOfRangeError Thrown if the map does not
     *         have a `T` with the specified key
     * @exceptsafe Provides strong exception safety.
     *
     * @note This implementation calls @ref unsafeLookup once, then uses templates
     *       and RTTI to determine if the value is of the expected type.
     *
     * @{
     */
    template <typename T, typename std::enable_if_t<!IS_SMART_PTR<T>, int> = 0>
    T& at(Key<K, T> const& key) {
        // Both casts are safe; see Effective C++, Item 3
        return const_cast<T&>(static_cast<const GenericMap&>(*this).at(key));
    }

    // Can't partially specialize method templates, rely on enable_if to avoid duplicates
    template <typename T,
              typename std::enable_if_t<
                      std::is_fundamental<T>::value || std::is_base_of<std::string, T>::value, int> = 0>
    T const& at(Key<K, T> const& key) const {
        static_assert(!std::is_const<T>::value,
                      "Due to implementation constraints, const keys are not supported.");
        try {
            auto foo = unsafeLookup(key.getId());
            return boost::get<T const&>(foo);
        } catch (boost::bad_get const&) {
            std::stringstream message;
            message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
            throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
        }
    }

    template <typename T, typename std::enable_if_t<std::is_base_of<Storable, T>::value, int> = 0>
    T const& at(Key<K, T> const& key) const {
        static_assert(!std::is_const<T>::value,
                      "Due to implementation constraints, const keys are not supported.");
        try {
            auto foo = unsafeLookup(key.getId());
            // Don't use pointer-based get, because it won't work after migrating to std::variant
            Storable const& value = boost::get<PolymorphicValue const&>(foo);
            T const* typedPointer = dynamic_cast<T const*>(&value);
            if (typedPointer != nullptr) {
                return *typedPointer;
            } else {
                std::stringstream message;
                message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
                throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
            }
        } catch (boost::bad_get const&) {
            std::stringstream message;
            message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
            throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
        }
    }

    template <typename T, typename std::enable_if_t<std::is_base_of<Storable, T>::value, int> = 0>
    std::shared_ptr<T> at(Key<K, std::shared_ptr<T>> const& key) const {
        static_assert(!std::is_const<T>::value,
                      "Due to implementation constraints, const keys are not supported.");
        try {
            auto foo = unsafeLookup(key.getId());
            auto pointer = boost::get<std::shared_ptr<Storable> const&>(foo);
            std::shared_ptr<T> typedPointer = std::dynamic_pointer_cast<T>(pointer);
            // shared_ptr can be empty without being null. dynamic_pointer_cast
            // only promises result of failed cast is empty, so test for that
            if (typedPointer.use_count() > 0) {
                return typedPointer;
            } else {
                std::stringstream message;
                message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
                throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
            }
        } catch (boost::bad_get const&) {
            std::stringstream message;
            message << "Key " << key << " not found, but a key labeled " << key.getId() << " is present.";
            throw LSST_EXCEPT(pex::exceptions::OutOfRangeError, message.str());
        }
    }

    /** @} */

    /// Return the number of key-value pairs in the map.
    virtual size_type size() const noexcept = 0;

    /// Return `true` if this map contains no key-value pairs.
    virtual bool empty() const noexcept = 0;

    /**
     * Return the maximum number of elements the container is able to hold due to system or library
     * implementation limitations.
     *
     * @note This value typically reflects the theoretical limit on the size of the container. At runtime, the
     * size of the container may be limited to a value smaller than max_size() by the amount of RAM available.
     */
    virtual size_type max_size() const noexcept = 0;

    /**
     * Return the number of elements mapped to the specified key.
     *
     * @tparam T the value corresponding to `key`
     * @param key key value of the elements to count
     *
     * @return number of `T` with key `key`, that is, either 1 or 0.
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @note This implementation calls @ref contains(Key<K,T> const&) const "contains".
     *
     */
    template <typename T>
    size_type count(Key<K, T> const& key) const {
        return contains(key) ? 1 : 0;
    }

    /**
     * Return `true` if this map contains a mapping whose key has the specified label.
     *
     * More formally, this method returns `true` if and only if this map contains a mapping with a key `k`
     * such that `k.getId() == key`. There can be at most one such mapping.
     *
     * @param key the weakly-typed key to search for
     *
     * @return `true` if this map contains a mapping for `key`, regardless of value type.
     *
     * @exceptsafe Provides strong exception safety.
     */
    virtual bool contains(K const& key) const = 0;

    /**
     * Return `true` if this map contains a mapping for the specified key.
     *
     * This is equivalent to testing whether `at(key)` would succeed.
     *
     * @tparam T the value corresponding to `key`
     * @param key the key to search for
     *
     * @return `true` if this map contains a mapping from the specified key to a `T`
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @note This implementation calls contains(K const&) const. If the call returns
     *       `true`, it calls @ref unsafeLookup, then uses templates and RTTI to
     *       determine if the value is of the expected type. The performance of
     *       this method depends strongly on the performance of
     *       contains(K const&).
     *
     * @{
     */
    // Can't partially specialize method templates, rely on enable_if to avoid duplicates
    template <typename T,
              typename std::enable_if_t<
                      std::is_fundamental<T>::value || std::is_base_of<std::string, T>::value, int> = 0>
    bool contains(Key<K, T> const& key) const {
        // Avoid actually getting and casting an object, if at all possible
        if (!contains(key.getId())) {
            return false;
        }

        auto foo = unsafeLookup(key.getId());
        // boost::variant has no equivalent to std::holds_alternative
        try {
            boost::get<T const&>(foo);
            return true;
        } catch (boost::bad_get const&) {
            return false;
        }
    }

    template <typename T, typename std::enable_if_t<std::is_base_of<Storable, T>::value, int> = 0>
    bool contains(Key<K, T> const& key) const {
        // Avoid actually getting and casting an object, if at all possible
        if (!contains(key.getId())) {
            return false;
        }

        auto foo = unsafeLookup(key.getId());
        try {
            // Don't use pointer-based get, because it won't work after migrating to std::variant
            Storable const& value = boost::get<PolymorphicValue const&>(foo);
            auto asT = dynamic_cast<T const*>(&value);
            return asT != nullptr;
        } catch (boost::bad_get const&) {
            return false;
        }
    }

    template <typename T, typename std::enable_if_t<std::is_base_of<Storable, T>::value, int> = 0>
    bool contains(Key<K, std::shared_ptr<T>> const& key) const {
        // Avoid actually getting and casting an object, if at all possible
        if (!contains(key.getId())) {
            return false;
        }

        auto foo = unsafeLookup(key.getId());
        try {
            auto pointer = boost::get<std::shared_ptr<Storable> const&>(foo);
            std::shared_ptr<T> typedPointer = std::dynamic_pointer_cast<T>(pointer);
            // shared_ptr can be empty without being null. dynamic_pointer_cast
            // only promises result of failed cast is empty, so test for that
            return typedPointer.use_count() > 0;
        } catch (boost::bad_get const&) {
            return false;
        }
    }

    /** @} */

    /**
     * Return the set of all keys, without type information.
     *
     * @return a copy of all keys currently in the map, in the same iteration order as this object. The set
     * will *not* be updated as this object changes, or vice versa.
     *
     * @note The keys are returned as a list, rather than a set, so that subclasses can give them a
     * well-defined iteration order.
     *
     * @exceptsafe Provides strong exception safety.
     */
    virtual std::vector<K> keys() const = 0;

    /**
     * Test for map equality.
     *
     * Two GenericMap objects are considered equal if they map the same keys to
     * the same values. The two objects do not need to have the same
     * implementation class. If either class orders its keys, the order
     * is ignored.
     *
     * @note This implementation calls @ref keys on both objects and compares
     *       the results. If the two objects have the same keys, it calls
     *       @ref unsafeLookup for each key and compares the values.
     *
     * @{
     */
    virtual bool operator==(GenericMap const& other) const {
        auto keys1 = this->keys();
        auto keys2 = other.keys();
        if (!std::is_permutation(keys1.begin(), keys1.end(), keys2.begin(), keys2.end())) {
            return false;
        }
        for (K const& key : keys1) {
            if (this->unsafeLookup(key) != other.unsafeLookup(key)) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(GenericMap const& other) const { return !(*this == other); }

    /** @} */

    /**
     * Apply an operation to each key-value pair in the map.
     *
     * @tparam Visitor a callable that takes a key and a value. See below for
     *                 exact requirements.
     * @param visitor the visitor to apply
     * @returns if `Visitor` has a return value, a `std::vector` of values
     *          returned from applying `visitor` to each key in @ref keys, in
     *          that order. Otherwise, `void`.
     *
     * @exceptsafe Provides the same level of exception safety as `Visitor`, or
     *             strong exception safety if `Visitor` cannot throw.
     *
     * A `Visitor` must define one or more `operator()` that take a
     * weakly-typed key and a value. Each `operator()` must return the same
     * type (which may be `void`). Through any combination of overloading or
     * templates, the visitor must accept values of the following types:
     *      * either `bool` or `bool const&`
     *      * either `std::int32_t` or `std::int32_t const&`
     *      * either `std::int64_t` or `std::int64_t const&`
     *      * either `float` or `float const&`
     *      * either `double` or `double const&`
     *      * `std::string const&`
     *      * `Storable const&`
     *      * `std::shared_ptr<Storable>`
     *
     * @note This implementation calls @ref keys, then calls @ref unsafeLookup
     *       for each key before passing the result to `visitor`.
     *
     * An example visitor that prints each key-value pair to standard output:
     *
     *     template <typename K>
     *     class Printer {
     *     public:
     *         template <typename V>
     *         void operator()(K const& key, V const& value) {
     *             std::cout << key << ": " << value << "," << std::endl;
     *         }
     *
     *         void operator()(K const& key, Storable const& value) {
     *             std::cout << key << ": ";
     *             try {
     *                 std::cout << value;
     *             } catch (UnsupportedOperationException const&) {
     *                 std::cout << "[unprintable]";
     *             }
     *             std::cout << "," << std::endl;
     *         }
     *
     *         void operator()(K const& key, std::shared_ptr<Storable> value) {
     *             if (value != nullptr) {
     *                 operator()(key, *value);
     *             } else {
     *                 operator()(key, "null");
     *             }
     *         }
     *     };
     */
    template <class Visitor>
    auto apply(Visitor&& visitor) const {
        // Delegate to private methods to hide special-casing of Visitor
        return _apply(visitor);
    }

    /**
     * Apply a modifying operation to each key-value pair in the map.
     *
     * @tparam Visitor a callable that takes a key and a value. Requirements as for
     *                 @ref apply(Visitor&&) const, except that it may take
     *                 non-`const` references to values.
     * @param visitor the visitor to apply
     * @returns if `Visitor` has a return value, a `std::vector` of values
     *          returned from applying `visitor` to each key in @ref keys, in
     *          that order. Otherwise, `void`.
     *
     * @exceptsafe Provides basic exception safety if `Visitor` is exception-safe.
     *
     * @note This implementation calls @ref keys, then calls @ref unsafeLookup
     *       for each key before passing the result to `visitor`.
     */
    template <class Visitor>
    auto apply(Visitor&& visitor) {
        // Delegate to private methods to hide special-casing of Visitor
        return _apply(visitor);
    }

private:
    // Icky TMP, but I can't find another way to get at the template arguments for variant :(
    // Methods have no definition but can't be deleted without breaking definition of StorableType
    /// @cond
    template <typename T>
    using _RemoveConstFromRef = std::add_lvalue_reference_t<std::remove_const_t<std::remove_reference_t<T>>>;
    template <typename... Types>
    static boost::variant<std::decay_t<Types>...> _referenceToType(boost::variant<Types...> const&) noexcept;
    template <typename... Types>
    static boost::variant<_RemoveConstFromRef<Types>...> _constRefToRef(
            boost::variant<Types...> const&) noexcept;
    /// @endcond

protected:
    /**
     * A type-agnostic reference to the value stored inside the map.
     *
     * Keys of any subclass of Storable are implemented using PolymorphicValue to preserve type.
     *
     * @{
     */
    // may need to use std::reference_wrapper when migrating to std::variant, but it confuses Boost
    using ConstValueReference =
            boost::variant<bool const&, std::int32_t const&, std::int64_t const&, float const&, double const&,
                           std::string const&, PolymorphicValue const&, std::shared_ptr<Storable> const&>;
    using ValueReference = decltype(_constRefToRef(std::declval<ConstValueReference>()));

    /** @} */

    /**
     * The types that can be stored in a map.
     *
     * These are the pass-by-value equivalents (using std::decay) of @ref ConstValueReference.
     */
    // this mouthful is shorter than the equivalent expression with result_of
    using StorableType = decltype(_referenceToType(std::declval<ConstValueReference>()));

    /**
     * Return a reference to the mapped value of the element with key equal to `key`.
     *
     * This method is the primary way to implement the GenericMap interface.
     *
     * @param key the key of the element to find
     *
     * @return the value mapped to `key`, if one exists
     *
     * @throws pex::exceptions::OutOfRangeError Thrown if the map does not have
     *         a value with the specified key
     * @exceptsafe Must provide strong exception safety.
     *
     * @{
     */
    virtual ConstValueReference unsafeLookup(K key) const = 0;

    ValueReference unsafeLookup(K key) {
        ConstValueReference constRef = static_cast<const GenericMap&>(*this).unsafeLookup(key);
        auto removeConst = [](auto const& value) -> ValueReference {
            // This cast is safe; see Effective C++, Item 3
            return const_cast<_RemoveConstFromRef<decltype(value)>>(value);
        };
        return boost::apply_visitor(removeConst, constRef);
    }

    /** @} */

private:
    // Type alias to properly handle Visitor output
    // Assume that each operator() has the same return type; variant will enforce it
    /// @cond
    template <class Visitor>
    using _VisitorResult = std::result_of_t<Visitor && (K&&, bool&)>;
    /// @endcond

    // No return value, const GenericMap
    template <class Visitor, typename std::enable_if_t<std::is_void<_VisitorResult<Visitor>>::value, int> = 0>
    void _apply(Visitor&& visitor) const {
        for (K const& key : keys()) {
            boost::variant<K> varKey = key;
            boost::apply_visitor(visitor, varKey, unsafeLookup(key));
        }
    }

    // Return value, const GenericMap
    template <class Visitor,
              typename std::enable_if_t<!std::is_void<_VisitorResult<Visitor>>::value, int> = 0>
    auto _apply(Visitor&& visitor) const {
        std::vector<_VisitorResult<Visitor>> results;
        results.reserve(size());

        for (K const& key : keys()) {
            boost::variant<K> varKey = key;
            results.emplace_back(boost::apply_visitor(visitor, varKey, unsafeLookup(key)));
        }
        return results;
    }

    // No return value, non-const GenericMap
    template <class Visitor, typename std::enable_if_t<std::is_void<_VisitorResult<Visitor>>::value, int> = 0>
    void _apply(Visitor&& visitor) {
        for (K const& key : keys()) {
            boost::variant<K> varKey = key;
            // Boost gets confused if we pass it a temporary variant
            ValueReference ref = unsafeLookup(key);
            boost::apply_visitor(visitor, varKey, ref);
        }
    }

    // Return value, non-const GenericMap
    template <class Visitor,
              typename std::enable_if_t<!std::is_void<_VisitorResult<Visitor>>::value, int> = 0>
    auto _apply(Visitor&& visitor) {
        std::vector<_VisitorResult<Visitor>> results;
        results.reserve(size());

        for (K const& key : keys()) {
            boost::variant<K> varKey = key;
            // Boost gets confused if we pass it a temporary variant
            ValueReference ref = unsafeLookup(key);
            results.emplace_back(boost::apply_visitor(visitor, varKey, ref));
        }
        return results;
    }
};

/**
 * Interface for a GenericMap that allows element addition and removal.
 *
 * In Python, a MutableGenericMap behaves like a ``collections.abc.MutableMapping``.
 *
 * @note Unlike standard library maps, this class does not support `operator[]` or `insert_or_assign`. This is
 * because these operations would have surprising behavior when dealing with keys of different types but the
 * same Key::getId().
 *
 */
template <typename K>
class MutableGenericMap : public GenericMap<K> {
protected:
    using typename GenericMap<K>::StorableType;

public:
    virtual ~MutableGenericMap() noexcept = default;

    /**
     * Remove all of the mappings from this map.
     *
     * After this call, the map will be empty.
     */
    virtual void clear() noexcept = 0;

    /**
     * Insert an element into the map, if the map doesn't already contain a mapping with the same or a
     * conflicting key.
     *
     * @tparam T the type of value to insert
     * @param key key to insert
     * @param value value to insert
     *
     * @return `true` if the insertion took place, `false` otherwise
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @note It is possible for a key with a value type other than `T` to prevent insertion. Callers can
     * safely assume `this->contains(key.getId())` as a postcondition, but not `this->contains(key)`.
     *
     * @note This implementation calls @ref contains(K const&) const "contains",
     *       then calls @ref unsafeInsert if there is no conflicting key.
     */
    template <typename T>
    bool insert(Key<K, T> const& key, T const& value) {
        if (this->contains(key.getId())) {
            return false;
        }

        return unsafeInsert(key.getId(), StorableType(value));
    }

    /**
     * Insert an element into the map, if the map doesn't already contain a mapping with a conflicting key.
     *
     * @tparam T the type of value to insert
     * @param key key to insert
     * @param value value to insert
     *
     * @return a pair consisting of a strongly-typed key for the value and a flag that is `true` if the
     * insertion took place and `false` otherwise
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @warning the type of the compiler-generated key may not always be what you expect. Callers should save
     * the returned key if they wish to retrieve the value later.
     */
    template <typename T>
    std::pair<Key<K, T>, bool> insert(K const& key, T const& value) {
        auto strongKey = makeKey<T>(key);
        // Construct return value in advance, so that exception from copying/moving Key is atomic
        auto result = make_pair(strongKey, false);
        result.second = insert(strongKey, value);
        return result;
    }

    /**
     * Remove the mapping for a key from this map, if it exists.
     *
     * @tparam T the type of value the key maps to
     * @param key the key to remove
     *
     * @return `true` if `key` was removed, `false` if it was not present
     *
     * @exceptsafe Provides strong exception safety.
     *
     * @note This implementation calls @ref contains(Key<K,T> const&) const "contains",
     *       then calls @ref unsafeErase if the key is present.
     */
    template <typename T>
    bool erase(Key<K, T> const& key) {
        if (this->contains(key)) {
            return unsafeErase(key.getId());
        } else {
            return false;
        }
    }

protected:
    /**
     * Create a new mapping with key equal to `key` and value equal to `value`.
     *
     * This method is the primary way to implement the MutableGenericMap interface.
     *
     * @param key the key of the element to insert. The method may assume that the map does not contain `key`.
     * @param value a reference to the value to insert.
     *
     * @return `true` if the insertion took place, `false` otherwise
     *
     * @exceptsafe Must provide strong exception safety.
     */
    virtual bool unsafeInsert(K key, StorableType&& value) = 0;

    /**
     * Remove the mapping for a key from this map, if it exists.
     *
     * @param key the key to remove
     *
     * @return `true` if `key` was removed, `false` if it was not present
     *
     * @exceptsafe Must provide strong exception safety.
     */
    virtual bool unsafeErase(K key) = 0;
};

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
