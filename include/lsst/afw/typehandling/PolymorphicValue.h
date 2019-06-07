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

#ifndef LSST_AFW_TYPEHANDLING_POLYMORPHICVALUE_H
#define LSST_AFW_TYPEHANDLING_POLYMORPHICVALUE_H

#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * Container that passes Storable objects by value while preserving type.
 *
 * This class is implicitly convertible to and from a reference to Storable,
 * but behaves like a value: changing the internal Storable changes the
 * object's state, and copying the object creates a new Storable.
 *
 * @note While a PolymorphicValue is always initialized with a Storable, it
 * may become empty if it is the source of a move-construction or
 * move-assignment. Conversion of an empty value to Storable& throws.
 */
/*
 * Note: I would like to disallow the empty state, as it serves no purpose,
 * but I can't think of any other sensible post-move state.
 */
class PolymorphicValue final {
public:
    /**
     * Create a new object containing a copy of a Storable.
     *
     * @param value the value to copy into a PolymorphicValue
     */
    PolymorphicValue(Storable const& value);
    ~PolymorphicValue() noexcept;

    /**
     * Try to copy a PolymorphicValue.
     *
     * @param other the PolymorphicValue to copy.
     *
     * @throws UnsupportedOperationException Thrown if a copy is required and
     *      the object in `other` does not implement Storable::cloneStorable.
     *
     * @{
     */
    PolymorphicValue(PolymorphicValue const& other);
    PolymorphicValue(PolymorphicValue&& other) noexcept;

    /** @} */

    /**
     * Try to assign a PolymorphicValue.
     *
     * To preserve the run-time type of the object in `other`, this method
     * swaps (and possibly copies) the Storables instead of relying on the
     * `Storable`'s `operator=`.
     *
     * @param other the PolymorphicValue to overwrite this value with.
     *
     * @throws UnsupportedOperationException Thrown if a copy is required and
     *      the object in `other` does not implement Storable::cloneStorable.
     *
     * @{
     */
    PolymorphicValue& operator=(PolymorphicValue const& other);
    PolymorphicValue& operator=(PolymorphicValue&& other) noexcept;

    /// Exchange the contents of this container and another.
    void swap(PolymorphicValue& other) noexcept;

    /** @} */

    /**
     * Check whether this object contains a Storable.
     *
     * @return `true` if this object has no Storable, `false` otherwise
     */
    bool empty() const noexcept;

    /**
     * Return a reference to the internal Storable, if one exists.
     *
     * @returns a reference to the internal object
     * @throws pex::exceptions::LogicError Thrown if this object is empty.
     *
     * @{
     */
    operator Storable&();
    operator Storable const&() const;
    Storable& get();
    Storable const& get() const;

    /** @} */

    /**
     * Test whether the contained Storables are equal.
     *
     * Empty PolymorphicValues compare equal to each other and unequal to any
     * non-empty PolymorphicValue.
     *
     * @{
     */
    bool operator==(PolymorphicValue const& other) const noexcept;
    bool operator!=(PolymorphicValue const& other) const noexcept { return !(*this == other); }

    /** @} */

    /**
     * Return a hash of this object (optional operation).
     *
     * @throws UnsupportedOperationException Thrown if the internal Storable
     *      is not hashable.
     */
    std::size_t hash_value() const;

private:
    // unique_ptr would be more appropriate, but Storable::cloneStorable must return shared_ptr
    std::shared_ptr<Storable> _value;
};

/**
 * Swap specialization for PolymorphicValue.
 *
 * @relatesalso PolymorphicValue
 */
inline void swap(PolymorphicValue& lhs, PolymorphicValue& rhs) noexcept { lhs.swap(rhs); }

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

namespace std {
/**
 * Hash specialization for PolymorphicValue.
 *
 * @returns the hash of the Storable inside the PolymorphicValue, or an
 *          arbitrary value if it is empty
 * @throws UnsupportedOperationException Thrown if the Storable is not hashable.
 */
template <>
struct hash<lsst::afw::typehandling::PolymorphicValue> {
    using argument_type = lsst::afw::typehandling::PolymorphicValue;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const { return obj.hash_value(); }
};

/// Swap specialization for PolymorphicValue.
template <>
inline void swap(lsst::afw::typehandling::PolymorphicValue& lhs,
                 lsst::afw::typehandling::PolymorphicValue& rhs) noexcept {
    lhs.swap(rhs);
}

}  // namespace std

#endif
