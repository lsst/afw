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

#include "lsst/pex/exceptions.h"

#include "lsst/afw/typehandling/PolymorphicValue.h"

namespace lsst {
namespace afw {
namespace typehandling {

PolymorphicValue::PolymorphicValue(Storable const& value) : _value(value.clone()) {}
// No move-constructor, because putting a pointer to Storable&& into a
// unique_ptr is safe only if the object was dynamically allocated

PolymorphicValue::~PolymorphicValue() noexcept = default;

PolymorphicValue::PolymorphicValue(PolymorphicValue const& other)
        : _value(other._value ? other._value->clone() : std::unique_ptr<Storable>()) {}
PolymorphicValue::PolymorphicValue(PolymorphicValue&&) = default;  // other._value emptied

PolymorphicValue& PolymorphicValue::operator=(PolymorphicValue const& other) {
    if (other._value) {
        _value = other._value->clone();
    } else {
        _value.reset();
    }
    return *this;
}
PolymorphicValue& PolymorphicValue::operator=(PolymorphicValue&& other) {
    using std::swap;
    swap(_value, other._value);
    return *this;
}

bool PolymorphicValue::empty() const noexcept { return !_value; }

PolymorphicValue::operator Storable&() { return get(); }
PolymorphicValue::operator Storable const&() const { return get(); }
Storable& PolymorphicValue::get() {
    // Both casts are safe; see Effective C++, Item 3
    return const_cast<Storable&>(static_cast<PolymorphicValue const&>(*this).get());
}
Storable const& PolymorphicValue::get() const {
    if (_value) {
        return *_value;
    } else {
        throw LSST_EXCEPT(pex::exceptions::LogicError, "Dereferenced empty PolymorphicValue.");
    }
}

bool PolymorphicValue::operator==(PolymorphicValue const& other) const noexcept {
    if (!empty() && !other.empty()) {
        return get().equals(other.get());
    } else {
        // At least one pointer is null; pointer comparison has desired behavior
        return _value == other._value;
    }
}

std::size_t PolymorphicValue::hash_value() const { return std::hash<Storable>()(get()); }

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst
