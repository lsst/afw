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

#ifndef LSST_AFW_TYPEHANDLING_PYTHON_H
#define LSST_AFW_TYPEHANDLING_PYTHON_H

#include "pybind11/pybind11.h"

#include "lsst/afw/typehandling/Storable.h"

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * "Trampoline" for Storable to let it be used as a base class in Python.
 *
 * Subclasses of Storable that are wrapped in %pybind11 should have a similar
 * helper that subclasses `StorableHelper<subclass>`. This helper can be
 * skipped if the subclass neither adds any virtual methods nor implements
 * any abstract methods.
 *
 * @tparam Base the exact (most specific) class being wrapped
 *
 * @see [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
 */
template <class Base = Storable>
class StorableHelper : public Base {
public:
    // Can't wrap clone(); PYBIND11_OVERLOAD chokes on unique_ptr return type

    std::string toString() const override {
        PYBIND11_OVERLOAD_NAME(std::string, Base, "__repr__", toString, );
    }

    std::size_t hash_value() const override {
        PYBIND11_OVERLOAD_NAME(std::size_t, Base, "__hash__", hash_value, );
    }

protected:
    bool equals(Base const& other) const noexcept override {
        PYBIND11_OVERLOAD_NAME(bool, Base, "__eq__", equals, other);
    }
};

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
