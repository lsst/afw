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

#ifndef LSST_AFW_TYPEHANDLING_STORABLE_H
#define LSST_AFW_TYPEHANDLING_STORABLE_H

#include <functional>
#include <memory>
#include <ostream>
#include <string>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace typehandling {

/**
 * Exception thrown by Storable operations for unimplemented operations.
 *
 * As with all RuntimeError, callers should assume that this exception may be thrown at any time.
 */
LSST_EXCEPTION_TYPE(UnsupportedOperationException, pex::exceptions::RuntimeError,
                    lsst::afw::typehandling::UnsupportedOperationException)

/**
 * Interface supporting iteration over heterogenous containers.
 *
 * Storable may be subclassed by either C++ or Python classes. Many operations defined by Storable are
 * optional, and may throw UnsupportedOperationException if they are not defined.
 *
 * @see StorableHelper
 */
class Storable : public table::io::Persistable {
public:
    virtual ~Storable() noexcept = 0;

    /**
     * Create a new object that is a copy of this one (optional operation).
     *
     * This operation is required for Storables that are stored in GenericMap
     * by value, but not for those stored by shared pointer.
     *
     * @throws UnsupportedOperationException Thrown if this object is not cloneable.
     *
     * @note If this class supports a `clone` operation, the two should behave
     *       identically except for the formal return type.
     */
    // Return shared_ptr to work around https://github.com/pybind/pybind11/issues/1138
    virtual std::shared_ptr<Storable> cloneStorable() const;

    /**
     * Create a string representation of this object (optional operation).
     *
     * @throws UnsupportedOperationException Thrown if this object does not have a string representation.
     */
    virtual std::string toString() const;

    /**
     * Return a hash of this object (optional operation).
     *
     * @throws UnsupportedOperationException Thrown if this object is not hashable.
     *
     * @note Subclass authors are responsible for any associated specializations of std::hash.
     */
    virtual std::size_t hash_value() const;

    /**
     * Compare this object to another Storable.
     *
     * Subclasses that implement equality comparison must override this
     * method to give results consistent with `operator==` for all inputs that
     * are accepted by both.
     *
     * @returns This implementation returns whether the two objects are the same.
     *
     * @warning This method compares an object to any type of Storable,
     * although cross-class comparisons should usually return `false`. If
     * cross-class comparisons are valid, implementers should take care that
     * they are symmetric and will give the same result no matter what the
     * compile-time types of the left- and right-hand sides are.
     */
    virtual bool equals(Storable const& other) const noexcept;
};

/**
 * Output operator for Storable.
 *
 * @param os the desired output stream
 * @param storable the object to print
 *
 * @returns a reference to `os`
 *
 * @throws UnsupportedOperationException Thrown if `storable` does not have an implementation of
 *      Storable::toString.
 *
 * @relatesalso Storable
 */
inline std::ostream& operator<<(std::ostream& os, Storable const& storable) {
    return os << storable.toString();
}

}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

namespace std {
/**
 * Generic hash to allow polymorphic access to Storable
 *
 * @throws UnsupportedOperationException Thrown if the argument is not hashable.
 */
template <>
struct hash<lsst::afw::typehandling::Storable> {
    using argument_type = lsst::afw::typehandling::Storable;
    using result_type = size_t;
    size_t operator()(argument_type const& obj) const { return obj.hash_value(); }
};
}  // namespace std

#endif
