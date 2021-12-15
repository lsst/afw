// -*- lsst-c++ -*-
#ifndef AFW_TABLE_types_h_INCLUDED
#define AFW_TABLE_types_h_INCLUDED

#include <cstdint>
#include <cstring>
#include <iostream>

#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"
#include "lsst/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/KeyBase.h"

/*
 *  This file contains macros and MPL vectors that list the types that can be used for fields.
 *  The macros are used to do explicit instantiation in several source files.
 */

// Scalar types: those that can serve as elements for other types, and use the default FieldBase template.
#define AFW_TABLE_SCALAR_FIELD_TYPE_N 7
#define AFW_TABLE_SCALAR_FIELD_TYPES \
    RecordId, std::uint16_t, std::int32_t, float, double, lsst::geom::Angle, std::uint8_t
#define AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_SCALAR_FIELD_TYPES BOOST_PP_RPAREN()

// Arrays types: the types we allow for Array fields.
#define AFW_TABLE_ARRAY_FIELD_TYPE_N 5
#define AFW_TABLE_ARRAY_FIELD_TYPES std::uint16_t, int, float, double, std::uint8_t
#define AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_ARRAY_FIELD_TYPES BOOST_PP_RPAREN()

// Field types: all the types we allow for fields.
#define AFW_TABLE_FIELD_TYPE_N 14
#define AFW_TABLE_FIELD_TYPES                                                                        \
    AFW_TABLE_SCALAR_FIELD_TYPES, Flag, std::string, Array<std::uint16_t>, Array<int>, Array<float>, \
            Array<double>, Array<std::uint8_t>

#define AFW_TABLE_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_FIELD_TYPES BOOST_PP_RPAREN()

namespace lsst {
namespace afw {
namespace table {

/// A compile-time list of types.
template <typename ...E> struct TypeList;

/// Partial specialization of TypeList used for recursive implementation of members.
template <typename T, typename ...E> struct TypeList<T, E...> {

    /// A constexpr variable that evaluates to true if U is in the list (no
    /// checking for const, reference, mutable, etc.)
    template <typename U>
    static constexpr bool contains = std::is_same_v<T, U> || TypeList<E...>::template contains<U>;

    /// Invoke func on a null pointer cast to `T const *` for each type `T` in
    /// the list.
    template <typename F>
    static void for_each_nullptr(F func) {
        func(static_cast<T const*>(nullptr));
        TypeList<E...>::for_each_nullptr(func);
    }
};

/// Sentinal specialization of TypeList with only one type, used to end recursion.
template <typename T> struct TypeList<T> {
    template <typename U>
    static constexpr bool contains = std::is_same_v<T, U>;

    template <typename F>
    static void for_each_nullptr(F func) {
        func(static_cast<T const*>(nullptr));
    }
};


/// A compile-time list of all field types.
using FieldTypes = TypeList<AFW_TABLE_FIELD_TYPES>;

/// A compile-time list of all array field types.
using ArrayFieldTypes = TypeList<AFW_TABLE_ARRAY_FIELD_TYPES>;

/// A compile-time list of all scalar field types.
using ScalarFieldTypes = TypeList<AFW_TABLE_SCALAR_FIELD_TYPES>;

}  // namespace table
}  // namespace afw
}  // namespace lsst

#endif  // !AFW_TABLE_types_h_INCLUDED
