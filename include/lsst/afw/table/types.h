// -*- lsst-c++ -*-
#ifndef AFW_TABLE_types_h_INCLUDED
#define AFW_TABLE_types_h_INCLUDED

#include <cstdint>
#include <cstring>
#include <iostream>

#include "boost/mpl/vector.hpp"
#include "boost/preprocessor/punctuation/paren.hpp"
#include "Eigen/Core"

#include "lsst/base.h"
#include "lsst/pex/exceptions.h"
#include "ndarray.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#include "lsst/afw/coord.h"
#include "lsst/afw/table/misc.h"
#include "lsst/afw/table/KeyBase.h"

/*
 *  This file contains macros and MPL vectors that list the types that can be used for fields.
 *  The macros are used to do explicit instantiation in several source files.
 */

// Scalar types: those that can serve as elements for other types, and use the default FieldBase template.
#define AFW_TABLE_SCALAR_FIELD_TYPE_N 6
#define AFW_TABLE_SCALAR_FIELD_TYPES RecordId, std::uint16_t, std::uint32_t, std::int32_t, float, double, Angle
#define AFW_TABLE_SCALAR_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_SCALAR_FIELD_TYPES BOOST_PP_RPAREN()

// Arrays types: the types we allow for Array fields.
#define AFW_TABLE_ARRAY_FIELD_TYPE_N 4
#define AFW_TABLE_ARRAY_FIELD_TYPES std::uint16_t, std::uint32_t, int, float, double
#define AFW_TABLE_ARRAY_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_ARRAY_FIELD_TYPES BOOST_PP_RPAREN()

// Field types: all the types we allow for fields.
#define AFW_TABLE_FIELD_TYPE_N 12
#define AFW_TABLE_FIELD_TYPES                                                                        \
    AFW_TABLE_SCALAR_FIELD_TYPES, Flag, std::string, Array<std::uint16_t>, Array<std::uint32_t>, Array<int>,\
            Array<float>, Array<double>

#define AFW_TABLE_FIELD_TYPE_TUPLE BOOST_PP_LPAREN() AFW_TABLE_FIELD_TYPES BOOST_PP_RPAREN()

namespace lsst {
namespace afw {
namespace table {

/// An MPL vector of scalar field types.
typedef boost::mpl::vector<AFW_TABLE_SCALAR_FIELD_TYPES> ScalarFieldTypes;

/// An MPL vector of all field types.
typedef boost::mpl::vector<AFW_TABLE_FIELD_TYPES> FieldTypes;
}
}
}  // namespace lsst::afw::table

#endif  // !AFW_TABLE_types_h_INCLUDED
