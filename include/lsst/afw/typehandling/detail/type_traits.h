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

#ifndef LSST_AFW_TYPEHANDLING_DETAIL_TYPETRAITS_H
#define LSST_AFW_TYPEHANDLING_DETAIL_TYPETRAITS_H

#include <type_traits>

namespace lsst {
namespace afw {
namespace typehandling {
namespace detail {

// Test for smart pointers as "any type with an element_type member"
// Second template parameter is a dummy to let us do some metaprogramming
template <typename, typename = void>
constexpr bool IS_SMART_PTR = false;
template <typename T>
constexpr bool IS_SMART_PTR<T, std::enable_if_t<std::is_object<typename T::element_type>::value>> = true;

}  // namespace detail
}  // namespace typehandling
}  // namespace afw
}  // namespace lsst

#endif
