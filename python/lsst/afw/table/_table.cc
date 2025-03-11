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

#include "nanobind/nanobind.h"

#include "lsst/cpputils/python.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace lsst {
namespace afw {
namespace table {

using cpputils::python::WrapperCollection;

void wrapAggregates(WrapperCollection&);
void wrapAliasMap(WrapperCollection&);
void wrapArrays(WrapperCollection&);
void wrapBase(WrapperCollection&);
void wrapBaseColumnView(WrapperCollection&);
void wrapExposure(WrapperCollection&);
void wrapIdFactory(WrapperCollection&);
void wrapMatch(WrapperCollection&);
void wrapSchema(WrapperCollection&);
void wrapSchemaMapper(WrapperCollection&);
void wrapSimple(WrapperCollection&);
void wrapSlots(WrapperCollection&);
void wrapSource(WrapperCollection&);
void wrapWcsUtils(WrapperCollection&);

NB_MODULE(_table, mod) {
    WrapperCollection wrappers(mod, "lsst.afw.table");
    wrapAliasMap(wrappers);
    wrapSchema(wrappers);
    wrapSchemaMapper(wrappers);
    wrapBaseColumnView(wrappers);
    wrapBase(wrappers);
    wrapIdFactory(wrappers);
    wrapArrays(wrappers);
    wrapAggregates(wrappers);
    wrapSlots(wrappers);
    wrapSimple(wrappers);
    wrapSource(wrappers);
    wrapExposure(wrappers);
    wrapMatch(wrappers);
    wrapWcsUtils(wrappers);
    wrappers.finish();
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
