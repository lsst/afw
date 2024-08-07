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

#include "pybind11/pybind11.h"

#include "lsst/cpputils/python.h"
#include "lsst/afw/table/IdFactory.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {

using PyIdFactory = py::class_<IdFactory, std::shared_ptr<IdFactory>>;

void wrapIdFactory(cpputils::python::WrapperCollection& wrappers) {
    wrappers.wrapType(PyIdFactory(wrappers.module, "IdFactory"), [](auto& mod, auto& cls) {
        cls.def("__call__", &IdFactory::operator());
        cls.def("notify", &IdFactory::notify, "id"_a);
        cls.def("clone", &IdFactory::clone);
        cls.def_static("makeSimple", IdFactory::makeSimple);
        cls.def_static("makeSource", IdFactory::makeSource, "expId"_a, "reserved"_a);
        cls.def_static("computeReservedFromMaxBits", IdFactory::computeReservedFromMaxBits, "maxBits"_a);
    });
}

}  // namespace table
}  // namespace afw
}  // namespace lsst
