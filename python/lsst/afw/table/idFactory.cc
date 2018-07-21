/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
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
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include "pybind11/pybind11.h"

#include "lsst/afw/table/IdFactory.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

using PyIdFactory = py::class_<IdFactory, std::shared_ptr<IdFactory>>;

PYBIND11_MODULE(idFactory, mod) {
    PyIdFactory cls(mod, "IdFactory");
    cls.def("__call__", &IdFactory::operator());
    cls.def("notify", &IdFactory::notify, "id"_a);
    cls.def("clone", &IdFactory::clone);
    cls.def_static("makeSimple", IdFactory::makeSimple);
    cls.def_static("makeSource", IdFactory::makeSource, "expId"_a, "reserved"_a);
}
}
}
}
}  // namespace lsst::afw::table::<anonymous>
