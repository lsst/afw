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
#include "pybind11/stl.h"

#include "ndarray/pybind11.h"

#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/BaseColumnView.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace {

using PyBaseColumnView = py::class_<BaseColumnView, std::shared_ptr<BaseColumnView>>;

using PyBitsColumn = py::class_<BitsColumn, std::shared_ptr<BitsColumn>>;

template <typename T, typename PyClass>
static void declareBaseColumnViewOverloads(PyClass &cls) {
    cls.def("_basicget", [](BaseColumnView & self, Key<T> const &key) ->
            typename ndarray::Array<T, 1> const { return self[key]; });
};

template <typename U, typename PyClass>
static void declareBaseColumnViewArrayOverloads(PyClass &cls) {
    cls.def("_basicget", [](BaseColumnView & self, Key<lsst::afw::table::Array<U>> const &key) ->
            typename ndarray::Array<U, 2, 1> const { return self[key]; });
};

template <typename PyClass>
static void declareBaseColumnViewFlagOverloads(PyClass &cls) {
    cls.def("_basicget",
            [](BaseColumnView &self, Key<Flag> const &key) -> ndarray::Array<bool const, 1, 1> const {
                return ndarray::copy(self[key]);
            });
};

static void declareBaseColumnView(py::module &mod) {
    // We can't call this "BaseColumnView" because that's the typedef for "ColumnViewT<BaseRecord>".
    // This is just a mostly-invisible implementation base class, so we use the same naming convention
    // we use for those.
    PyBaseColumnView cls(mod, "_BaseColumnViewBase");
    cls.def("getTable", &BaseColumnView::getTable);
    cls.def_property_readonly("table", &BaseColumnView::getTable);
    cls.def("getSchema", &BaseColumnView::getSchema);
    cls.def_property_readonly("schema", &BaseColumnView::getSchema);
    // _getBits supports a Python version of getBits that accepts None and field names as keys
    cls.def("_getBits", &BaseColumnView::getBits);
    cls.def("getAllBits", &BaseColumnView::getAllBits);
    declareBaseColumnViewOverloads<std::uint8_t>(cls);
    declareBaseColumnViewOverloads<std::uint16_t>(cls);
    declareBaseColumnViewOverloads<std::int32_t>(cls);
    declareBaseColumnViewOverloads<std::int64_t>(cls);
    declareBaseColumnViewOverloads<float>(cls);
    declareBaseColumnViewOverloads<double>(cls);
    declareBaseColumnViewFlagOverloads(cls);
    // std::string columns are not supported, because numpy string arrays
    // do not have the same memory model as ours.
    declareBaseColumnViewArrayOverloads<std::uint8_t>(cls);
    declareBaseColumnViewArrayOverloads<std::uint16_t>(cls);
    declareBaseColumnViewArrayOverloads<int>(cls);
    declareBaseColumnViewArrayOverloads<float>(cls);
    declareBaseColumnViewArrayOverloads<double>(cls);
    // Angle requires custom wrappers, because ndarray doesn't
    // recognize it natively; we just return a double view
    // (e.g. radians).
    using AngleArray = ndarray::Array<Angle, 1>;
    using DoubleArray = ndarray::Array<double, 1>;
    cls.def(
        "_basicget",
        [](BaseColumnView & self, Key<Angle> const &key) -> DoubleArray {
            ndarray::Array<Angle,1,0> a = self[key];
            return ndarray::detail::ArrayAccess<DoubleArray>::construct(
                reinterpret_cast<double*>(a.getData()),
                ndarray::detail::ArrayAccess<AngleArray>::getCore(a)
            );
        }
    );
}

static void declareBitsColumn(py::module &mod) {
    PyBitsColumn cls(mod, "BitsColumn");
    cls.def("getArray", &BitsColumn::getArray);
    cls.def_property_readonly("array", &BitsColumn::getArray);
    cls.def("getBit", (BitsColumn::IntT (BitsColumn::*)(Key<Flag> const &) const) & BitsColumn::getBit,
            "key"_a);
    cls.def("getBit", (BitsColumn::IntT (BitsColumn::*)(std::string const &) const) & BitsColumn::getBit,
            "name"_a);
    cls.def("getMask", (BitsColumn::IntT (BitsColumn::*)(Key<Flag> const &) const) & BitsColumn::getMask,
            "key"_a);
    cls.def("getMask", (BitsColumn::IntT (BitsColumn::*)(std::string const &) const) & BitsColumn::getMask,
            "name"_a);
}

PYBIND11_PLUGIN(baseColumnView) {
    py::module mod("baseColumnView");

    py::module::import("lsst.afw.table.schema");

    declareBaseColumnView(mod);
    declareBitsColumn(mod);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::table::<anonymous>
