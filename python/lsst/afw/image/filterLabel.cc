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

#include <exception>

#include "lsst/utils/python.h"

#include "lsst/afw/image/FilterLabel.h"
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/table/io/python.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyFilterLabel = py::class_<FilterLabel, std::shared_ptr<FilterLabel>, typehandling::Storable>;

// Macro to convert an exception without defining a global handler for it
#define _DELEGATE_EXCEPTION(call, cpp_ex, py_ex) \
    try {                                        \
        return call;                             \
    } catch (cpp_ex const& e) {                  \
        std::throw_with_nested(py_ex(e.what())); \
    }

PyFilterLabel declare(py::module& mod) { return PyFilterLabel(mod, "FilterLabel"); }

void define(PyFilterLabel& cls) {
    cls.def_static("fromBandPhysical", &FilterLabel::fromBandPhysical, "band"_a, "physical"_a);
    cls.def_static("fromBand", &FilterLabel::fromBand, "band"_a);
    cls.def_static("fromPhysical", &FilterLabel::fromPhysical, "physical"_a);

    cls.def("hasBandLabel", &FilterLabel::hasBandLabel);
    cls.def_property_readonly("bandLabel", [](FilterLabel const& label) {
        _DELEGATE_EXCEPTION(label.getBandLabel(), pex::exceptions::LogicError, std::runtime_error);
    });
    cls.def("hasPhysicalLabel", &FilterLabel::hasPhysicalLabel);
    cls.def_property_readonly("physicalLabel", [](FilterLabel const& label) {
        _DELEGATE_EXCEPTION(label.getPhysicalLabel(), pex::exceptions::LogicError, std::runtime_error);
    });
}

PYBIND11_MODULE(filterLabel, mod) {
    // import inheritance dependencies
    py::module::import("lsst.afw.typehandling");
    // then declare classes
    auto cls = declare(mod);
    // then import dependencies used in method signatures
    // none
    // and now we can safely define methods and other attributes
    define(cls);
}

}  // namespace
}  // namespace image
}  // namespace afw
}  // namespace lsst
