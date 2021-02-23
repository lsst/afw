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
#include "pybind11/stl.h"
#include "lsst/utils/python.h"

#include <cmath>
#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/image/Filter.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyFilterProperty = py::class_<FilterProperty, std::shared_ptr<FilterProperty>>;

using PyFilter = py::class_<Filter, std::shared_ptr<Filter>, typehandling::Storable>;

void declareFilterProperty(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(PyFilterProperty(wrappers.module, "FilterProperty"), [](auto &mod,
                                                                              auto &clsFilterProperty) {
        clsFilterProperty.def(py::init<std::string const &, double, double, double, bool>(), "name"_a,
                              "lambdaEff"_a, "lambdaMin"_a = NAN, "lambdaMax"_a = NAN, "force"_a = false);
        // note: metadata should be defaulted with "metadata"_a=daf::base::PropertySet()
        // but that causes an error about copying when the Python extension is imported
        clsFilterProperty.def(py::init<std::string const &, daf::base::PropertySet const &, bool>(), "name"_a,
                              "metadata"_a, "force"_a = false);
        clsFilterProperty.def(
                "__eq__",
                [](FilterProperty const &self, FilterProperty const &other) { return self == other; },
                py::is_operator());
        clsFilterProperty.def(
                "__ne__",
                [](FilterProperty const &self, FilterProperty const &other) { return self != other; },
                py::is_operator());
        clsFilterProperty.def("getName", &FilterProperty::getName);
        clsFilterProperty.def("getLambdaEff", &FilterProperty::getLambdaEff);
        clsFilterProperty.def("getLambdaMin", &FilterProperty::getLambdaMin);
        clsFilterProperty.def("getLambdaMax", &FilterProperty::getLambdaMax);
        clsFilterProperty.def_static("reset", &FilterProperty::reset);
        clsFilterProperty.def_static("lookup", &FilterProperty::lookup, "name"_a);
    });
}
void declareFilter(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(PyFilter(wrappers.module, "Filter"), [](auto &mod, auto &clsFilter) {
        clsFilter.def(py::init<std::string const &, bool const>(), "name"_a, "force"_a = false);
        clsFilter.def(py::init<int>(), "id"_a = Filter::UNKNOWN);
        clsFilter.def(py::init<std::shared_ptr<daf::base::PropertySet const>, bool const>(), "metadata"_a,
                      "force"_a = false);
        clsFilter.def(
                "__eq__", [](Filter const &self, Filter const &other) { return self == other; },
                py::is_operator());
        clsFilter.def(
                "__ne__", [](Filter const &self, Filter const &other) { return self != other; },
                py::is_operator());
        clsFilter.def_readonly_static("AUTO", &Filter::AUTO);
        clsFilter.def_readonly_static("UNKNOWN", &Filter::UNKNOWN);
        clsFilter.def("getId", &Filter::getId);
        clsFilter.def("getName", &Filter::getName);
        clsFilter.def("getCanonicalName", &Filter::getCanonicalName);
        clsFilter.def("getAliases", &Filter::getAliases);
        clsFilter.def("getFilterProperty", &Filter::getFilterProperty);
        clsFilter.def_static("reset", &Filter::reset);
        clsFilter.def_static("define", &Filter::define, "filterProperty"_a, "id"_a = Filter::AUTO,
                             "force"_a = false);
        clsFilter.def_static("defineAlias", &Filter::defineAlias, "oldName"_a, "newName"_a,
                             "force"_a = false);
        clsFilter.def_static("getNames", &Filter::getNames);
    });
}
}  // namespace
void wrapFilter(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.addSignatureDependency("lsst.daf.base");
    wrappers.addInheritanceDependency("lsst.afw.typehandling");
    declareFilterProperty(wrappers);
    declareFilter(wrappers);
    wrappers.wrap(
            [](auto &mod) { mod.def("stripFilterKeywords", &detail::stripFilterKeywords, "metadata"_a); });
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
