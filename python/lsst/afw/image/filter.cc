/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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

#include "lsst/daf/base/PropertySet.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/image/Filter.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyFilterProperty = py::class_<FilterProperty, std::shared_ptr<FilterProperty>>;

using PyFilter = py::class_<Filter, std::shared_ptr<Filter>>;

PYBIND11_PLUGIN(filter) {
    py::module mod("filter");

    py::module::import("lsst.daf.base");
    py::module::import("lsst.pex.policy");

    mod.def("stripFilterKeywords", &detail::stripFilterKeywords, "metadata"_a);

    PyFilterProperty clsFilterProperty(mod, "FilterProperty");
    clsFilterProperty.def(py::init<std::string const &, double, bool>(),
                          "name"_a, "lambdaEff"_a, "force"_a=false);
    // note: metadata should be defaulted with "metadata"_a=daf::base::PropertySet()
    // but that causes an error about copying when the Python extension is imported
    clsFilterProperty.def(py::init<std::string const &, daf::base::PropertySet const &, bool>(),
                          "name"_a, "metadata"_a, "force"_a=false);
    clsFilterProperty.def(py::init<std::string const &, pex::policy::Policy const &, bool>(),
                          "name"_a, "policy"_a, "force"_a=false);
    clsFilterProperty.def("__eq__",
                  [](FilterProperty const & self, FilterProperty const & other) { return self == other; },
                  py::is_operator());
    clsFilterProperty.def("__ne__",
                  [](FilterProperty const & self, FilterProperty const & other) { return self != other; },
                  py::is_operator());
    clsFilterProperty.def("getName", &FilterProperty::getName);
    clsFilterProperty.def("getLambdaEff", &FilterProperty::getLambdaEff);
    clsFilterProperty.def_static("reset", &FilterProperty::reset);
    clsFilterProperty.def_static("lookup", &FilterProperty::lookup, "name"_a);

    PyFilter clsFilter(mod, "Filter");
    clsFilter.def(py::init<std::string const &, bool const>(), "name"_a, "force"_a=false);
    clsFilter.def(py::init<int>(), "id"_a=Filter::UNKNOWN);
    clsFilter.def(py::init<std::shared_ptr<daf::base::PropertySet const>, bool const>(),
                  "metadata"_a, "force"_a=false);
    clsFilter.def("__eq__",
                  [](Filter const & self, Filter const & other) { return self == other; },
                  py::is_operator());
    clsFilter.def("__ne__",
                  [](Filter const & self, Filter const & other) { return self != other; },
                  py::is_operator());
    clsFilter.def_readonly_static("AUTO", &Filter::AUTO);
    clsFilter.def_readonly_static("UNKNOWN", &Filter::UNKNOWN);
    clsFilter.def("getId", &Filter::getId);
    clsFilter.def("getName", &Filter::getName);
    clsFilter.def("getCanonicalName", &Filter::getCanonicalName);
    clsFilter.def("getAliases", &Filter::getAliases);
    clsFilter.def("getFilterProperty", &Filter::getFilterProperty);
    clsFilter.def_static("reset", &Filter::reset);
    clsFilter.def_static("define", &Filter::define, "filterProperty"_a, "id"_a=Filter::AUTO, "force"_a=false);
    clsFilter.def_static("defineAlias", &Filter::defineAlias, "oldName"_a, "newName"_a, "force"_a=false);
    clsFilter.def_static("getNames", &Filter::getNames);

    return mod.ptr();
}

}}}}  // namespace lsst::afw::image::<anonymous>
