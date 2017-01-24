/* 
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/table/io/pybind11.h"  // for declarePersistableFacade
#include "lsst/afw/math/Function.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {

template <typename ReturnT>
void declareFunction(py::module &mod, std::string const &suffix) {
    auto const name = "Function" + suffix;

    table::io::declarePersistableFacade<Function<ReturnT>>(mod, name.c_str());

    py::class_<Function<ReturnT>, std::shared_ptr<Function<ReturnT>>,
               table::io::PersistableFacade<Function<ReturnT>>, table::io::Persistable>
        cls(mod, name.c_str());

    cls.def(py::init<unsigned int>(), "nParams"_a);
    cls.def(py::init<std::vector<double> const &>(), "params"_a);

    cls.def("getNParameters", &Function<ReturnT>::getNParameters);
    cls.def("getParameters", &Function<ReturnT>::getParameters, py::return_value_policy::copy);
    cls.def("getParameter", &Function<ReturnT>::getParameter, "index"_a);
    cls.def("isLinearCombination", &Function<ReturnT>::isLinearCombination);
    cls.def("setParameter", &Function<ReturnT>::setParameter, "index"_a, "value"_a);
    cls.def("setParameters", &Function<ReturnT>::setParameters);
    cls.def("toString", &Function<ReturnT>::toString, "prefix"_a = "");
}

template <typename ReturnT>
void declareFunction1(py::module &mod, const std::string &suffix) {
    auto const name = "Function1" + suffix;

    table::io::declarePersistableFacade<Function1<ReturnT>>(mod, name.c_str());

    py::class_<Function1<ReturnT>, std::shared_ptr<Function1<ReturnT>>,
               table::io::PersistableFacade<Function1<ReturnT>>, Function<ReturnT>>
        cls(mod, name.c_str());

    cls.def("clone", &Function1<ReturnT>::clone);
    cls.def("__call__", &Function1<ReturnT>::operator(), "x"_a);
    cls.def("toString", &Function1<ReturnT>::toString, "prefix"_a = "");
    cls.def("computeCache", &Function1<ReturnT>::computeCache, "n"_a);
}

template <typename ReturnT>
void declareFunction2(py::module &mod, const std::string &suffix) {
    auto const name = "Function2" + suffix;

    table::io::declarePersistableFacade<Function2<ReturnT>>(mod, name.c_str());

    py::class_<Function2<ReturnT>, std::shared_ptr<Function2<ReturnT>>,
               table::io::PersistableFacade<Function2<ReturnT>>, Function<ReturnT>>
        cls(mod, name.c_str());

    cls.def("clone", &Function2<ReturnT>::clone);
    cls.def("__call__", &Function2<ReturnT>::operator(), "x"_a, "y"_a);
    cls.def("toString", &Function2<ReturnT>::toString, "prefix"_a = "");
    cls.def("getDFuncDParameters", &Function2<ReturnT>::getDFuncDParameters, "x"_a, "y"_a);
}

template <typename ReturnT>
void declareBasePolynomialFunction2(py::module &mod, const std::string &suffix) {
    auto const name = "BasePolynomialFunction2" + suffix;

    py::class_<BasePolynomialFunction2<ReturnT>, std::shared_ptr<BasePolynomialFunction2<ReturnT>>,
               Function2<ReturnT>>
        cls(mod, name.c_str());

    cls.def("getOrder", &BasePolynomialFunction2<ReturnT>::getOrder);
    cls.def("isLinearCombination", &BasePolynomialFunction2<ReturnT>::isLinearCombination);
    cls.def_static("nParametersFromOrder", &BasePolynomialFunction2<ReturnT>::nParametersFromOrder, "order"_a);
    cls.def_static("orderFromNParameters", &BasePolynomialFunction2<ReturnT>::orderFromNParameters, "nParameters"_a);
    cls.def("getDFuncDParameters", &BasePolynomialFunction2<ReturnT>::getDFuncDParameters, "x"_a, "y"_a);
}

template <typename ReturnT>
void declareNullFunction1(py::module &mod, const std::string &suffix) {
    auto const name = "NullFunction1" + suffix;

    py::class_<NullFunction1<ReturnT>, std::shared_ptr<NullFunction1<ReturnT>>, Function1<ReturnT>> cls(
        mod, name.c_str());

    cls.def(py::init<>());

    cls.def("clone", &NullFunction1<ReturnT>::clone);
}

template <typename ReturnT>
void declareNullFunction2(py::module &mod, const std::string &suffix) {
    auto const name = "NullFunction2" + suffix;

    py::class_<NullFunction2<ReturnT>, std::shared_ptr<NullFunction2<ReturnT>>, Function2<ReturnT>> cls(
        mod, name.c_str());

    cls.def(py::init<>());

    cls.def("clone", &NullFunction2<ReturnT>::clone);
}

template <typename ReturnT>
void declareAllFunctions(py::module &mod, const std::string &suffix) {
    declareFunction<ReturnT>(mod, suffix);
    declareFunction1<ReturnT>(mod, suffix);
    declareFunction2<ReturnT>(mod, suffix);
    declareBasePolynomialFunction2<ReturnT>(mod, suffix);
    declareNullFunction1<ReturnT>(mod, suffix);
    declareNullFunction2<ReturnT>(mod, suffix);
};

}  // namespace <anonymous>

PYBIND11_PLUGIN(_function) {
    py::module mod("_function", "Python wrapper for afw _function library");

    declareAllFunctions<float>(mod, "F");
    declareAllFunctions<double>(mod, "D");

    return mod.ptr();
}

}  // namespace math
}  // namespace afw
}  // namespace lsst