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

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/math/Function.h"

namespace py = pybind11;

using namespace lsst::afw::math;

PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

template<typename ReturnT>
void declareFunctions(py::module &mod, const std::string & suffix){
    /* Function */
    py::class_<Function<ReturnT>> clsFunction(mod, ("Function"+suffix).c_str());
    /* Function Constructors */
    clsFunction.def(py::init<unsigned int>());
    clsFunction.def(py::init<std::vector<double> const &>());
    /* Function Members */
    clsFunction.def("getNParameters", &Function<ReturnT>::getNParameters);
    clsFunction.def("setParameters", &Function<ReturnT>::setParameters);

    /* Function1 */
    py::class_<Function1<ReturnT>, std::shared_ptr<Function1<ReturnT>>, Function<ReturnT>> clsFunction1(mod, ("Function1"+suffix).c_str());

    /* Function2 */
    py::class_<Function2<ReturnT>, Function<ReturnT>> clsFunction2(mod, ("Function2"+suffix).c_str());

    /* BasePolynomialFunction2 */
    py::class_<BasePolynomialFunction2<ReturnT>,
               std::shared_ptr<BasePolynomialFunction2<ReturnT>>,
               Function2<ReturnT> > 
                   clsBasePolynomialFunction2(mod, ("BasePolynomialFunction2" + suffix).c_str());
    clsBasePolynomialFunction2.def_static("nParametersFromOrder",
                                   BasePolynomialFunction2<ReturnT>::nParametersFromOrder);
    /* NullFunction1 */
    py::class_<NullFunction1<ReturnT>,
               std::shared_ptr<NullFunction1<ReturnT>>,
               Function1<ReturnT>> 
                   clsNullFunction1(mod, ("NullFunction1" + suffix).c_str());

    /* NullFunction1 Constructors */
    clsNullFunction1.def(py::init<>());

    /* NullFunction1 Members */
    clsNullFunction1.def("clone", &NullFunction1<ReturnT>::clone);

    /* NullFunction2 */
    py::class_<NullFunction2<ReturnT>,
               std::shared_ptr<NullFunction2<ReturnT>>,
               Function2<ReturnT>> 
                   clsNullFunction2(mod, ("NullFunction2" + suffix).c_str());

    /* NullFunction2 Constructors */
    clsNullFunction2.def(py::init<>());

    /* NullFunction2 Members */
    clsNullFunction2.def("clone", &NullFunction1<ReturnT>::clone);
};

PYBIND11_PLUGIN(_function) {
    py::module mod("_function", "Python wrapper for afw _function library");
    
    declareFunctions<float>(mod, "F");
    declareFunctions<double>(mod, "D");
    

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}