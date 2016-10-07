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

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"

#include "lsst/afw/math/functionLibrary.h"
#include "lsst/afw/math/Function.h"

namespace py = pybind11;

using namespace pybind11::literals;

using namespace lsst::afw::math;

PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

template <typename ReturnT>
void declarePolynomialFunctions(py::module &mod, const std::string & suffix) {
    /* PolynomialFunction1 */
    py::class_<PolynomialFunction1<ReturnT>,
               std::shared_ptr<PolynomialFunction1<ReturnT>>,
               Function1<ReturnT>> 
                   clsPolynomialFunction1(mod, ("PolynomialFunction1" + suffix).c_str());

    /* PolynomialFunction1 Constructors */
    clsPolynomialFunction1.def(py::init<unsigned int>());
    clsPolynomialFunction1.def(py::init<std::vector<double> const &>());

    /* PolynomialFunction1 Members */
    clsPolynomialFunction1.def("__call__", &PolynomialFunction1<ReturnT>::operator());
    clsPolynomialFunction1.def("getOrder", &PolynomialFunction1<ReturnT>::getOrder);
    clsPolynomialFunction1.def("clone", &PolynomialFunction1<ReturnT>::clone);

    /* PolynomialFunction2 */
    py::class_<PolynomialFunction2<ReturnT>,
               std::shared_ptr<PolynomialFunction2<ReturnT>>,
               BasePolynomialFunction2<ReturnT>>
                   clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());
    /* PolynomialFunction2 Constructors */
    clsPolynomialFunction2.def(py::init<unsigned int>());
    clsPolynomialFunction2.def(py::init<std::vector<double> const &>());
    
    /* PolynomialFunction2 Members */
    clsPolynomialFunction2.def("__call__", &PolynomialFunction2<ReturnT>::operator());
    clsPolynomialFunction2.def("clone", &PolynomialFunction2<ReturnT>::clone);
    clsPolynomialFunction2.def("getOrder", &PolynomialFunction2<ReturnT>::getOrder);
    clsPolynomialFunction2.def("getDFuncDParameters", &PolynomialFunction2<ReturnT>::getDFuncDParameters);
};

template <typename ReturnT>
void declareChebyshevFunctions(py::module &mod, const std::string & suffix) {
    /* Chebyshev1Function1 */
    py::class_<Chebyshev1Function1<ReturnT>,
               std::shared_ptr<Chebyshev1Function1<ReturnT>>,
               Function1<ReturnT>> clsChebyshev1Function1(mod, ("Chebyshev1Function1" + suffix).c_str());
    /* Chebyshev1Function1 Consructors */
    clsChebyshev1Function1.def(py::init<unsigned int, double, double>(),
                              "order"_a,
                              "minX"_a=-1,
                              "maxX"_a=1);
    clsChebyshev1Function1.def(py::init<std::vector<double>, double, double>(),
                              "params"_a,
                              "minX"_a=-1,
                              "maxX"_a=1);
    /* Chebyshev1Function1 Members */
    clsChebyshev1Function1.def("__call__", &Chebyshev1Function1<ReturnT>::operator());
    clsChebyshev1Function1.def("getMinX", &Chebyshev1Function1<ReturnT>::getMinX);
    clsChebyshev1Function1.def("getMaxX", &Chebyshev1Function1<ReturnT>::getMaxX);
    clsChebyshev1Function1.def("getOrder", &Chebyshev1Function1<ReturnT>::getOrder);
    clsChebyshev1Function1.def("clone", &Chebyshev1Function1<ReturnT>::clone);
    
    /* Chebyshev1Function2 */
    py::class_<Chebyshev1Function2<ReturnT>,
               std::shared_ptr<Chebyshev1Function2<ReturnT>>,
               BasePolynomialFunction2<ReturnT>>
                   clsChebyshev1Function2(mod, ("Chebyshev1Function2" + suffix).c_str());
    /* Chebyshev1Function2 Consructors */
    clsChebyshev1Function2.def(py::init<unsigned int, lsst::afw::geom::Box2D const &>(),
                              "order"_a,
                              "xyRange"_a=
                                  lsst::afw::geom::Box2D(lsst::afw::geom::Point2D(-1.0, -1.0),
                                                         lsst::afw::geom::Point2D( 1.0,  1.0)));
    clsChebyshev1Function2.def(py::init<std::vector<double>, lsst::afw::geom::Box2D const &>(),
                              "order"_a,
                              "xyRange"_a=
                                  lsst::afw::geom::Box2D(lsst::afw::geom::Point2D(-1.0, -1.0),
                                                         lsst::afw::geom::Point2D( 1.0,  1.0)));
    /* Chebyshev1Function2 Members */
    clsChebyshev1Function2.def("__call__", &Chebyshev1Function2<ReturnT>::operator());
    clsChebyshev1Function2.def("clone", &Chebyshev1Function2<ReturnT>::clone);
    clsChebyshev1Function2.def("getOrder", &Chebyshev1Function2<ReturnT>::getOrder);
    clsChebyshev1Function2.def("getXYRange", &Chebyshev1Function2<ReturnT>::getXYRange);
    clsChebyshev1Function2.def("truncate", &Chebyshev1Function2<ReturnT>::truncate);
};

template <typename ReturnT>
void declareGaussianFunctions(py::module &mod, const std::string & suffix) {
    /* GaussianFunction1 */
    py::class_<GaussianFunction1<ReturnT>,
               std::shared_ptr<GaussianFunction1<ReturnT>>,
               Function1<ReturnT>>
                   clsGaussianFunction1(mod, ("GaussianFunction1" + suffix).c_str());
    /* GaussianFunction1 Constructors */
    clsGaussianFunction1.def(py::init<double>());
    /* GaussianFunction1 Members */
    clsGaussianFunction1.def("__call__", &GaussianFunction1<ReturnT>::operator());
    clsGaussianFunction1.def("clone", &GaussianFunction1<ReturnT>::clone);

    /* GaussianFunction2 */
    py::class_<GaussianFunction2<ReturnT>,
               std::shared_ptr<GaussianFunction2<ReturnT>>,
               Function2<ReturnT>>
                   clsGaussianFunction2(mod, ("GaussianFunction2" + suffix).c_str());
    /* GaussianFunction2 Constructors */
    clsGaussianFunction2.def(py::init<double, double, double>(),
                             "sigma1"_a, "sigma2"_a, "angle"_a=0.0);
    /* GaussianFunction2 Members */
    clsGaussianFunction2.def("__call__", &GaussianFunction2<ReturnT>::operator());
    clsGaussianFunction2.def("clone", &GaussianFunction2<ReturnT>::clone);

    /* DoubleGaussianFunction2 */
    py::class_<DoubleGaussianFunction2<ReturnT>,
               std::shared_ptr<DoubleGaussianFunction2<ReturnT>>,
               Function2<ReturnT>>
                   clsDoubleGaussianFunction2(mod, ("DoubleGaussianFunction2" + suffix).c_str());
    /* DoubleGaussianFunction2 Constructors */
    clsDoubleGaussianFunction2.def(py::init<double, double, double>(),
                             "sigma1"_a, "sigma2"_a=0, "ampl"_a=0);
    /* DoubleGaussianFunction2 Members */
    clsDoubleGaussianFunction2.def("__call__", &DoubleGaussianFunction2<ReturnT>::operator());
    clsDoubleGaussianFunction2.def("clone", &DoubleGaussianFunction2<ReturnT>::clone);
};

template <typename ReturnT>
void declareIntegerDeltaFunctions(py::module &mod, const std::string & suffix) {
    /* IntegerDeltaFunction2 */
    py::class_<IntegerDeltaFunction2<ReturnT>,
               std::shared_ptr<IntegerDeltaFunction2<ReturnT>>,
               Function2<ReturnT>>
                   clsIntegerDeltaFunction2(mod, ("IntegerDeltaFunction2" + suffix).c_str());
    /* IntegerDeltaFunction2 Constructors */
    clsIntegerDeltaFunction2.def(py::init<double, double>());
    /* IntegerDeltaFunction2 Members */
    clsIntegerDeltaFunction2.def("__call__", &IntegerDeltaFunction2<ReturnT>::operator());
    clsIntegerDeltaFunction2.def("clone", &IntegerDeltaFunction2<ReturnT>::clone);
};

template <typename ReturnT>
void declareLanczosFunctions(py::module &mod, const std::string & suffix) {
    /* LanczosFunction1 */
    py::class_<LanczosFunction1<ReturnT>,
               std::shared_ptr<LanczosFunction1<ReturnT>>,
               Function1<ReturnT>>
                   clsLanczosFunction1(mod, ("LanczosFunction1" + suffix).c_str());
    /* LanczosFunction1 Constructors */
    clsLanczosFunction1.def(py::init<unsigned int, double>(),
                            "n"_a, "xOffset"_a=0.0);
    /* LanczosFunction1 Members */
    clsLanczosFunction1.def("__call__", &LanczosFunction1<ReturnT>::operator());
    clsLanczosFunction1.def("clone", &LanczosFunction1<ReturnT>::clone);
    clsLanczosFunction1.def("getOrder", &LanczosFunction1<ReturnT>::getOrder);
    
    /* LanczosFunction2 */
    py::class_<LanczosFunction2<ReturnT>,
               std::shared_ptr<LanczosFunction2<ReturnT>>,
               Function2<ReturnT>>
                   clsLanczosFunction2(mod, ("LanczosFunction2" + suffix).c_str());
    /* LanczosFunction2 Constructors */
    clsLanczosFunction2.def(py::init<unsigned int, double, double>(),
                            "n"_a, "xOffset"_a=0.0, "yOffset"_a=0.0);
    /* LanczosFunction2 Members */
    clsLanczosFunction2.def("__call__", &LanczosFunction2<ReturnT>::operator());
    clsLanczosFunction2.def("clone", &LanczosFunction2<ReturnT>::clone);
    clsLanczosFunction2.def("getOrder", &LanczosFunction2<ReturnT>::getOrder);
};

PYBIND11_PLUGIN(_functionLibrary) {
    py::module mod("_functionLibrary", "Python wrapper for afw _functionLibrary library");
    
    declarePolynomialFunctions<float>(mod, "F");
    declareChebyshevFunctions<float>(mod, "F");
    declareGaussianFunctions<float>(mod, "F");
    declareIntegerDeltaFunctions<float>(mod, "F");
    declareLanczosFunctions<float>(mod, "F");
    
    declarePolynomialFunctions<double>(mod, "D");
    declareChebyshevFunctions<double>(mod, "D");
    declareGaussianFunctions<double>(mod, "D");
    declareIntegerDeltaFunctions<double>(mod, "D");
    declareLanczosFunctions<double>(mod, "D");

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}