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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"

#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/afw/math/Function.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_DECLARE_HOLDER_TYPE(MyType, std::shared_ptr<MyType>);

namespace lsst {
namespace afw {
namespace math {

template <typename ReturnT>
void declarePolynomialFunctions(py::module &mod, const std::string &suffix) {
    /* PolynomialFunction1 */
    py::class_<PolynomialFunction1<ReturnT>, std::shared_ptr<PolynomialFunction1<ReturnT>>,
               Function1<ReturnT>>
            clsPolynomialFunction1(mod, ("PolynomialFunction1" + suffix).c_str());

    clsPolynomialFunction1.def(py::init<std::vector<double> const &>(), "params"_a);
    clsPolynomialFunction1.def(py::init<unsigned int>(), "order"_a);

    clsPolynomialFunction1.def("__call__", &PolynomialFunction1<ReturnT>::operator(), "x"_a);
    clsPolynomialFunction1.def("clone", &PolynomialFunction1<ReturnT>::clone);
    clsPolynomialFunction1.def("isLinearCombination", &PolynomialFunction1<ReturnT>::isLinearCombination);
    clsPolynomialFunction1.def("getOrder", &PolynomialFunction1<ReturnT>::getOrder);
    clsPolynomialFunction1.def("toString", &PolynomialFunction1<ReturnT>::toString, "prefix"_a = "");

    /* PolynomialFunction2 */
    py::class_<PolynomialFunction2<ReturnT>, std::shared_ptr<PolynomialFunction2<ReturnT>>,
               BasePolynomialFunction2<ReturnT>>
            clsPolynomialFunction2(mod, ("PolynomialFunction2" + suffix).c_str());

    clsPolynomialFunction2.def(py::init<std::vector<double> const &>(), "params"_a);
    clsPolynomialFunction2.def(py::init<unsigned int>(), "order"_a);

    clsPolynomialFunction2.def("__call__", &PolynomialFunction2<ReturnT>::operator(), "x"_a, "y"_a);
    clsPolynomialFunction2.def("clone", &PolynomialFunction2<ReturnT>::clone);
    clsPolynomialFunction2.def("getOrder", &PolynomialFunction2<ReturnT>::getOrder);
    clsPolynomialFunction2.def("getDFuncDParameters", &PolynomialFunction2<ReturnT>::getDFuncDParameters);
    clsPolynomialFunction2.def("toString", &PolynomialFunction2<ReturnT>::toString, "prefix"_a = "");
    clsPolynomialFunction2.def("isPersistable", &PolynomialFunction2<ReturnT>::isPersistable);
};

template <typename ReturnT>
void declareChebyshevFunctions(py::module &mod, const std::string &suffix) {
    /* Chebyshev1Function1 */
    py::class_<Chebyshev1Function1<ReturnT>, std::shared_ptr<Chebyshev1Function1<ReturnT>>,
               Function1<ReturnT>>
            clsChebyshev1Function1(mod, ("Chebyshev1Function1" + suffix).c_str());

    clsChebyshev1Function1.def(py::init<std::vector<double>, double, double>(), "params"_a, "minX"_a = -1,
                               "maxX"_a = 1);
    clsChebyshev1Function1.def(py::init<unsigned int, double, double>(), "order"_a, "minX"_a = -1,
                               "maxX"_a = 1);

    clsChebyshev1Function1.def("__call__", &Chebyshev1Function1<ReturnT>::operator(), "x"_a);
    clsChebyshev1Function1.def("clone", &Chebyshev1Function1<ReturnT>::clone);
    clsChebyshev1Function1.def("getMinX", &Chebyshev1Function1<ReturnT>::getMinX);
    clsChebyshev1Function1.def("getMaxX", &Chebyshev1Function1<ReturnT>::getMaxX);
    clsChebyshev1Function1.def("getOrder", &Chebyshev1Function1<ReturnT>::getOrder);
    clsChebyshev1Function1.def("isLinearCombination", &Chebyshev1Function1<ReturnT>::isLinearCombination);
    clsChebyshev1Function1.def("toString", &Chebyshev1Function1<ReturnT>::toString, "prefix"_a = "");

    /* Chebyshev1Function2 */
    py::class_<Chebyshev1Function2<ReturnT>, std::shared_ptr<Chebyshev1Function2<ReturnT>>,
               BasePolynomialFunction2<ReturnT>>
            clsChebyshev1Function2(mod, ("Chebyshev1Function2" + suffix).c_str());

    clsChebyshev1Function2.def(py::init<std::vector<double>, geom::Box2D const &>(), "params"_a,
                               "xyRange"_a = geom::Box2D(geom::Point2D(-1.0, -1.0), geom::Point2D(1.0, 1.0)));
    clsChebyshev1Function2.def(py::init<unsigned int, geom::Box2D const &>(), "order"_a,
                               "xyRange"_a = geom::Box2D(geom::Point2D(-1.0, -1.0), geom::Point2D(1.0, 1.0)));

    clsChebyshev1Function2.def("__call__", &Chebyshev1Function2<ReturnT>::operator(), "x"_a, "y"_a);
    clsChebyshev1Function2.def("clone", &Chebyshev1Function2<ReturnT>::clone);
    clsChebyshev1Function2.def("getXYRange", &Chebyshev1Function2<ReturnT>::getXYRange);
    clsChebyshev1Function2.def("truncate", &Chebyshev1Function2<ReturnT>::truncate, "order"_a);
    clsChebyshev1Function2.def("toString", &Chebyshev1Function2<ReturnT>::toString, "prefix"_a = "");
    clsChebyshev1Function2.def("isPersistable", &Chebyshev1Function2<ReturnT>::isPersistable);
};

template <typename ReturnT>
void declareGaussianFunctions(py::module &mod, const std::string &suffix) {
    /* GaussianFunction1 */
    py::class_<GaussianFunction1<ReturnT>, std::shared_ptr<GaussianFunction1<ReturnT>>, Function1<ReturnT>>
            clsGaussianFunction1(mod, ("GaussianFunction1" + suffix).c_str());

    clsGaussianFunction1.def(py::init<double>(), "sigma"_a);

    clsGaussianFunction1.def("__call__", &GaussianFunction1<ReturnT>::operator(), "x"_a);
    clsGaussianFunction1.def("clone", &GaussianFunction1<ReturnT>::clone);
    clsGaussianFunction1.def("toString", &GaussianFunction1<ReturnT>::toString, "prefix"_a = "");

    /* GaussianFunction2 */
    py::class_<GaussianFunction2<ReturnT>, std::shared_ptr<GaussianFunction2<ReturnT>>, Function2<ReturnT>>
            clsGaussianFunction2(mod, ("GaussianFunction2" + suffix).c_str());

    clsGaussianFunction2.def(py::init<double, double, double>(), "sigma1"_a, "sigma2"_a, "angle"_a = 0.0);

    clsGaussianFunction2.def("__call__", &GaussianFunction2<ReturnT>::operator(), "x"_a, "y"_a);
    clsGaussianFunction2.def("clone", &GaussianFunction2<ReturnT>::clone);
    clsGaussianFunction2.def("toString", &GaussianFunction2<ReturnT>::toString, "prefix"_a = "");
    clsGaussianFunction2.def("isPersistable", &GaussianFunction2<ReturnT>::isPersistable);

    /* DoubleGaussianFunction2 */
    py::class_<DoubleGaussianFunction2<ReturnT>, std::shared_ptr<DoubleGaussianFunction2<ReturnT>>,
               Function2<ReturnT>>
            clsDoubleGaussianFunction2(mod, ("DoubleGaussianFunction2" + suffix).c_str());

    clsDoubleGaussianFunction2.def(py::init<double, double, double>(), "sigma1"_a, "sigma2"_a = 0,
                                   "ampl"_a = 0);

    clsDoubleGaussianFunction2.def("__call__", &DoubleGaussianFunction2<ReturnT>::operator(), "x"_a, "y"_a);
    clsDoubleGaussianFunction2.def("clone", &DoubleGaussianFunction2<ReturnT>::clone);
    clsDoubleGaussianFunction2.def("toString", &DoubleGaussianFunction2<ReturnT>::toString, "prefix"_a = "");
    clsDoubleGaussianFunction2.def("isPersistable", &DoubleGaussianFunction2<ReturnT>::isPersistable);
};

template <typename ReturnT>
void declareIntegerDeltaFunctions(py::module &mod, const std::string &suffix) {
    /* IntegerDeltaFunction1 */
    py::class_<IntegerDeltaFunction1<ReturnT>, std::shared_ptr<IntegerDeltaFunction1<ReturnT>>,
               Function1<ReturnT>>
            clsIntegerDeltaFunction1(mod, ("IntegerDeltaFunction1" + suffix).c_str());

    clsIntegerDeltaFunction1.def(py::init<double>(), "xo"_a);

    clsIntegerDeltaFunction1.def("__call__", &IntegerDeltaFunction1<ReturnT>::operator(), "x"_a);
    clsIntegerDeltaFunction1.def("clone", &IntegerDeltaFunction1<ReturnT>::clone);
    clsIntegerDeltaFunction1.def("toString", &IntegerDeltaFunction1<ReturnT>::toString, "prefix"_a = "");

    /* IntegerDeltaFunction2 */
    py::class_<IntegerDeltaFunction2<ReturnT>, std::shared_ptr<IntegerDeltaFunction2<ReturnT>>,
               Function2<ReturnT>>
            clsIntegerDeltaFunction2(mod, ("IntegerDeltaFunction2" + suffix).c_str());

    clsIntegerDeltaFunction2.def(py::init<double, double>(), "xo"_a, "yo"_a);

    clsIntegerDeltaFunction2.def("__call__", &IntegerDeltaFunction2<ReturnT>::operator(), "x"_a, "y"_a);
    clsIntegerDeltaFunction2.def("clone", &IntegerDeltaFunction2<ReturnT>::clone);
    clsIntegerDeltaFunction2.def("toString", &IntegerDeltaFunction2<ReturnT>::toString, "prefix"_a = "");
};

template <typename ReturnT>
void declareLanczosFunctions(py::module &mod, const std::string &suffix) {
    /* LanczosFunction1 */
    py::class_<LanczosFunction1<ReturnT>, std::shared_ptr<LanczosFunction1<ReturnT>>, Function1<ReturnT>>
            clsLanczosFunction1(mod, ("LanczosFunction1" + suffix).c_str());

    clsLanczosFunction1.def(py::init<unsigned int, double>(), "n"_a, "xOffset"_a = 0.0);

    clsLanczosFunction1.def("__call__", &LanczosFunction1<ReturnT>::operator(), "x"_a);
    clsLanczosFunction1.def("clone", &LanczosFunction1<ReturnT>::clone);
    clsLanczosFunction1.def("getOrder", &LanczosFunction1<ReturnT>::getOrder);
    clsLanczosFunction1.def("toString", &LanczosFunction1<ReturnT>::toString, "prefix"_a = "");

    /* LanczosFunction2 */
    py::class_<LanczosFunction2<ReturnT>, std::shared_ptr<LanczosFunction2<ReturnT>>, Function2<ReturnT>>
            clsLanczosFunction2(mod, ("LanczosFunction2" + suffix).c_str());

    clsLanczosFunction2.def(py::init<unsigned int, double, double>(), "n"_a, "xOffset"_a = 0.0,
                            "yOffset"_a = 0.0);

    clsLanczosFunction2.def("__call__", &LanczosFunction2<ReturnT>::operator(), "x"_a, "y"_a);
    clsLanczosFunction2.def("clone", &LanczosFunction2<ReturnT>::clone);
    clsLanczosFunction2.def("getOrder", &LanczosFunction2<ReturnT>::getOrder);
    clsLanczosFunction2.def("toString", &LanczosFunction2<ReturnT>::toString, "prefix"_a = "");
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

    return mod.ptr();
}

}  // namespace math
}  // namespace afw
}  // namespace lsst
