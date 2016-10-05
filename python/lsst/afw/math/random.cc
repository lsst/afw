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

#include "lsst/afw/image/Image.h"
#include "lsst/afw/math/Random.h"

namespace py = pybind11;

using namespace lsst::afw::math;
using namespace pybind11::literals;

template <typename ImageT>
void declareRandomImage(py::module & mod) {
    mod.def("randomUniformImage", (void (*)(ImageT *, Random &)) randomUniformImage<ImageT>);
    mod.def("randomUniformPosImage", (void (*)(ImageT *, Random &)) randomUniformPosImage<ImageT>);
    mod.def("randomUniformIntImage", (void (*)(ImageT *, Random &, unsigned long)) randomUniformIntImage<ImageT>);
    mod.def("randomFlatImage", (void (*)(ImageT *, Random &, double const, double const)) randomFlatImage<ImageT>);
    mod.def("randomGaussianImage", (void (*)(ImageT *, Random &)) randomGaussianImage<ImageT>);
    mod.def("randomChisqImage", (void (*)(ImageT *, Random &, double const)) randomChisqImage<ImageT>);
    mod.def("randomPoissonImage", (void (*)(ImageT *, Random &, double const)) randomPoissonImage<ImageT>);
}

PYBIND11_PLUGIN(_random) {
    py::module mod("_random", "Python wrapper for afw _random library");

    py::class_<Random> clsRandom(mod, "Random");

    /* Member types and enums */
    py::enum_<Random::Algorithm>(clsRandom, "Algorithm")
        .value("MT19937", Random::Algorithm::MT19937)
        .value("RANLXS0", Random::Algorithm::RANLXS0)
        .value("RANLXS1", Random::Algorithm::RANLXS1)
        .value("RANLXS2", Random::Algorithm::RANLXS2)
        .value("RANLXD1", Random::Algorithm::RANLXD1)
        .value("RANLXD2", Random::Algorithm::RANLXD2)
        .value("RANLUX", Random::Algorithm::RANLUX)
        .value("RANLUX389", Random::Algorithm::RANLUX389)
        .value("CMRG", Random::Algorithm::CMRG)
        .value("MRG", Random::Algorithm::MRG)
        .value("TAUS", Random::Algorithm::TAUS)
        .value("TAUS2", Random::Algorithm::TAUS2)
        .value("GFSR4", Random::Algorithm::GFSR4)
        .value("NUM_ALGORITHMS", Random::Algorithm::NUM_ALGORITHMS)
        .export_values();

    /* Constructors */
    clsRandom.def(py::init<Random::Algorithm, unsigned long>(),
            "algorithm"_a=Random::Algorithm::MT19937, "seed"_a=1);
    clsRandom.def(py::init<std::string const &, unsigned long>(),
            "algorithm"_a, "seed"_a=1);
    clsRandom.def(py::init<lsst::pex::policy::Policy::Ptr const>(),
            "policy"_a);

    /* Members */
    clsRandom.def("deepCopy", &Random::deepCopy);
    clsRandom.def("getAlgorithm", &Random::getAlgorithm);
    clsRandom.def("getAlgorithmName", &Random::getAlgorithmName);
    clsRandom.def_static("getAlgorithmNames", &Random::getAlgorithmNames);
    clsRandom.def("getSeed", &Random::getSeed);
    clsRandom.def("uniform", &Random::uniform);
    clsRandom.def("uniformPos", &Random::uniformPos);
    clsRandom.def("uniformInt", &Random::uniformInt);
    clsRandom.def("flat", &Random::flat);
    clsRandom.def("gaussian", &Random::gaussian);
    clsRandom.def("chisq", &Random::chisq);
    clsRandom.def("poisson", &Random::poisson);

    // getState and setState are special, their std::string cannot
    // be converted to a Python string (needs to go to bytes instead)
    // thus use the same solution as employed with Swig
    clsRandom.def("getState", [](Random & self) -> py::object {
        std::string state = self.getState();
        return py::object{PyBytes_FromStringAndSize(state.data(), state.size()), false};
    });
    clsRandom.def("setState", [](Random & self, py::bytes const & state) {
        self.setState(state);
    });

    /* Module level */
    declareRandomImage<lsst::afw::image::Image<double>>(mod);
    declareRandomImage<lsst::afw::image::Image<float>>(mod);

    return mod.ptr();
}