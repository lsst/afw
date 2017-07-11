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

#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/table/io/Persistable.h"

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/table/io/python.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::math;

PYBIND11_PLUGIN(_kernel) {
    py::module mod("_kernel", "Python wrapper for afw _kernel library");

    lsst::afw::table::io::python::declarePersistableFacade<Kernel>(mod, "Kernel");

    py::class_<Kernel, std::shared_ptr<Kernel>, lsst::daf::base::Persistable,
               lsst::afw::table::io::PersistableFacade<Kernel>, lsst::afw::table::io::Persistable>
            clsKernel(mod, "Kernel");

    clsKernel.def("clone", &Kernel::clone);
    clsKernel.def("resized", &Kernel::resized, "width"_a, "height"_a);
    clsKernel.def("computeImage", &Kernel::computeImage, "image"_a, "doNormalize"_a, "x"_a = 0.0,
                  "y"_a = 0.0);
    clsKernel.def("getDimensions", &Kernel::getDimensions);
    clsKernel.def("setDimensions", &Kernel::setDimensions);
    clsKernel.def("setWidth", &Kernel::setWidth);
    clsKernel.def("setHeight", &Kernel::setHeight);
    clsKernel.def("getWidth", &Kernel::getWidth);
    clsKernel.def("getHeight", &Kernel::getHeight);
    clsKernel.def("getCtr", &Kernel::getCtr);
    clsKernel.def("getCtrX", &Kernel::getCtrX);
    clsKernel.def("getCtrY", &Kernel::getCtrY);
    clsKernel.def("getBBox", &Kernel::getBBox);
    clsKernel.def("getNKernelParameters", &Kernel::getNKernelParameters);
    clsKernel.def("getNSpatialParameters", &Kernel::getNSpatialParameters);
    clsKernel.def("getSpatialFunction", &Kernel::getSpatialFunction);
    clsKernel.def("getSpatialFunctionList", &Kernel::getSpatialFunctionList);
    clsKernel.def("getKernelParameter", &Kernel::getKernelParameter);
    clsKernel.def("getKernelParameters", &Kernel::getKernelParameters);
    clsKernel.def("growBBox", &Kernel::growBBox);
    clsKernel.def("shrinkBBox", &Kernel::shrinkBBox);
    clsKernel.def("setCtr", &Kernel::setCtr);
    clsKernel.def("setCtrX", &Kernel::setCtrX);
    clsKernel.def("setCtrY", &Kernel::setCtrY);
    clsKernel.def("getSpatialParameters", &Kernel::getSpatialParameters);
    clsKernel.def("isSpatiallyVarying", &Kernel::isSpatiallyVarying);
    clsKernel.def("setKernelParameters",
                  (void (Kernel::*)(std::vector<double> const &)) & Kernel::setKernelParameters);
    clsKernel.def("setKernelParameters",
                  (void (Kernel::*)(std::pair<double, double> const &)) & Kernel::setKernelParameters);
    clsKernel.def("setSpatialParameters", &Kernel::setSpatialParameters);
    clsKernel.def("computeKernelParametersFromSpatialModel",
                  &Kernel::computeKernelParametersFromSpatialModel);
    clsKernel.def("toString", &Kernel::toString, "prefix"_a = "");
    clsKernel.def("computeCache", &Kernel::computeCache);
    clsKernel.def("getCacheSize", &Kernel::getCacheSize);

    py::class_<FixedKernel, std::shared_ptr<FixedKernel>, Kernel> clsFixedKernel(mod, "FixedKernel");

    clsFixedKernel.def(py::init<>());
    clsFixedKernel.def(py::init<lsst::afw::image::Image<Kernel::Pixel> const &>(), "image"_a);
    clsFixedKernel.def(py::init<lsst::afw::math::Kernel const &, lsst::afw::geom::Point2D const &>(),
                       "kernel"_a, "pos"_a);
    clsFixedKernel.def("clone", &FixedKernel::clone);
    clsFixedKernel.def("resized", &FixedKernel::resized, "width"_a, "height"_a);
    clsFixedKernel.def("toString", &FixedKernel::toString, "prefix"_a = "");
    clsFixedKernel.def("getSum", &FixedKernel::getSum);
    clsFixedKernel.def("isPersistable", &FixedKernel::isPersistable);

    py::class_<AnalyticKernel, std::shared_ptr<AnalyticKernel>, Kernel> clsAnalyticKernel(mod,
                                                                                          "AnalyticKernel");
    clsAnalyticKernel.def(py::init<>());
    // Workaround for NullSpatialFunction and py::arg not playing well with Citizen
    clsAnalyticKernel.def(py::init<int, int, AnalyticKernel::KernelFunction const &>(), "width"_a, "height"_a,
                          "kernelFunction"_a);
    clsAnalyticKernel.def(
            py::init<int, int, AnalyticKernel::KernelFunction const &, Kernel::SpatialFunction const &>(),
            "width"_a, "height"_a, "kernelFunction"_a, "spatialFunction"_a);
    clsAnalyticKernel.def(py::init<int, int, AnalyticKernel::KernelFunction const &,
                                   std::vector<Kernel::SpatialFunctionPtr> const &>(),
                          "width"_a, "height"_a, "kernelFunction"_a, "spatialFunctionList"_a);
    clsAnalyticKernel.def("clone", &AnalyticKernel::clone);
    clsAnalyticKernel.def("resized", &AnalyticKernel::resized, "width"_a, "height"_a);
    clsAnalyticKernel.def("computeImage", &AnalyticKernel::computeImage, "image"_a, "doNormalize"_a,
                          "x"_a = 0.0, "y"_a = 0.0);
    clsAnalyticKernel.def("getKernelParameters", &AnalyticKernel::getKernelParameters);
    clsAnalyticKernel.def("getKernelFunction", &AnalyticKernel::getKernelFunction);
    clsAnalyticKernel.def("toString", &AnalyticKernel::toString, "prefix"_a = "");
    clsAnalyticKernel.def("isPersistable", &AnalyticKernel::isPersistable);

    py::class_<DeltaFunctionKernel, std::shared_ptr<DeltaFunctionKernel>, Kernel> clsDeltaFunctionKernel(
            mod, "DeltaFunctionKernel");

    clsDeltaFunctionKernel.def(py::init<int, int, lsst::afw::geom::Point2I const &>(), "width"_a, "height"_a,
                               "point"_a);
    clsDeltaFunctionKernel.def("clone", &DeltaFunctionKernel::clone);
    clsDeltaFunctionKernel.def("resized", &DeltaFunctionKernel::resized, "width"_a, "height"_a);
    clsDeltaFunctionKernel.def("getPixel", &DeltaFunctionKernel::getPixel);
    clsDeltaFunctionKernel.def("toString", &DeltaFunctionKernel::toString, "prefix"_a = "");
    clsDeltaFunctionKernel.def("isPersistable", &DeltaFunctionKernel::isPersistable);

    py::class_<LinearCombinationKernel, std::shared_ptr<LinearCombinationKernel>, Kernel>
            clsLinearCombinationKernel(mod, "LinearCombinationKernel");

    clsLinearCombinationKernel.def(py::init<>());
    clsLinearCombinationKernel.def(py::init<KernelList const &, std::vector<double> const &>(),
                                   "kernelList"_a, "kernelParameters"_a);
    clsLinearCombinationKernel.def(py::init<KernelList const &, Kernel::SpatialFunction const &>(),
                                   "kernelList"_a, "spatialFunction"_a);
    clsLinearCombinationKernel.def(
            py::init<KernelList const &, std::vector<Kernel::SpatialFunctionPtr> const &>(), "kernelList"_a,
            "spatialFunctionList"_a);
    clsLinearCombinationKernel.def("clone", &LinearCombinationKernel::clone);
    clsLinearCombinationKernel.def("resized", &LinearCombinationKernel::resized, "width"_a, "height"_a);
    clsLinearCombinationKernel.def("getKernelParameters", &LinearCombinationKernel::getKernelParameters);
    clsLinearCombinationKernel.def("getKernelList", &LinearCombinationKernel::getKernelList);
    clsLinearCombinationKernel.def("getKernelSumList", &LinearCombinationKernel::getKernelSumList);
    clsLinearCombinationKernel.def("getNBasisKernels", &LinearCombinationKernel::getNBasisKernels);
    clsLinearCombinationKernel.def("checkKernelList", &LinearCombinationKernel::checkKernelList);
    clsLinearCombinationKernel.def("isDeltaFunctionBasis", &LinearCombinationKernel::isDeltaFunctionBasis);
    clsLinearCombinationKernel.def("refactor", &LinearCombinationKernel::refactor);
    clsLinearCombinationKernel.def("toString", &LinearCombinationKernel::toString, "prefix"_a = "");
    clsLinearCombinationKernel.def("isPersistable", &LinearCombinationKernel::isPersistable);

    py::class_<SeparableKernel, std::shared_ptr<SeparableKernel>, Kernel> clsSeparableKernel(
            mod, "SeparableKernel");

    clsSeparableKernel.def(py::init<>());
    // Workaround for NullSpatialFunction and py::arg not playing well with Citizen
    clsSeparableKernel.def(py::init<int, int, SeparableKernel::KernelFunction const &,
                                    SeparableKernel::KernelFunction const &>(),
                           "width"_a, "height"_a, "kernelColFunction"_a, "kernelRowFunction"_a);
    clsSeparableKernel.def(
            py::init<int, int, SeparableKernel::KernelFunction const &,
                     SeparableKernel::KernelFunction const &, Kernel::SpatialFunction const &>(),
            "width"_a, "height"_a, "kernelColFunction"_a, "kernelRowFunction"_a, "spatialFunction"_a);
    clsSeparableKernel.def(py::init<int, int, SeparableKernel::KernelFunction const &,
                                    SeparableKernel::KernelFunction const &,
                                    std::vector<Kernel::SpatialFunctionPtr> const &>(),
                           "width"_a, "height"_a, "kernelColFunction"_a, "kernelRowFunction"_a,
                           "spatialFunctionList"_a);
    clsSeparableKernel.def("clone", &SeparableKernel::clone);
    clsSeparableKernel.def("resized", &SeparableKernel::resized, "width"_a, "height"_a);
    clsSeparableKernel.def("computeVectors", &SeparableKernel::computeVectors);
    clsSeparableKernel.def("getKernelParameter", &SeparableKernel::getKernelParameter);
    clsSeparableKernel.def("getKernelParameters", &SeparableKernel::getKernelParameters);
    clsSeparableKernel.def("getKernelColFunction", &SeparableKernel::getKernelColFunction);
    clsSeparableKernel.def("getKernelRowFunction", &SeparableKernel::getKernelRowFunction);
    clsSeparableKernel.def("toString", &SeparableKernel::toString, "prefix"_a = "");
    clsSeparableKernel.def("computeCache", &SeparableKernel::computeCache);
    clsSeparableKernel.def("getCacheSize", &SeparableKernel::getCacheSize);

    return mod.ptr();
}