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
//#include <pybind11/stl.h>

#include "lsst/afw/table/io/python.h"  // for declarePersistableFacade
#include "lsst/afw/detection/GaussianPsf.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

PYBIND11_PLUGIN(_gaussianPsf) {
    py::module mod("_gaussianPsf", "Python wrapper for afw _gaussianPsf library");

    table::io::python::declarePersistableFacade<GaussianPsf>(mod, "GaussianPsf");

    py::class_<GaussianPsf, std::shared_ptr<GaussianPsf>, afw::table::io::PersistableFacade<GaussianPsf>, Psf>
            clsGaussianPsf(mod, "GaussianPsf");

    /* Constructors */
    clsGaussianPsf.def(py::init<int, int, double>(), "width"_a, "height"_a, "sigma"_a);
    clsGaussianPsf.def(py::init<geom::Extent2I const &, double>(), "dimensions"_a, "sigma"_a);

    /* Members */
    clsGaussianPsf.def("clone", &GaussianPsf::clone);
    clsGaussianPsf.def("getDimensions", &GaussianPsf::getDimensions);
    clsGaussianPsf.def("getSigma", &GaussianPsf::getSigma);
    clsGaussianPsf.def("isPersistable", &GaussianPsf::isPersistable);

    return mod.ptr();
}
}
}
}  // lsst::afw::detection