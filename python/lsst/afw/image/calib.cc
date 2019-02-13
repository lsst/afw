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

#include <memory>
#include <vector>

#include "ndarray/pybind11.h"

#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/image/Calib.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

template <typename T>
void declareVectorOperations(py::module& mod) {
    typedef ndarray::Array<T, 1> Array;
    typedef ndarray::Array<T const, 1> ConstArray;
    mod.def("abMagFromFlux", (Array(*)(ConstArray const&)) & abMagFromFlux<T>, "flux"_a);
    mod.def("abMagErrFromFluxErr", (Array(*)(ConstArray const&, ConstArray const&)) & abMagErrFromFluxErr<T>,
            "fluxErr"_a, "flux"_a);
    mod.def("fluxFromABMag", (Array(*)(ConstArray const&)) & fluxFromABMag<T>, "mag"_a);
    mod.def("fluxErrFromABMagErr", (Array(*)(ConstArray const&, ConstArray const&)) & fluxErrFromABMagErr<T>,
            "magErr"_a, "mag"_a);
}

PYBIND11_MODULE(calib, mod) {
    /* Module level */
    mod.def("abMagFromFlux", (double (*)(double)) & abMagFromFlux, "flux"_a);
    mod.def("abMagErrFromFluxErr", (double (*)(double, double)) & abMagErrFromFluxErr, "fluxErr"_a, "flux"_a);
    mod.def("fluxFromABMag", (double (*)(double)) & fluxFromABMag, "mag"_a);
    mod.def("fluxErrFromABMagErr", (double (*)(double, double)) & fluxErrFromABMagErr, "magErr"_a, "mag"_a);
    declareVectorOperations<float>(mod);
    declareVectorOperations<double>(mod);
}
}  // namespace
}  // namespace image
}  // namespace afw
}  // namespace lsst
