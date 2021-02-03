/*
 * LSST Data Management System
 * Copyright 2008-2021  AURA/LSST.
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
#include <lsst/utils/python.h>
namespace lsst {
namespace afw {
namespace math {

void wrapApproximate(lsst::utils::python::WrapperCollection &);
void wrapBackground(lsst::utils::python::WrapperCollection &);
void wrapBoundedField(lsst::utils::python::WrapperCollection &);
void wrapChebyshevBoundedField(lsst::utils::python::WrapperCollection &);
void wrapConvolveImage(lsst::utils::python::WrapperCollection &);
void wrapFunction(lsst::utils::python::WrapperCollection &);
void wrapFunctionLibrary(lsst::utils::python::WrapperCollection &);
void wrapGaussianProcess(lsst::utils::python::WrapperCollection &);
void wrapInterpolate(lsst::utils::python::WrapperCollection &);
void wrapKernel(lsst::utils::python::WrapperCollection &);
void wrapLeastSquares(lsst::utils::python::WrapperCollection &);
void wrapMinimize(lsst::utils::python::WrapperCollection &);
void wrapOffsetImage(lsst::utils::python::WrapperCollection &);
void wrapPixelAreaBoundedField(lsst::utils::python::WrapperCollection &);
void wrapProductBoundedField(lsst::utils::python::WrapperCollection &);
void wrapRandom(lsst::utils::python::WrapperCollection &);
void wrapSpatialCell(lsst::utils::python::WrapperCollection &);
void wrapStack(lsst::utils::python::WrapperCollection &);
void wrapStatistics(lsst::utils::python::WrapperCollection &);
void wrapTransformBoundedField(lsst::utils::python::WrapperCollection &);
void wrapWarpExposure(lsst::utils::python::WrapperCollection &);

PYBIND11_MODULE(_math, mod) {
    lsst::utils::python::WrapperCollection wrappers(mod, "lsst.afw.math");
    wrapApproximate(wrappers);
    wrapBackground(wrappers);
    wrapBoundedField(wrappers);
    wrapChebyshevBoundedField(wrappers);
    wrapConvolveImage(wrappers);
    wrapFunction(wrappers);
    wrapFunctionLibrary(wrappers);
    wrapGaussianProcess(wrappers);
    wrapInterpolate(wrappers);
    wrapKernel(wrappers);
    wrapLeastSquares(wrappers);
    wrapMinimize(wrappers);
    wrapOffsetImage(wrappers);
    wrapPixelAreaBoundedField(wrappers);
    wrapProductBoundedField(wrappers);
    wrapRandom(wrappers);
    wrapSpatialCell(wrappers);
    wrapStack(wrappers);
    wrapStatistics(wrappers);
    wrapTransformBoundedField(wrappers);
    wrapWarpExposure(wrappers);
    wrappers.finish();
}
}  // namespace math
}  // namespace afw
}  // namespace lsst