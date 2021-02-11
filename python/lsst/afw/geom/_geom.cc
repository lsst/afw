/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11/pybind11.h"
#include "lsst/utils/python.h"

namespace lsst {
namespace afw {
namespace geom {

void wrapEndpoint(lsst::utils::python::WrapperCollection &);
namespace polygon {
void wrapPolygon(lsst::utils::python::WrapperCollection &);
}
void wrapSipApproximation(lsst::utils::python::WrapperCollection &);
void wrapSkyWcs(lsst::utils::python::WrapperCollection &);
void wrapSpan(lsst::utils::python::WrapperCollection &);
void wrapSpanSet(lsst::utils::python::WrapperCollection &);
void wrapTransform(lsst::utils::python::WrapperCollection &);
void wrapTransformFactory(lsst::utils::python::WrapperCollection &);
void wrapWcsUtils(lsst::utils::python::WrapperCollection &);
namespace detail {
void wrapFrameSetUtils(lsst::utils::python::WrapperCollection &);
}

PYBIND11_MODULE(_geom, mod) {
    lsst::utils::python::WrapperCollection wrappers(mod, "lsst.afw.geom");
    wrapEndpoint(wrappers);
    polygon::wrapPolygon(wrappers);
    wrapSipApproximation(wrappers);
    wrapSkyWcs(wrappers);
    wrapSpan(wrappers);
    wrapSpanSet(wrappers);
    wrapTransform(wrappers);
    wrapTransformFactory(wrappers);
    wrappers.makeSubmodule("wcsUtils");
    wrapWcsUtils(wrappers);
    wrappers.finish();
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
