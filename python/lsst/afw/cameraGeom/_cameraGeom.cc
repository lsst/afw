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

#include <pybind11/pybind11.h>
#include <lsst/cpputils/python.h>
namespace lsst {
namespace afw {
namespace cameraGeom {

void wrapAmplifier(lsst::cpputils::python::WrapperCollection &);
void wrapCamera(lsst::cpputils::python::WrapperCollection &);
void wrapCameraSys(lsst::cpputils::python::WrapperCollection &);
void wrapDetector(lsst::cpputils::python::WrapperCollection &);
void wrapDetectorCollection(lsst::cpputils::python::WrapperCollection &);
void wrapOrientation(lsst::cpputils::python::WrapperCollection &);
void wrapTransformMap(lsst::cpputils::python::WrapperCollection &);

PYBIND11_MODULE(_cameraGeom, mod) {
    lsst::cpputils::python::WrapperCollection wrappers(mod, "lsst.afw.cameraGeom");
    wrapAmplifier(wrappers);
    wrapDetectorCollection(wrappers);
    wrapDetector(wrappers);
    wrapCamera(wrappers);
    wrapCameraSys(wrappers);
    wrapOrientation(wrappers);
    wrapTransformMap(wrappers);
    wrappers.finish();
}
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
