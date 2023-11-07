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
namespace image {
void wrapCalib(lsst::utils::python::WrapperCollection &);
void wrapColor(lsst::utils::python::WrapperCollection &);
void wrapCoaddInputs(lsst::utils::python::WrapperCollection &);
void wrapDefect(lsst::utils::python::WrapperCollection &);
void wrapExposure(lsst::utils::python::WrapperCollection &);
void wrapExposureInfo(lsst::utils::python::WrapperCollection &);
void wrapFilterLabel(lsst::utils::python::WrapperCollection &);
void wrapImagePca(lsst::utils::python::WrapperCollection &);
void wrapImageSummary(lsst::utils::python::WrapperCollection &);
void wrapImageUtils(lsst::utils::python::WrapperCollection &);
void wrapPhotoCalib(lsst::utils::python::WrapperCollection &);
void wrapReaders(lsst::utils::python::WrapperCollection &);
void wrapTransmissionCurve(lsst::utils::python::WrapperCollection &);
void wrapVisitInfo(lsst::utils::python::WrapperCollection &);

PYBIND11_MODULE(_imageLib, mod) {
    lsst::utils::python::WrapperCollection wrappers(mod, "lsst.afw.image");
    wrapCalib(wrappers);
    wrapColor(wrappers);
    wrapCoaddInputs(wrappers);
    wrapDefect(wrappers);
    wrapExposureInfo(wrappers);
    wrapFilterLabel(wrappers);
    wrapImagePca(wrappers);
    wrapImageSummary(wrappers);
    wrapImageUtils(wrappers);
    wrapPhotoCalib(wrappers);
    wrapReaders(wrappers);
    wrapTransmissionCurve(wrappers);
    wrapVisitInfo(wrappers);
    wrappers.finish();
}
}  // namespace image
}  // namespace afw
}  // namespace lsst
