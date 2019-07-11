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

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace detection {

using utils::python::WrapperCollection;

void wrapFootprint(WrapperCollection&);
void wrapFootprintCtrl(WrapperCollection&);
void wrapFootprintMerge(WrapperCollection&);
void wrapFootprintSet(WrapperCollection&);
void wrapGaussianPsf(WrapperCollection&);
void wrapHeavyFootprint(WrapperCollection&);
void wrapPeak(WrapperCollection&);
void wrapPsf(WrapperCollection&);
void wrapThreshold(WrapperCollection&);

PYBIND11_MODULE(_detection, mod) {
    WrapperCollection wrappers(mod, "lsst.afw.detection");
    wrapPsf(wrappers);
    wrapFootprintCtrl(wrappers);
    wrapFootprint(wrappers);
    wrapThreshold(wrappers);
    wrapFootprintSet(wrappers);
    wrapFootprintMerge(wrappers);
    wrapPeak(wrappers);
    wrapGaussianPsf(wrappers);
    wrapHeavyFootprint(wrappers);
    wrappers.finish();
}

}  // namespace detection
}  // namespace afw
}  // namespace lsst
