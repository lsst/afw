/*
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
#include "pybind11/stl.h"
#include "lsst/afw/cameraGeom/DetectorCollection.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_MODULE(detectorCollection, mod){
    py::module::import("lsst.afw.cameraGeom.detector");

    py::class_<DetectorCollection, std::shared_ptr<DetectorCollection>> cls(mod, "DetectorCollection");

    cls.def(
        py::init(&DetectorCollection::make),
        "list"_a
    );
    cls.def("getNameMap", &DetectorCollection::getNameMap);
    cls.def("getIdMap", &DetectorCollection::getIdMap);
    cls.def("getFpBBox", &DetectorCollection::getFpBBox);
    cls.def("__len__", &DetectorCollection::size);
    cls.def(
        "get",
        py::overload_cast<std::string const &, std::shared_ptr<Detector>>(
            &DetectorCollection::get, py::const_
        ),
        "name"_a, "default"_a=nullptr
    );
    cls.def(
        "get",
        py::overload_cast<int, std::shared_ptr<Detector>>(
            &DetectorCollection::get, py::const_
        ),
        "id"_a, "default"_a=nullptr
    );
    cls.def(
        "__contains__",
        [](DetectorCollection const & self, std::string const &name) {
            return self.get(name) != nullptr;
        }
    );
    cls.def(
        "__contains__",
        [](DetectorCollection const & self, int id) {
            return self.get(id) != nullptr;
        }
    );
}
} // cameraGeom
} // afw
} // lsst
