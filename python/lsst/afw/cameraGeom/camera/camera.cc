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
#include "lsst/afw/cameraGeom/Camera.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_MODULE(camera, mod){
    py::module::import("lsst.afw.cameraGeom.detectorCollection");
    py::module::import("lsst.afw.cameraGeom.transformMap");

    py::class_<Camera, DetectorCollection, std::shared_ptr<Camera>> cls(mod, "Camera");

    cls.def(py::init(&Camera::make), "name"_a, "detectorList"_a, "transformMap"_a, "pupilFactoryName"_a);
    // Python-only constructor that takes a PupilFactory type object for
    // backwards compatibility.
    cls.def(
        py::init(
            [](
                std::string const & name,
                Camera::DetectorList const & detectorList,
                std::shared_ptr<TransformMap> transformMap,
                py::object pupilFactoryClass
            ) {
                std::string pupilFactoryName = py::str("{}.{}").format(
                    pupilFactoryClass.attr("__module__"),
                    pupilFactoryClass.attr("__name__")
                );
                return Camera::make(name, detectorList, transformMap, pupilFactoryName);
            }
        )
    );
    cls.def("getName", &Camera::getName);
    cls.def("getPupilFactoryName", &Camera::getPupilFactoryName);
    cls.def("findDetectors", &Camera::findDetectors, "point"_a, "cameraSys"_a);
    cls.def("findDetectorsList", &Camera::findDetectorsList, "pointList"_a, "cameraSys"_a);
    cls.def("getTransformMap", &Camera::getTransformMap);
    // transform methods are wrapped with lambdas that translate exceptions for backwards compatibility
    cls.def(
        "getTransform",
        [](Camera const & self, CameraSys const & fromSys, CameraSys const & toSys) {
            try {
                return self.getTransform(fromSys, toSys);
            } catch (pex::exceptions::NotFoundError & err) {
                PyErr_SetString(PyExc_KeyError, err.what());
                throw py::error_already_set();
            }
        },
        "fromSys"_a, "toSys"_a
    );
    cls.def(
        "transform",
        [](
            Camera const & self,
            lsst::geom::Point2D const & point,
            CameraSys const & fromSys,
            CameraSys const & toSys
        ) {
            try {
                return self.transform(point, fromSys, toSys);
            } catch (pex::exceptions::NotFoundError & err) {
                PyErr_SetString(PyExc_KeyError, err.what());
                throw py::error_already_set();
            }
        },
        "point"_a, "fromSys"_a, "toSys"_a
    );
    cls.def(
        "transform",
        [](
            Camera const & self,
            std::vector<lsst::geom::Point2D> const & points,
            CameraSys const & fromSys,
            CameraSys const & toSys
        ) {
            try {
                return self.transform(points, fromSys, toSys);
            } catch (pex::exceptions::NotFoundError & err) {
                PyErr_SetString(PyExc_KeyError, err.what());
                throw py::error_already_set();
            }
        },
        "points"_a, "fromSys"_a, "toSys"_a
    );
}

} // cameraGeom
} // afw
} // lsst
