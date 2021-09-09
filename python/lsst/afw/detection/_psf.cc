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

#include <memory>

#include <pybind11/pybind11.h>

#include "lsst/utils/python.h"
#include "lsst/utils/python/PySharedPtr.h"

#include "lsst/geom/Point.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/detection/python.h"

namespace py = pybind11;
using namespace pybind11::literals;

using lsst::utils::python::PySharedPtr;

namespace lsst {
namespace afw {
namespace detection {


void wrapPsf(utils::python::WrapperCollection& wrappers) {
    wrappers.addInheritanceDependency("lsst.afw.typehandling");
    wrappers.addSignatureDependency("lsst.afw.geom.ellipses");
    wrappers.addSignatureDependency("lsst.afw.image");
    wrappers.addSignatureDependency("lsst.afw.fits");

    auto clsPsf = wrappers.wrapType(
            py::class_<Psf, PySharedPtr<Psf>, typehandling::Storable, PsfTrampoline<>>(
                wrappers.module, "Psf"
            ),
            [](auto& mod, auto& cls) {
                table::io::python::addPersistableMethods<Psf>(cls);
                cls.def(py::init<bool, size_t>(), "isFixed"_a=false, "capacity"_a=100);  // Constructor for pure-Python subclasses
                cls.def("clone", &Psf::clone);
                cls.def("resized", &Psf::resized, "width"_a, "height"_a);

                // Position-required overloads. Can (likely) remove overload_cast<> once deprecation period for
                // default position argument ends.
                cls.def("computeImage",
                        py::overload_cast<lsst::geom::Point2D, image::Color, Psf::ImageOwnerEnum>(&Psf::computeImage, py::const_),
                        "position"_a,
                        "color"_a = image::Color(),
                        "owner"_a = Psf::ImageOwnerEnum::COPY
                );
                cls.def("computeKernelImage",
                        py::overload_cast<lsst::geom::Point2D, image::Color, Psf::ImageOwnerEnum>(&Psf::computeKernelImage, py::const_),
                        "position"_a,
                        "color"_a = image::Color(),
                        "owner"_a = Psf::ImageOwnerEnum::COPY
                );
                cls.def("computePeak",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::computePeak, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("computeApertureFlux",
                        py::overload_cast<double, lsst::geom::Point2D, image::Color>(&Psf::computeApertureFlux, py::const_),
                        "radius"_a,
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("computeShape",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::computeShape, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("computeBBox",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::computeBBox, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("computeImageBBox",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::computeImageBBox, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("computeKernelBBox",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::computeKernelBBox, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );
                cls.def("getLocalKernel",
                        py::overload_cast<lsst::geom::Point2D, image::Color>(&Psf::getLocalKernel, py::const_),
                        "position"_a,
                        "color"_a = image::Color()
                );

                // Deprecated default position argument overloads.
                cls.def("computeImage",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeImage();
                        }
                );
                cls.def("computeKernelImage",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeKernelImage();
                        }
                );
                cls.def("computePeak",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computePeak();
                        }
                );
                cls.def("computeApertureFlux",
                        [](const Psf& psf, double radius) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeApertureFlux(radius);
                        },
                        "radius"_a
                );
                cls.def("computeShape",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeShape();
                        }
                );
                cls.def("computeBBox",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeBBox();
                        }
                );
                cls.def("computeImageBBox",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeImageBBox();
                        }
                );
                cls.def("computeKernelBBox",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.computeKernelBBox();
                        }
                );
                cls.def("getLocalKernel",
                        [](const Psf& psf) {
                            py::gil_scoped_acquire gil;
                            auto warnings = py::module::import("warnings");
                            auto FutureWarning = py::handle(PyEval_GetBuiltins())["FutureWarning"];
                            warnings.attr("warn")(
                                "Default position argument overload is deprecated and will be "
                                "removed in version 24.0.  Please explicitly specify a position.",
                                "category"_a=FutureWarning
                            );
                            return psf.getLocalKernel();
                        }
                );
                // End deprecated default position argument overloads.

                cls.def("getAverageColor", &Psf::getAverageColor);
                cls.def("getAveragePosition", &Psf::getAveragePosition);
                cls.def_static("recenterKernelImage", &Psf::recenterKernelImage, "im"_a, "position"_a,
                               "warpAlgorithm"_a = "lanczos5", "warpBuffer"_a = 5);
                cls.def("getCacheCapacity", &Psf::getCacheCapacity);
                cls.def("setCacheCapacity", &Psf::setCacheCapacity);
            }
    );

    wrappers.wrapType(py::enum_<Psf::ImageOwnerEnum>(clsPsf, "ImageOwnerEnum"), [](auto& mod, auto& enm) {
        enm.value("COPY", Psf::ImageOwnerEnum::COPY);
        enm.value("INTERNAL", Psf::ImageOwnerEnum::INTERNAL);
        enm.export_values();
    });
}

}  // namespace detection
}  // namespace afw
}  // namespace lsst
