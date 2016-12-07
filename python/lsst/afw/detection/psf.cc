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

#include <memory>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/image/Color.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/pybind11.h"  // for declarePersistableFacade
#include "lsst/afw/detection/Psf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {
    auto const NullPoint = geom::Point2D(std::numeric_limits<double>::quiet_NaN());
}

PYBIND11_PLUGIN(_psf) {
    py::module mod("_psf", "Python wrapper for afw _psf library");

    /* Module level */
    table::io::declarePersistableFacade<Psf>(mod, "Psf");
    py::class_<Psf,
               std::shared_ptr<Psf>,
               daf::base::Persistable,
               afw::table::io::Persistable,
               table::io::PersistableFacade<Psf>,
               daf::base::Citizen> cls(mod, "Psf");

    /* Member types and enums */
    py::enum_<Psf::ImageOwnerEnum>(cls, "ImageOwnerEnum")
        .value("COPY", Psf::ImageOwnerEnum::COPY)
        .value("INTERNAL", Psf::ImageOwnerEnum::INTERNAL)
        .export_values();

    /* Constructors */

    /* Operators */

    /* Members */
    cls.def("clone", &Psf::clone);
    cls.def("computeImage", &Psf::computeImage,
            "position"_a=NullPoint, "color"_a=image::Color(), "owner"_a=Psf::ImageOwnerEnum::COPY);
    cls.def("computeKernelImage", &Psf::computeKernelImage,
            "position"_a=NullPoint, "color"_a=image::Color(), "owner"_a=Psf::ImageOwnerEnum::COPY);
    cls.def("computePeak", &Psf::computePeak,
            "position"_a=NullPoint, "color"_a=image::Color());
    cls.def("computeApertureFlux", &Psf::computeApertureFlux,
            "radius"_a, "position"_a=NullPoint, "color"_a=image::Color());
    cls.def("computeShape", &Psf::computeShape,
            "position"_a=NullPoint, "color"_a=image::Color());
    cls.def("getLocalKernel", &Psf::getLocalKernel,
            "position"_a=NullPoint, "color"_a=image::Color());
    cls.def("getAverageColor", &Psf::getAverageColor);
    cls.def("getAveragePosition", &Psf::getAveragePosition);
    cls.def_static("recenterKernelImage", &Psf::recenterKernelImage,
            "im"_a, "position"_a, "warpAlgorithm"_a="lanczos5", "warpBuffer"_a=5);

    return mod.ptr();
}

}}}  // namespace lsst::afw::detection;
