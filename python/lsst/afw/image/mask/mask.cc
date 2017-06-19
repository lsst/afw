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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/image/LsstImageTypes.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/fits.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

template <typename MaskPixelT>
using PyMask = py::class_<Mask<MaskPixelT>, std::shared_ptr<Mask<MaskPixelT>>, ImageBase<MaskPixelT>>;

template <typename MaskPixelT>
static void declareMask(py::module &mod, std::string const &suffix) {
    PyMask<MaskPixelT> cls(mod, ("Mask" + suffix).c_str());

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "width"_a, "height"_a, "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<unsigned int, unsigned int, MaskPixelT,
                     typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "width"_a, "height"_a, "initialValue"_a,
            "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<geom::Extent2I const &, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "dimensions"_a = geom::Extent2I(), "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<geom::Extent2I const &, MaskPixelT, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "dimensions"_a = geom::Extent2I(), "initialValue"_a,
            "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<geom::Box2I const &, typename Mask<MaskPixelT>::MaskPlaneDict const &>(), "bbox"_a,
            "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<geom::Box2I const &, MaskPixelT, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "bbox"_a, "initialValue"_a, "planeDefs"_a = typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<const Mask<MaskPixelT> &, const bool>(), "src"_a, "deep"_a = false);
    cls.def(py::init<const Mask<MaskPixelT> &, const geom::Box2I &, ImageOrigin const, const bool>(), "src"_a,
            "bbox"_a, "origin"_a = PARENT, "deep"_a = false);
    cls.def(py::init<ndarray::Array<MaskPixelT, 2, 1> const &, bool, geom::Point2I const &>(), "array"_a,
            "deep"_a = false, "xy0"_a = geom::Point2I());
    cls.def(py::init<std::string const &, int, std::shared_ptr<lsst::daf::base::PropertySet>,
                     geom::Box2I const &, ImageOrigin, bool>(),
            "fileName"_a, "hdu"_a = INT_MIN, "metadata"_a = nullptr, "bbox"_a = geom::Box2I(),
            "origin"_a = PARENT, "conformMasks"_a = false);
    cls.def(py::init<fits::MemFileManager &, int, std::shared_ptr<lsst::daf::base::PropertySet>,
                     geom::Box2I const &, ImageOrigin, bool>(),
            "manager"_a, "hdu"_a = INT_MIN, "metadata"_a = nullptr, "bbox"_a = geom::Box2I(),
            "origin"_a = PARENT, "conformMasks"_a = false);
    cls.def(py::init<fits::Fits &, std::shared_ptr<lsst::daf::base::PropertySet>, geom::Box2I const &,
                     ImageOrigin, bool>(),
            "fitsFile"_a, "metadata"_a = nullptr, "bbox"_a = geom::Box2I(), "origin"_a = PARENT,
            "conformMasks"_a = false);

    /* Operators */
    cls.def("__ior__", [](Mask<MaskPixelT> &self, Mask<MaskPixelT> &other) { return self |= other; });
    cls.def("__ior__", [](Mask<MaskPixelT> &self, MaskPixelT const other) { return self |= other; });
    cls.def("__ior__", [](Mask<MaskPixelT> &self, int other) { return self |= other; });
    cls.def("__iand__", [](Mask<MaskPixelT> &self, Mask<MaskPixelT> &other) { return self &= other; });
    cls.def("__iand__", [](Mask<MaskPixelT> &self, MaskPixelT const other) { return self &= other; });
    cls.def("__iand__", [](Mask<MaskPixelT> &self, int other) { return self &= other; });
    cls.def("__ixor__", [](Mask<MaskPixelT> &self, Mask<MaskPixelT> &other) { return self ^= other; });
    cls.def("__ixor__", [](Mask<MaskPixelT> &self, MaskPixelT const other) { return self ^= other; });
    cls.def("__ixor__", [](Mask<MaskPixelT> &self, int other) { return self ^= other; });

    /* Members */
    cls.def("swap", (void (Mask<MaskPixelT>::*)(Mask<MaskPixelT> &)) & Mask<MaskPixelT>::swap);
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(std::string const &,
                                                     std::shared_ptr<lsst::daf::base::PropertySet const>,
                                                     std::string const &) const) &
                                 Mask<MaskPixelT>::writeFits,
            "fileName"_a, "metadata"_a = std::shared_ptr<lsst::daf::base::PropertySet>(), "mode"_a = "w");
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(fits::MemFileManager &,
                                                     std::shared_ptr<lsst::daf::base::PropertySet const>,
                                                     std::string const &) const) &
                                 Mask<MaskPixelT>::writeFits,
            "manager"_a, "metadata"_a = std::shared_ptr<lsst::daf::base::PropertySet>(), "mode"_a = "w");
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(
                                 fits::Fits &, std::shared_ptr<lsst::daf::base::PropertySet const>) const) &
                                 Mask<MaskPixelT>::writeFits,
            "fitsfile"_a, "metadata"_a = std::shared_ptr<lsst::daf::base::PropertySet const>());
    cls.def_static("readFits", (Mask<MaskPixelT>(*)(std::string const &, int))Mask<MaskPixelT>::readFits,
                   "filename"_a, "hdu"_a = INT_MIN);
    cls.def_static("readFits", (Mask<MaskPixelT>(*)(fits::MemFileManager &, int))Mask<MaskPixelT>::readFits,
                   "manager"_a, "hdu"_a = INT_MIN);
    cls.def_static("interpret", Mask<MaskPixelT>::interpret);
    cls.def("getAsString", &Mask<MaskPixelT>::getAsString);
    cls.def("clearAllMaskPlanes", &Mask<MaskPixelT>::clearAllMaskPlanes);
    cls.def("clearMaskPlane", &Mask<MaskPixelT>::clearMaskPlane);
    cls.def("setMaskPlaneValues", &Mask<MaskPixelT>::setMaskPlaneValues);
    cls.def_static("parseMaskPlaneMetadata", Mask<MaskPixelT>::parseMaskPlaneMetadata);
    cls.def_static("clearMaskPlaneDict", Mask<MaskPixelT>::clearMaskPlaneDict);
    cls.def_static("removeMaskPlane", Mask<MaskPixelT>::removeMaskPlane);
    cls.def("removeAndClearMaskPlane", &Mask<MaskPixelT>::removeAndClearMaskPlane, "name"_a,
            "removeFromDefault"_a = false);
    cls.def_static("getMaskPlane", Mask<MaskPixelT>::getMaskPlane);
    cls.def_static("getPlaneBitMask", (MaskPixelT(*)(const std::string &))Mask<MaskPixelT>::getPlaneBitMask);
    cls.def_static("getPlaneBitMask",
                   (MaskPixelT(*)(const std::vector<std::string> &))Mask<MaskPixelT>::getPlaneBitMask);
    cls.def_static("getNumPlanesMax", Mask<MaskPixelT>::getNumPlanesMax);
    cls.def_static("getNumPlanesUsed", Mask<MaskPixelT>::getNumPlanesUsed);
    cls.def("getMaskPlaneDict", &Mask<MaskPixelT>::getMaskPlaneDict);
    cls.def("printMaskPlanes", &Mask<MaskPixelT>::printMaskPlanes);
    cls.def_static("addMaskPlanesToMetadata", Mask<MaskPixelT>::addMaskPlanesToMetadata);
    cls.def("conformMaskPlanes", &Mask<MaskPixelT>::conformMaskPlanes);
    cls.def_static("addMaskPlane", (int (*)(const std::string &))Mask<MaskPixelT>::addMaskPlane);

    /**
     * @internal Set an image to the value val
     */
    cls.def("set", [](Mask<MaskPixelT> &self, MaskPixelT val) { self = val; });

    /**
     * @internal Set pixel (x,y) to val
     */
    cls.def("set", [](Mask<MaskPixelT> &self, int x, int y, double val) {
        self(x, y, image::CheckIndices(true)) = val;
    });

    cls.def("get", [](Mask<MaskPixelT> const &self, int x, int y) -> MaskPixelT {
        return self(x, y, image::CheckIndices(true));
    });

    cls.def("get", [](Mask<MaskPixelT> const &self, int x, int y, int plane) -> bool {
        return self(x, y, plane, image::CheckIndices(true));
    });
}

PYBIND11_PLUGIN(mask) {
    py::module mod("mask");

    py::module::import("lsst.afw.image.image");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareMask<lsst::afw::image::MaskPixel>(mod, "U");

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::image::<anonymous>
