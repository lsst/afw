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

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/image/Mask.h"
#include "lsst/afw/fits.h"

namespace py = pybind11;

using namespace lsst::afw::image;
using namespace pybind11::literals;

template <typename MaskPixelT>
void declareMask(py::module & mod, std::string const & suffix) {
    py::class_<Mask<MaskPixelT>, std::shared_ptr<Mask<MaskPixelT>>, ImageBase<MaskPixelT>> cls(mod, ("Mask" + suffix).c_str());

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "width"_a, "height"_a, "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<unsigned int, unsigned int, MaskPixelT, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "width"_a, "height"_a, "initialValue"_a, "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<lsst::afw::geom::Extent2I const &, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "dimensions"_a=lsst::afw::geom::Extent2I(), "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<lsst::afw::geom::Extent2I const &, MaskPixelT, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "dimensions"_a=lsst::afw::geom::Extent2I(), "initialValue"_a, "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<lsst::afw::geom::Box2I const &, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "bbox"_a, "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<lsst::afw::geom::Box2I const &, MaskPixelT, typename Mask<MaskPixelT>::MaskPlaneDict const &>(),
            "bbox"_a, "initialValue"_a, "planeDefs"_a=typename Mask<MaskPixelT>::MaskPlaneDict());
    cls.def(py::init<std::string const &, int, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin, bool>(),
            "fileName"_a, "hdu"_a=0, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT, "conformMasks"_a=false);
    cls.def(py::init<lsst::afw::fits::MemFileManager &, int, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin, bool>(),
            "manager"_a, "hdu"_a=0, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT, "conformMasks"_a=false);
    cls.def(py::init<lsst::afw::fits::Fits &, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin, bool>(),
            "fitsFile"_a, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT, "conformMasks"_a=false);
    cls.def(py::init<const Mask<MaskPixelT>&, const bool>(),
            "src"_a, "deep"_a=false);
    cls.def(py::init<const Mask<MaskPixelT>&, const lsst::afw::geom::Box2I &, ImageOrigin const, const bool>(),
            "src"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);
    cls.def(py::init<ndarray::Array<MaskPixelT,2,1> const &, bool, lsst::afw::geom::Point2I const &>(),
            "array"_a, "deep"_a=false, "xy0"_a=lsst::afw::geom::Point2I());

    /* Operators */
    cls.def("__ior__", [](Mask<MaskPixelT> & self, Mask<MaskPixelT> & other) { return self |= other; }, py::is_operator());
    cls.def("__ior__", [](Mask<MaskPixelT> & self, MaskPixelT const other) { return self |= other; }, py::is_operator());
    cls.def("__ior__", [](Mask<MaskPixelT> & self, int other) { return self |= other; }, py::is_operator());
    cls.def("__iand__", [](Mask<MaskPixelT> & self, Mask<MaskPixelT> & other) { return self &= other; }, py::is_operator());
    cls.def("__iand__", [](Mask<MaskPixelT> & self, MaskPixelT const other) { return self &= other; }, py::is_operator());
    cls.def("__iand__", [](Mask<MaskPixelT> & self, int other) { return self &= other; }, py::is_operator());
    cls.def("__ixor__", [](Mask<MaskPixelT> & self, Mask<MaskPixelT> & other) { return self ^= other; }, py::is_operator());
    cls.def("__ixor__", [](Mask<MaskPixelT> & self, MaskPixelT const other) { return self ^= other; }, py::is_operator());
    cls.def("__ixor__", [](Mask<MaskPixelT> & self, int other) { return self ^= other; }, py::is_operator());

    /* Members */
    cls.def("swap", (void (Mask<MaskPixelT>::*)(Mask<MaskPixelT>&)) &Mask<MaskPixelT>::swap);
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(std::string const &, CONST_PTR(lsst::daf::base::PropertySet), std::string const &) const) &Mask<MaskPixelT>::writeFits,
            "fileName"_a, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "mode"_a="w");
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(lsst::afw::fits::MemFileManager &, CONST_PTR(lsst::daf::base::PropertySet), std::string const &) const) &Mask<MaskPixelT>::writeFits,
            "manager"_a, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "mode"_a="w");
    cls.def("writeFits", (void (Mask<MaskPixelT>::*)(lsst::afw::fits::Fits &, CONST_PTR(lsst::daf::base::PropertySet)) const) &Mask<MaskPixelT>::writeFits,
            "fitsfile"_a, "metadata"_a=CONST_PTR(lsst::daf::base::PropertySet)());
    cls.def_static("readFits", (Mask<MaskPixelT> (*)(std::string const &, int)) Mask<MaskPixelT>::readFits,
                   "filename"_a, "hdu"_a=0);
    cls.def_static("readFits", (Mask<MaskPixelT> (*)(lsst::afw::fits::MemFileManager &, int)) Mask<MaskPixelT>::readFits,
                   "manager"_a, "hdu"_a=0);
    cls.def_static("interpret", Mask<MaskPixelT>::interpret);
    cls.def("getAsString", &Mask<MaskPixelT>::getAsString);
    cls.def("clearAllMaskPlanes", &Mask<MaskPixelT>::clearAllMaskPlanes);
    cls.def("clearMaskPlane", &Mask<MaskPixelT>::clearMaskPlane);
    cls.def("setMaskPlaneValues", &Mask<MaskPixelT>::setMaskPlaneValues);
    cls.def_static("parseMaskPlaneMetadata", Mask<MaskPixelT>::parseMaskPlaneMetadata);
    cls.def_static("clearMaskPlaneDict", Mask<MaskPixelT>::clearMaskPlaneDict);
    cls.def_static("removeMaskPlane", Mask<MaskPixelT>::removeMaskPlane);
    cls.def("removeAndClearMaskPlane", &Mask<MaskPixelT>::removeAndClearMaskPlane,
            "name"_a, "removeFromDefault"_a=false);
    cls.def_static("getMaskPlane", Mask<MaskPixelT>::getMaskPlane);
    cls.def_static("getPlaneBitMask", (MaskPixelT (*)(const std::string&)) Mask<MaskPixelT>::getPlaneBitMask);
    cls.def_static("getPlaneBitMask", (MaskPixelT (*)(const std::vector<std::string> &)) Mask<MaskPixelT>::getPlaneBitMask);
    cls.def_static("getNumPlanesMax", Mask<MaskPixelT>::getNumPlanesMax);
    cls.def_static("getNumPlanesUsed", Mask<MaskPixelT>::getNumPlanesUsed);
    cls.def("getMaskPlaneDict", &Mask<MaskPixelT>::getMaskPlaneDict);
    cls.def("printMaskPlanes", &Mask<MaskPixelT>::printMaskPlanes);
    cls.def_static("addMaskPlanesToMetadata", Mask<MaskPixelT>::addMaskPlanesToMetadata);
    cls.def("conformMaskPlanes", &Mask<MaskPixelT>::conformMaskPlanes);
    cls.def_static("addMaskPlane", (int (*)(const std::string &)) Mask<MaskPixelT>::addMaskPlane);

    /**
     * Set an image to the value val
     */
    cls.def("set", [](Mask<MaskPixelT> & self, MaskPixelT val) {
        self = val;
    });

    /**
     * Set pixel (x,y) to val
     */
    cls.def("set", [](Mask<MaskPixelT> & self, int x, int y, double val) {
        self(x, y, lsst::afw::image::CheckIndices(true)) = val;
    });

    cls.def("get", [](Mask<MaskPixelT> const & self, int x, int y) -> MaskPixelT {
        return self(x, y, lsst::afw::image::CheckIndices(true));
    });

    cls.def("get", [](Mask<MaskPixelT> const & self, int x, int y, int plane) -> bool {
        return self(x, y, plane, lsst::afw::image::CheckIndices(true));
    });
}

PYBIND11_PLUGIN(_mask) {
    py::module mod("_mask", "Python wrapper for afw _mask library");

    if (_import_array() < 0) { 
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); 
        return nullptr; 
    } 

    declareMask<std::uint16_t>(mod, "U");

    return mod.ptr();
}