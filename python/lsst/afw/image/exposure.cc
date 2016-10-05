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
//#include <pybind11/stl.h>

#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/detection/Psf.h"

namespace py = pybind11;

using namespace lsst::afw::image;

template <typename PixelT>
void declareExposure(py::module & mod, const std::string & suffix) {
    py::class_<Exposure<PixelT>> cls(mod, ("Exposure" + suffix).c_str());

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, CONST_PTR(Wcs)>(),
            py::arg("width"), py::arg("height"), py::arg("wcs")=CONST_PTR(Wcs)());
    cls.def(py::init<lsst::afw::geom::Extent2I const &, CONST_PTR(Wcs)>(),
            py::arg("dimensions")=lsst::afw::geom::Extent2I(), py::arg("wcs")=CONST_PTR(Wcs)());
    cls.def(py::init<lsst::afw::geom::Box2I const &, CONST_PTR(Wcs)>(),
            py::arg("bbox"), py::arg("wcs")=CONST_PTR(Wcs)());
    cls.def(py::init<typename Exposure<PixelT>::MaskedImageT &, CONST_PTR(Wcs)>(),
            py::arg("maskedImage"), py::arg("wcs")=CONST_PTR(Wcs)());
    cls.def(py::init<std::string const &, lsst::afw::geom::Box2I const&, ImageOrigin, bool>(),
            py::arg("fileName"), py::arg("bbox")=lsst::afw::geom::Box2I(), py::arg("origin")=PARENT, py::arg("conformMasks")=false);
//    cls.def(py::init<lsst::afw::fits::MemFileManager &, lsst::afw::geom::Box2I const &, ImageOrigin, bool>(),
//            py::arg("manager"), py::arg("bbox")=lsst::afw::geom::Box2I(), py::arg("origin")=PARENT, py::arg("conformMasks")=false);
//    cls.def(py::init<lsst::afw::fits::Fits &, lsst::afw::geom::Box2I const &, ImageOrigin, bool>(),
//            py::arg("fitsFile"), py::arg("bbox")=lsst::afw::geom::Box2I(), py::arg("origin")=PARENT, py::arg("conformMasks")=false);
    cls.def(py::init<Exposure<PixelT> const &, bool>(),
            py::arg("other"), py::arg("deep")=false);
//    cls.def(py::init<Exposure<PixelT> const &, lsst::afw::geom::Box2I const&, ImageOrigin, bool>(),
//            py::arg("other"), py::arg("bbox"), py::arg("origin")=PARENT, py::arg("deep")=false);


    /* Members */
    cls.def("getMaskedImage", (typename Exposure<PixelT>::MaskedImageT (Exposure<PixelT>::*)()) &Exposure<PixelT>::getMaskedImage);
    cls.def("getWcs", (PTR(Wcs) (Exposure<PixelT>::*)()) &Exposure<PixelT>::getWcs);
//    cls.def("getDetector", &Exposure<PixelT>::getDetector);
    cls.def("getFilter", &Exposure<PixelT>::getFilter);
    cls.def("getMetadata", &Exposure<PixelT>::getMetadata);
    cls.def("setMetadata", &Exposure<PixelT>::setMetadata);
    cls.def("getWidth", &Exposure<PixelT>::getWidth);
    cls.def("getHeight", &Exposure<PixelT>::getHeight);
    cls.def("getDimensions", &Exposure<PixelT>::getDimensions);
    cls.def("getX0", &Exposure<PixelT>::getX0);
    cls.def("getY0", &Exposure<PixelT>::getY0);
    cls.def("getXY0", &Exposure<PixelT>::getXY0);
    cls.def("getBBox", &Exposure<PixelT>::getBBox);
    cls.def("setXY0", &Exposure<PixelT>::setXY0);
    cls.def("setMaskedImage", &Exposure<PixelT>::setMaskedImage);
    cls.def("setWcs", &Exposure<PixelT>::setWcs);
//    cls.def("setDetector", &Exposure<PixelT>::setDetector);
    cls.def("setFilter", &Exposure<PixelT>::setFilter);
//    cls.def("setCalib", &Exposure<PixelT>::setCalib);
//    cls.def("getCalib", &Exposure<PixelT>::getCalib);
    cls.def("setPsf", &Exposure<PixelT>::setPsf);
    cls.def("getPsf", (PTR(lsst::afw::detection::Psf) (Exposure<PixelT>::*)()) &Exposure<PixelT>::getPsf);
    cls.def("hasPsf", &Exposure<PixelT>::hasPsf);
    cls.def("hasWcs", &Exposure<PixelT>::hasWcs);
    cls.def("getInfo", (PTR(ExposureInfo) (Exposure<PixelT>::*)())&Exposure<PixelT>::getInfo);

    cls.def("writeFits", (void (Exposure<PixelT>::*)(std::string const &) const) &Exposure<PixelT>::writeFits);
    cls.def("writeFits", (void (Exposure<PixelT>::*)(lsst::afw::fits::MemFileManager &) const) &Exposure<PixelT>::writeFits);
    cls.def("writeFits", (void (Exposure<PixelT>::*)(lsst::afw::fits::Fits &) const) &Exposure<PixelT>::writeFits);

    cls.def_static("readFits", (Exposure<PixelT> (*)(std::string const &)) Exposure<PixelT>::readFits);
    cls.def_static("readFits", (Exposure<PixelT> (*)(lsst::afw::fits::MemFileManager &)) Exposure<PixelT>::readFits);
}

PYBIND11_PLUGIN(_exposure) {
    py::module mod("_exposure", "Python wrapper for afw _exposure library");

    declareExposure<int>(mod, "I");
    declareExposure<float>(mod, "F");
    declareExposure<double>(mod, "D");
    declareExposure<std::uint16_t>(mod, "U");
    declareExposure<std::uint64_t>(mod, "L");
    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}