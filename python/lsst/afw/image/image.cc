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
#include <pybind11/operators.h>
//#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/ImageSlice.h"
#include "lsst/afw/fits.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::image;

template <typename PixelT>
void declareImageBase(py::module & mod, const std::string & suffix) {
    py::class_<ImageBase<PixelT>, std::shared_ptr<ImageBase<PixelT>>, lsst::daf::base::Persistable, lsst::daf::base::Citizen> cls(mod, ("ImageBase" + suffix).c_str());

    cls.def(py::init<const lsst::afw::geom::Extent2I>(),
            "dimensions"_a=lsst::afw::geom::Extent2I());
    cls.def(py::init<const ImageBase<PixelT>&, const bool>(),
            "src"_a, "deep"_a=false);
    cls.def(py::init<const ImageBase<PixelT>&, const lsst::afw::geom::Box2I&, const ImageOrigin, bool>(),
            "src"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);
    cls.def(py::init<typename ImageBase<PixelT>::Array const &, bool, lsst::afw::geom::Point2I const &>(),
            "array"_a, "deep"_a=false, "xy0"_a=lsst::afw::geom::Point2I());

    cls.def("assign", &ImageBase<PixelT>::assign,
        "rhs"_a, "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT, py::is_operator());
    cls.def("getWidth", &ImageBase<PixelT>::getWidth);
    cls.def("getHeight", &ImageBase<PixelT>::getHeight);
    cls.def("getX0", &ImageBase<PixelT>::getX0);
    cls.def("getY0", &ImageBase<PixelT>::getY0);
    cls.def("getXY0", &ImageBase<PixelT>::getXY0);
    cls.def("positionToIndex", &ImageBase<PixelT>::positionToIndex);
    cls.def("indexToPosition", &ImageBase<PixelT>::indexToPosition);
    cls.def("getDimensions", &ImageBase<PixelT>::getDimensions);
    cls.def("getArray", (typename ImageBase<PixelT>::Array (ImageBase<PixelT>::*)()) &ImageBase<PixelT>::getArray);
    cls.def("setXY0", (void (ImageBase<PixelT>::*)(lsst::afw::geom::Point2I const)) &ImageBase<PixelT>::setXY0);
    cls.def("setXY0", (void (ImageBase<PixelT>::*)(int const, int const)) &ImageBase<PixelT>::setXY0);
    cls.def("getBBox", &ImageBase<PixelT>::getBBox);
}

template <typename PixelT>
py::class_<Image<PixelT>> declareImage(py::module & mod, const std::string & suffix) {
    py::class_<Image<PixelT>, std::shared_ptr<Image<PixelT>>, ImageBase<PixelT>> cls(mod, ("Image" + suffix).c_str());

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, PixelT>(),
            "width"_a, "height"_a, "intialValue"_a=0);
    cls.def(py::init<lsst::afw::geom::Extent2I const &, PixelT>(),
            "dimensions"_a=lsst::afw::geom::Extent2I(), "initialValue"_a=0);
    cls.def(py::init<lsst::afw::geom::Box2I const &, PixelT>(),
            "bbox"_a, "initialValue"_a=0);
    cls.def(py::init<Image<PixelT> const &, lsst::afw::geom::Box2I const &, ImageOrigin const, const bool>(),
            "rhs"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);
    cls.def(py::init<std::string const &, int, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin>(),
            "fileName"_a, "hdu"_a=0, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<lsst::afw::fits::MemFileManager &, int, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin>(),
            "manager"_a, "hdu"_a=0, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<lsst::afw::fits::Fits &, PTR(lsst::daf::base::PropertySet), lsst::afw::geom::Box2I const &, ImageOrigin>(),
            "fitsFile"_a, "metadata"_a=PTR(lsst::daf::base::PropertySet)(), "bbox"_a=lsst::afw::geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<ndarray::Array<PixelT,2,1> const &, bool, lsst::afw::geom::Point2I const &>(),
            "array"_a, "deep"_a=false, "xy0"_a=lsst::afw::geom::Point2I());

    /* Operators */
    cls.def(py::self += PixelT());
    cls.def(py::self += Image<PixelT>());
    cls.def("__iadd__", [](Image<PixelT> &lhs, lsst::afw::math::Function2<double> const & rhs) {
                return lhs += rhs;
            }, py::is_operator());
    cls.def(py::self -= PixelT());
    cls.def(py::self -= Image<PixelT>());
    cls.def("__isub__", [](Image<PixelT> &lhs, lsst::afw::math::Function2<double> const & rhs) {
                return lhs -= rhs;
            }, py::is_operator());
    cls.def(py::self *= PixelT());
    cls.def(py::self *= Image<PixelT>());
    cls.def(py::self /= PixelT());
    cls.def(py::self /= Image<PixelT>());

    /* Members */
    cls.def("scaledPlus", &Image<PixelT>::scaledPlus);
    cls.def("scaledMinus", &Image<PixelT>::scaledMinus);
    cls.def("scaledMultiplies", &Image<PixelT>::scaledMultiplies);
    cls.def("scaledDivides", &Image<PixelT>::scaledDivides);

    cls.def("writeFits", (void (Image<PixelT>::*)(std::string const&, CONST_PTR(lsst::daf::base::PropertySet), std::string const&) const) &Image<PixelT>::writeFits,
            "fileName"_a, "metadata"_a=CONST_PTR(lsst::daf::base::PropertySet)(), "mode"_a="w");
    cls.def("writeFits", (void (Image<PixelT>::*)(lsst::afw::fits::MemFileManager &, CONST_PTR(lsst::daf::base::PropertySet), std::string const&) const) &Image<PixelT>::writeFits,
            "manager"_a, "metadata"_a=CONST_PTR(lsst::daf::base::PropertySet)(), "mode"_a="w");
    cls.def("writeFits", (void (Image<PixelT>::*)(lsst::afw::fits::Fits &, CONST_PTR(lsst::daf::base::PropertySet)) const) &Image<PixelT>::writeFits,
            "fitsfile"_a, "metadata"_a=CONST_PTR(lsst::daf::base::PropertySet)());
    cls.def_static("readFits", (Image<PixelT> (*)(std::string const &, int)) Image<PixelT>::readFits,
            "filename"_a, "hdu"_a=0);
    cls.def_static("readFits", (Image<PixelT> (*)(lsst::afw::fits::MemFileManager &, int)) Image<PixelT>::readFits,
            "manager"_a, "hdu"_a=0);
    cls.def("sqrt", &Image<PixelT>::sqrt);

    /* Add-ons for Python interface only */
    cls.def("set", [](Image<PixelT> &img, double val) { img=val; });
    cls.def("set", [](Image<PixelT> &img, int x, int y, double val) { img(x, y, lsst::afw::image::CheckIndices(true))=val; });
    cls.def("get", [](Image<PixelT> &img, int x, int y) { return img(x, y, lsst::afw::image::CheckIndices(true)); });
    cls.def("set0", [](Image<PixelT> &img, int x, int y, double val) { img.set0(x, y, val, lsst::afw::image::CheckIndices(true)); });
    cls.def("set0", [](Image<PixelT> &img, int x, int y) { return img.get0(x, y, lsst::afw::image::CheckIndices(true)); });

    return cls;
}

/* Declare ImageSlice operators separately since they are only instantiated for Image<float> and Image<double> */
template <typename PixelT>
void addImageSliceOperators(py::class_<Image<PixelT>> & cls) {
    cls.def("__add__", [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self + other; }, py::is_operator());
    cls.def("__sub__", [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self - other; }, py::is_operator());
    cls.def("__mul__", [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self * other; }, py::is_operator());
    cls.def("__truediv__", [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self / other; }, py::is_operator());
    cls.def("__iadd__", [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self += other; return self; }, py::is_operator());
    cls.def("__isub__", [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self -= other; return self; }, py::is_operator());
    cls.def("__imul__", [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self *= other; return self; }, py::is_operator());
    cls.def("__itruediv__", [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self /= other; return self; }, py::is_operator());
}

PYBIND11_PLUGIN(_image) {
    py::module mod("_image", "Python wrapper for afw _image library");

    if (_import_array() < 0) { 
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); 
        return nullptr; 
    } 

    py::enum_<ImageOrigin>(mod, "ImageOrigin")
        .value("PARENT", ImageOrigin::PARENT)
        .value("LOCAL", ImageOrigin::LOCAL)
        .export_values();

    declareImageBase<int>(mod, "I");
    declareImageBase<float>(mod, "F");
    declareImageBase<double>(mod, "D");
    declareImageBase<std::uint16_t>(mod, "U");
    declareImageBase<std::uint64_t>(mod, "L");

    declareImage<int>(mod, "I");
    auto clsImageF = declareImage<float>(mod, "F");
    addImageSliceOperators<float>(clsImageF);
    auto clsImageD = declareImage<double>(mod, "D");
    addImageSliceOperators<double>(clsImageD);
    declareImage<std::uint16_t>(mod, "U");
    declareImage<std::uint64_t>(mod, "L");

    return mod.ptr();
}