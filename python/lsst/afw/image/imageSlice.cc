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

#include "lsst/afw/image/ImageSlice.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

template <typename PixelT>
static void declareImageSlice(py::module &mod, std::string const &suffix) {
    using Class = ImageSlice<PixelT>;

    py::module::import("lsst.afw.image.image");

    py::class_<Class, std::shared_ptr<Class>, Image<PixelT>> cls(mod, ("ImageSlice" + suffix).c_str());

    cls.def(py::init<Image<PixelT> const &>(), "img"_a);

    py::enum_<typename Class::ImageSliceType>(cls, "ImageSliceType")
            .value("ROW", Class::ImageSliceType::ROW)
            .value("COLUMN", Class::ImageSliceType::COLUMN)
            .export_values();

    cls.def("getImageSliceType", &Class::getImageSliceType);

    cls.def("__add__", [](ImageSlice<PixelT> &self, Image<PixelT> const &other) { return self + other; },
            py::is_operator());
    cls.def("__mul__", [](ImageSlice<PixelT> &self, Image<PixelT> const &other) { return self * other; },
            py::is_operator());
    cls.def("__iadd__", [](ImageSlice<PixelT> &self, Image<PixelT> const &other) {
        self += other;
        return self;
    });
    cls.def("__imul__", [](ImageSlice<PixelT> &self, Image<PixelT> const &other) {
        self *= other;
        return self;
    });

    cls.def("__add__",
            [](Image<PixelT> const &self, ImageSlice<PixelT> const &other) { return self + other; },
            py::is_operator());
    cls.def("__sub__",
            [](Image<PixelT> const &self, ImageSlice<PixelT> const &other) { return self - other; },
            py::is_operator());
    cls.def("__mul__",
            [](Image<PixelT> const &self, ImageSlice<PixelT> const &other) { return self * other; },
            py::is_operator());
    cls.def("__truediv__",
            [](Image<PixelT> const &self, ImageSlice<PixelT> const &other) { return self / other; },
            py::is_operator());
    cls.def("__iadd__", [](Image<PixelT> &self, ImageSlice<PixelT> const &other) {
        self += other;
        return self;
    });
    cls.def("__isub__", [](Image<PixelT> &self, ImageSlice<PixelT> const &other) {
        self -= other;
        return self;
    });
    cls.def("__imul__", [](Image<PixelT> &self, ImageSlice<PixelT> const &other) {
        self *= other;
        return self;
    });
    cls.def("__itruediv__", [](Image<PixelT> &self, ImageSlice<PixelT> const &other) {
        self /= other;
        return self;
    });
}

PYBIND11_MODULE(imageSlice, mod) {
    py::module::import("lsst.afw.image.image");

    declareImageSlice<float>(mod, "F");
    declareImageSlice<double>(mod, "D");
}
}
}
}
}  // namespace lsst::afw::image::<anonymous>