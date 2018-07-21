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

#include "lsst/afw/image/ImagePca.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace image {

namespace {

template <typename ImageT>
static void declareImagePca(py::module &mod, std::string const &suffix) {
    py::class_<ImagePca<ImageT>, std::shared_ptr<ImagePca<ImageT>>> cls(mod, ("ImagePca" + suffix).c_str());

    cls.def(py::init<bool>(), "constantWeight"_a = true);

    cls.def("addImage", &ImagePca<ImageT>::addImage, "img"_a, "flux"_a = 0.0);
    cls.def("getImageList", &ImagePca<ImageT>::getImageList);
    cls.def("getDimensions", &ImagePca<ImageT>::getDimensions);
    cls.def("getMean", &ImagePca<ImageT>::getMean);
    cls.def("analyze", &ImagePca<ImageT>::analyze);
    cls.def("updateBadPixels", &ImagePca<ImageT>::updateBadPixels);
    cls.def("getEigenValues", &ImagePca<ImageT>::getEigenValues);
    cls.def("getEigenImages", &ImagePca<ImageT>::getEigenImages);
}

template <typename Image1T, typename Image2T>
static void declareInnerProduct(py::module &mod) {
    mod.def("innerProduct",
            (double (*)(Image1T const &, Image2T const &, int const))innerProduct<Image1T, Image2T>, "lhs"_a,
            "rhs"_a, "border"_a = 0);
}

}  // namespace

PYBIND11_MODULE(imagePca, mod) {
    declareImagePca<Image<int>>(mod, "I");
    declareImagePca<Image<float>>(mod, "F");
    declareImagePca<Image<double>>(mod, "D");
    declareImagePca<Image<std::uint16_t>>(mod, "U");
    declareImagePca<Image<std::uint64_t>>(mod, "L");
    declareImagePca<MaskedImage<int>>(mod, "MI");
    declareImagePca<MaskedImage<float>>(mod, "MF");
    declareImagePca<MaskedImage<double>>(mod, "MD");
    declareImagePca<MaskedImage<std::uint16_t>>(mod, "MU");
    declareImagePca<MaskedImage<std::uint64_t>>(mod, "ML");

    declareInnerProduct<Image<int>, Image<int>>(mod);
    declareInnerProduct<Image<float>, Image<float>>(mod);
    declareInnerProduct<Image<double>, Image<double>>(mod);
    declareInnerProduct<Image<std::uint16_t>, Image<std::uint16_t>>(mod);
    declareInnerProduct<Image<std::uint64_t>, Image<std::uint64_t>>(mod);

    declareInnerProduct<Image<float>, Image<double>>(mod);
    declareInnerProduct<Image<double>, Image<float>>(mod);
}
}
}
}  // lsst::afw::image