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

#include "lsst/afw/fits.h"
#include "lsst/afw/image/MaskedImage.h"

namespace py = pybind11;

using namespace lsst::afw::image;
using namespace pybind11::literals;

template <typename ImagePixelT> // addtional template types do not seem to be needed
void declareMaskedImage(py::module & mod, const std::string & suffix) {
    using MI = MaskedImage<ImagePixelT>;
    using lsst::daf::base::PropertySet;
    using lsst::afw::geom::Box2I;
    using lsst::afw::geom::Extent2I;

    py::class_<MI, std::shared_ptr<MI>, lsst::daf::base::Persistable, lsst::daf::base::Citizen> cls(mod, ("MaskedImage" + suffix).c_str());

    /* Member types and enums */

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, typename MI::MaskPlaneDict const&>(),
            "width"_a, "height"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<Extent2I, typename MI::MaskPlaneDict const&>(),
            "dimensions"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<typename MI::ImagePtr, typename MI::MaskPtr, typename MI::VariancePtr>(),
            "image"_a, "mask"_a=typename MI::MaskPtr(), "variance"_a=typename MI::VariancePtr());
    cls.def(py::init<Box2I const &, typename MI::MaskPlaneDict const&>(),
            "bbox"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<std::string const &,
                     PTR(PropertySet),
                     Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     PTR(PropertySet),
                     PTR(PropertySet),
                     PTR(PropertySet)>(),
                     "fileName"_a,
                     "metadata"_a=PTR(PropertySet)(),
                     "bbox"_a=Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=PTR(PropertySet)(),
                     "maskMetadata"_a=PTR(PropertySet)(),
                     "varianceMetadata"_a=PTR(PropertySet)());
    cls.def(py::init<lsst::afw::fits::MemFileManager &,
                     PTR(PropertySet),
                     Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     PTR(PropertySet),
                     PTR(PropertySet),
                     PTR(PropertySet)>(),
                     "manager"_a,
                     "metadata"_a=PTR(PropertySet)(),
                     "bbox"_a=Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=PTR(PropertySet)(),
                     "maskMetadata"_a=PTR(PropertySet)(),
                     "varianceMetadata"_a=PTR(PropertySet)());
    cls.def(py::init<lsst::afw::fits::Fits &,
                     PTR(PropertySet),
                     Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     PTR(PropertySet),
                     PTR(PropertySet),
                     PTR(PropertySet)>(),
                     "fitsfile"_a,
                     "metadata"_a=PTR(PropertySet)(),
                     "bbox"_a=Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=PTR(PropertySet)(),
                     "maskMetadata"_a=PTR(PropertySet)(),
                     "varianceMetadata"_a=PTR(PropertySet)());
    cls.def(py::init<MI const &, bool>(),
            "rhs"_a, "deep"_a=false);
    cls.def(py::init<MI const &, Box2I const &, ImageOrigin, bool>(),
            "rhs"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);

    /* Operators */
    cls.def("swap", &MI::swap);
    cls.def("assign", &MI::assign,
            "rhs"_a, "bbox"_a=Box2I(), "origin"_a=PARENT, py::is_operator());

    cls.def("__ilshift__", &MI::operator<<=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(ImagePixelT const)) &MI::operator+=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(MI const &)) &MI::operator+=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator+=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(lsst::afw::math::Function2<double> const &)) &MI::operator+=, py::is_operator());
    cls.def("scaledPlus", &MI::scaledPlus);
    cls.def("__isub__", (MI& (MI::*)(ImagePixelT const)) &MI::operator-=, py::is_operator());
    cls.def("__isub__", (MI& (MI::*)(MI const &)) &MI::operator-=, py::is_operator());
    cls.def("__isub__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator-=, py::is_operator());
    cls.def("__isub__", (MI& (MI::*)(lsst::afw::math::Function2<double> const &)) &MI::operator-=, py::is_operator());
    cls.def("scaledMinus", &MI::scaledMinus);
    cls.def("__imul__", (MI& (MI::*)(ImagePixelT const)) &MI::operator*=, py::is_operator());
    cls.def("__imul__", (MI& (MI::*)(MI const &)) &MI::operator*=, py::is_operator());
    cls.def("__imul__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator*=, py::is_operator());
    cls.def("scaledMultiplies", &MI::scaledMultiplies);
    cls.def("__idiv__", (MI& (MI::*)(ImagePixelT const)) &MI::operator/=, py::is_operator());
    cls.def("__idiv__", (MI& (MI::*)(MI const &)) &MI::operator/=, py::is_operator());
    cls.def("__idiv__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator/=, py::is_operator());
    cls.def("scaledDivides", &MI::scaledDivides);

    /* Members */
    cls.def("writeFits", (void (MI::*)(std::string const &,
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet)) const) &MI::writeFits,
                                       "fileName"_a,
                                       "metadata"_a=CONST_PTR(PropertySet)(),
                                       "imageMetadata"_a=CONST_PTR(PropertySet)(),
                                       "maskMetadata"_a=CONST_PTR(PropertySet)(),
                                       "varianceMetadata"_a=CONST_PTR(PropertySet)());
    cls.def("writeFits", (void (MI::*)(lsst::afw::fits::MemFileManager &,
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet)) const) &MI::writeFits,
                                       "manager"_a,
                                       "metadata"_a=CONST_PTR(PropertySet)(),
                                       "imageMetadata"_a=CONST_PTR(PropertySet)(),
                                       "maskMetadata"_a=CONST_PTR(PropertySet)(),
                                       "varianceMetadata"_a=CONST_PTR(PropertySet)());
    cls.def("writeFits", (void (MI::*)(lsst::afw::fits::Fits &,
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet),
                                       CONST_PTR(PropertySet)) const) &MI::writeFits,
                                       "fitsfile"_a,
                                       "metadata"_a=CONST_PTR(PropertySet)(),
                                       "imageMetadata"_a=CONST_PTR(PropertySet)(),
                                       "maskMetadata"_a=CONST_PTR(PropertySet)(),
                                       "varianceMetadata"_a=CONST_PTR(PropertySet)());
    cls.def_static("readFits", (MI (*)(std::string const &)) MI::readFits,
                   "filename"_a);
    cls.def_static("readFits", (MI (*)(lsst::afw::fits::MemFileManager &)) MI::readFits,
                   "manager"_a);
    cls.def("getImage", &MI::getImage,
            "noThrow"_a=false);
    cls.def("getMask", &MaskedImage<ImagePixelT>::getMask,
            "noThrow"_a=false);
    cls.def("getVariance", &MaskedImage<ImagePixelT>::getVariance,
            "noThrow"_a=false);
    cls.def("getWidth", &MaskedImage<ImagePixelT>::getWidth);
    cls.def("getHeight", &MaskedImage<ImagePixelT>::getHeight);
    cls.def("getDimensions", &MaskedImage<ImagePixelT>::getDimensions);
    cls.def("getBBox", &MaskedImage<ImagePixelT>::getBBox);
    cls.def("getX0", &MaskedImage<ImagePixelT>::getX0);
    cls.def("getY0", &MaskedImage<ImagePixelT>::getY0);
    cls.def("getXY0", &MaskedImage<ImagePixelT>::getXY0);
    cls.def("setXY0", (void (MI::*)(int const, int const)) &MaskedImage<ImagePixelT>::setXY0,
            "x0"_a, "y0"_a);
    cls.def("setXY0", (void (MI::*)(lsst::afw::geom::Point2I const)) &MaskedImage<ImagePixelT>::setXY0,
            "origin"_a);
    cls.def("indexToPosition", &MaskedImage<ImagePixelT>::indexToPosition);
    cls.def("positionToIndex", &MaskedImage<ImagePixelT>::positionToIndex);
}

PYBIND11_PLUGIN(_maskedImage) {
    py::module mod("_maskedImage", "Python wrapper for afw _maskedImage library");

    declareMaskedImage<int>(mod, "I");
    declareMaskedImage<float>(mod, "F");
    declareMaskedImage<double>(mod, "D");
    declareMaskedImage<std::uint16_t>(mod, "U");
    declareMaskedImage<std::uint64_t>(mod, "L");

    /* Module level */

    return mod.ptr();
}