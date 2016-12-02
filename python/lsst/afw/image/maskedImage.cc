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
#include <pybind11/stl.h>

#include "lsst/afw/fits.h"
#include "lsst/afw/image/MaskedImage.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {

namespace {
/**
Declare a constructor that takes a MaskedImage of FromPixelT and returns a MaskedImage cast to ToPixelT

The mask and variance must be of the standard types.

@param[in] src  The MaskedImage to cast.
@param[in] deep  Make a deep copy? Must be specified and must be `true`, for disambiguation.
*/
template <typename FromPixelT, typename ToPixelT>
void declareCastConstructor(py::class_<MaskedImage<ToPixelT, MaskPixel, VariancePixel>,
                                       std::shared_ptr<MaskedImage<ToPixelT, MaskPixel, VariancePixel>>,
                                       lsst::daf::base::Persistable,
                                       lsst::daf::base::Citizen> & cls) {
    cls.def(py::init<MaskedImage<FromPixelT, MaskPixel, VariancePixel> const &, bool const>(),
            "src"_a, "deep"_a);
}

template <typename ImagePixelT>  // only the image type varies; mask and variance are fixed
py::class_<MaskedImage<ImagePixelT, MaskPixel, VariancePixel>,
           std::shared_ptr<MaskedImage<ImagePixelT, MaskPixel, VariancePixel>>,
           lsst::daf::base::Persistable,
           lsst::daf::base::Citizen> declareMaskedImage(py::module & mod, const std::string & suffix) {
    using MI = MaskedImage<ImagePixelT, MaskPixel, VariancePixel>;

    py::class_<MI, std::shared_ptr<MI>, lsst::daf::base::Persistable, lsst::daf::base::Citizen>
        cls(mod, ("MaskedImage" + suffix).c_str());

    mod.def("makeMaskedImage", &makeMaskedImage<ImagePixelT, MaskPixel, VariancePixel>,
            "image"_a, "mask"_a=std::shared_ptr<Mask<MaskPixel>>(),
            "variance"_a=std::shared_ptr<Image<VariancePixel>>());

    /* Member types and enums */

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, typename MI::MaskPlaneDict const&>(),
            "width"_a, "height"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<geom::Extent2I, typename MI::MaskPlaneDict const&>(),
            "dimensions"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<typename MI::ImagePtr, typename MI::MaskPtr, typename MI::VariancePtr>(),
            "image"_a, "mask"_a=typename MI::MaskPtr(), "variance"_a=typename MI::VariancePtr());
    cls.def(py::init<geom::Box2I const &, typename MI::MaskPlaneDict const&>(),
            "bbox"_a, "planeDict"_a=typename MI::MaskPlaneDict());
    cls.def(py::init<std::string const &,
                     std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>>(),
                     "fileName"_a,
                     "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "bbox"_a=geom::Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet>());
    cls.def(py::init<lsst::afw::fits::MemFileManager &,
                     std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>>(),
                     "manager"_a,
                     "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "bbox"_a=geom::Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet>());
    cls.def(py::init<lsst::afw::fits::Fits &,
                     std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &,
                     ImageOrigin,
                     bool,
                     bool,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>,
                     std::shared_ptr<daf::base::PropertySet>>(),
                     "fitsfile"_a,
                     "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "bbox"_a=geom::Box2I(),
                     "origin"_a=PARENT,
                     "conformMasks"_a=false,
                     "needAllHdus"_a=false,
                     "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet>(),
                     "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet>());
    cls.def(py::init<MI const &, bool>(),
            "rhs"_a, "deep"_a=false);
    cls.def(py::init<MI const &, geom::Box2I const &, ImageOrigin, bool>(),
            "rhs"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);

    /* Operators */
    cls.def("swap", &MI::swap);
    cls.def("assign", &MI::assign,
            "rhs"_a, "bbox"_a=geom::Box2I(), "origin"_a=PARENT, py::is_operator());

    cls.def("__ilshift__", &MI::operator<<=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(ImagePixelT const)) &MI::operator+=, py::is_operator());
    cls.def("__iadd__", (MI& (MI::*)(MI const &)) &MI::operator+=, py::is_operator());
    cls.def("__iadd__",
            (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &))&MI::operator+=, py::is_operator());
    cls.def("__iadd__",
            (MI& (MI::*)(lsst::afw::math::Function2<double> const &)) &MI::operator+=, py::is_operator());
    cls.def("scaledPlus", &MI::scaledPlus);
    cls.def("__isub__", (MI& (MI::*)(ImagePixelT const)) &MI::operator-=, py::is_operator());
    cls.def("__isub__", (MI& (MI::*)(MI const &)) &MI::operator-=, py::is_operator());
    cls.def("__isub__",
            (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator-=, py::is_operator());
    cls.def("__isub__",
            (MI& (MI::*)(lsst::afw::math::Function2<double> const &)) &MI::operator-=, py::is_operator());
    cls.def("scaledMinus", &MI::scaledMinus);
    cls.def("__imul__", (MI& (MI::*)(ImagePixelT const)) &MI::operator*=, py::is_operator());
    cls.def("__imul__", (MI& (MI::*)(MI const &)) &MI::operator*=, py::is_operator());
    cls.def("__imul__",
            (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator*=, py::is_operator());
    cls.def("scaledMultiplies", &MI::scaledMultiplies);
    cls.def("__idiv__", (MI& (MI::*)(ImagePixelT const)) &MI::operator/=, py::is_operator());
    cls.def("__idiv__", (MI& (MI::*)(MI const &)) &MI::operator/=, py::is_operator());
    cls.def("__idiv__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator/=, py::is_operator());
    cls.def("__itruediv__", (MI& (MI::*)(ImagePixelT const)) &MI::operator/=, py::is_operator());
    cls.def("__itruediv__", (MI& (MI::*)(MI const &)) &MI::operator/=, py::is_operator());
    cls.def("__itruediv__", (MI& (MI::*)(lsst::afw::image::Image<ImagePixelT> const &)) &MI::operator/=, py::is_operator());
    cls.def("scaledDivides", &MI::scaledDivides);

    /* Members */
    cls.def("writeFits", (void (MI::*)(std::string const &,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>) const) &MI::writeFits,
                                       "fileName"_a,
                                       "metadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet const>());
    cls.def("writeFits", (void (MI::*)(lsst::afw::fits::MemFileManager &,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>) const) &MI::writeFits,
                                       "manager"_a,
                                       "metadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet const>());
    cls.def("writeFits", (void (MI::*)(lsst::afw::fits::Fits &,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>,
                                       std::shared_ptr<daf::base::PropertySet const>) const) &MI::writeFits,
                                       "fitsfile"_a,
                                       "metadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "imageMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "maskMetadata"_a=std::shared_ptr<daf::base::PropertySet const>(),
                                       "varianceMetadata"_a=std::shared_ptr<daf::base::PropertySet const>());
    cls.def_static("readFits", (MI (*)(std::string const &)) MI::readFits,
                   "filename"_a);
    cls.def_static("readFits", (MI (*)(lsst::afw::fits::MemFileManager &)) MI::readFits,
                   "manager"_a);
    cls.def("getImage", &MI::getImage,
            "noThrow"_a=false);
    cls.def("getMask", &MI::getMask,
            "noThrow"_a=false);
    cls.def("getVariance", &MI::getVariance,
            "noThrow"_a=false);
    cls.def("getWidth", &MI::getWidth);
    cls.def("getHeight", &MI::getHeight);
    cls.def("getDimensions", &MI::getDimensions);
    cls.def("getBBox", &MI::getBBox,
            "origin"_a=PARENT);
    cls.def("getX0", &MI::getX0);
    cls.def("getY0", &MI::getY0);
    cls.def("getXY0", &MI::getXY0);
    cls.def("setXY0", (void (MI::*)(int const, int const)) &MI::setXY0,
            "x0"_a, "y0"_a);
    cls.def("setXY0", (void (MI::*)(lsst::afw::geom::Point2I const)) &MI::setXY0,
            "origin"_a);
    cls.def("indexToPosition", &MI::indexToPosition);
    cls.def("positionToIndex", &MI::positionToIndex);

    return cls;
}

template <typename ImagePixelT> // addtional template types do not seem to be needed
void declareMakeMaskedImage(py::module & mod) {
    mod.def("makeMaskedImage", makeMaskedImage<ImagePixelT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>,
            "image"_a, "mask"_a=Mask<lsst::afw::image::MaskPixel>::Ptr(), "variance"_a=Image<lsst::afw::image::VariancePixel>::Ptr());
}
}

PYBIND11_PLUGIN(_maskedImage) {
    py::module mod("_maskedImage", "Python wrapper for afw _maskedImage library");

    auto clsMaskedImageF = declareMaskedImage<float>(mod, "F");
    auto clsMaskedImageD = declareMaskedImage<double>(mod, "D");
    auto clsMaskedImageI = declareMaskedImage<int>(mod, "I");
    auto clsMaskedImageU = declareMaskedImage<std::uint16_t>(mod, "U");
    auto clsMaskedImageL = declareMaskedImage<std::uint64_t>(mod, "L");

    // Declare constructors for casting all exposure types to to float and double
    // (the only two types of casts that Python supports)
    declareCastConstructor<int, float>(clsMaskedImageF);
    declareCastConstructor<int, double>(clsMaskedImageD);
    declareCastConstructor<float, double>(clsMaskedImageD);
    declareCastConstructor<double, float>(clsMaskedImageF);
    declareCastConstructor<std::uint16_t, float>(clsMaskedImageF);
    declareCastConstructor<std::uint16_t, double>(clsMaskedImageD);
    declareCastConstructor<std::uint64_t, float>(clsMaskedImageF);
    declareCastConstructor<std::uint64_t, double>(clsMaskedImageD);

    /* Yes, only for float and std::uint16_t */
    clsMaskedImageF.def("convertD", [](MaskedImage<float> & self){
         return lsst::afw::image::MaskedImage<double,
                                            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(self, true);
    });

    clsMaskedImageU.def("convertF", [](MaskedImage<std::uint16_t> & self){
         return lsst::afw::image::MaskedImage<float,
                                            lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>(self, true);
    });

    /* Module level */
    declareMakeMaskedImage<int>(mod);
    declareMakeMaskedImage<float>(mod);
    declareMakeMaskedImage<double>(mod);
    declareMakeMaskedImage<std::uint16_t>(mod);
    declareMakeMaskedImage<std::uint64_t>(mod);

    return mod.ptr();
}
}}}  // namspace lsst::afw::image
