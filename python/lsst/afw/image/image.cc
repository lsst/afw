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

namespace lsst {
namespace afw {
namespace image {

/**
Declare a constructor that takes a MaskedImage of FromPixelT and returns a MaskedImage cast to ToPixelT

The mask and variance must be of the standard types.

@param[in] src  The MaskedImage to cast.
@param[in] deep  Make a deep copy? Must be specified and must be `true`, for disambiguation.
*/
template <typename FromPixelT, typename ToPixelT>
void declareCastConstructor(py::class_<Image<ToPixelT>,
                                       std::shared_ptr<Image<ToPixelT>>,
                                       ImageBase<ToPixelT>> & cls) {
    cls.def(py::init<Image<FromPixelT> const &, bool const>(),
            "src"_a, "deep"_a);
}

template <typename PixelT>
void declareImageBase(py::module & mod, const std::string & suffix) {
    py::class_<ImageBase<PixelT>,
               std::shared_ptr<ImageBase<PixelT>>,
               daf::base::Persistable,
               daf::base::Citizen> cls(mod, ("ImageBase" + suffix).c_str());

    cls.def(py::init<const geom::Extent2I>(),
            "dimensions"_a=geom::Extent2I());
    cls.def(py::init<const ImageBase<PixelT>&, const bool>(),
            "src"_a, "deep"_a=false);
    cls.def(py::init<const ImageBase<PixelT>&, const geom::Box2I&, const ImageOrigin, bool>(),
            "src"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);
    cls.def(py::init<typename ImageBase<PixelT>::Array const &, bool, geom::Point2I const &>(),
            "array"_a, "deep"_a=false, "xy0"_a=geom::Point2I());

    cls.def("assign", &ImageBase<PixelT>::assign,
            "rhs"_a, "bbox"_a=geom::Box2I(), "origin"_a=PARENT, py::is_operator());
    cls.def("getWidth", &ImageBase<PixelT>::getWidth);
    cls.def("getHeight", &ImageBase<PixelT>::getHeight);
    cls.def("getX0", &ImageBase<PixelT>::getX0);
    cls.def("getY0", &ImageBase<PixelT>::getY0);
    cls.def("getXY0", &ImageBase<PixelT>::getXY0);
    cls.def("positionToIndex", &ImageBase<PixelT>::positionToIndex,
            "position"_a, "xOrY"_a);
    cls.def("indexToPosition", &ImageBase<PixelT>::indexToPosition,
            "index"_a, "xOrY"_a);
    cls.def("getDimensions", &ImageBase<PixelT>::getDimensions);
    cls.def("getArray",
            (typename ImageBase<PixelT>::Array (ImageBase<PixelT>::*)()) &ImageBase<PixelT>::getArray);
    cls.def("setXY0",
            (void (ImageBase<PixelT>::*)(geom::Point2I const)) &ImageBase<PixelT>::setXY0,
            "xy0"_a);
    cls.def("setXY0", (void (ImageBase<PixelT>::*)(int const, int const)) &ImageBase<PixelT>::setXY0,
            "x0"_a, "y0"_a);
    cls.def("getBBox", &ImageBase<PixelT>::getBBox,
            "origin"_a=PARENT);
}

template <typename PixelT>
py::class_<Image<PixelT>,
           std::shared_ptr<Image<PixelT>>,
           ImageBase<PixelT>> declareImage(py::module & mod, const std::string & suffix) {

    py::class_<Image<PixelT>,
               std::shared_ptr<Image<PixelT>>,
               ImageBase<PixelT>> cls(mod, ("Image" + suffix).c_str());

    /* Constructors */
    cls.def(py::init<unsigned int, unsigned int, PixelT>(),
            "width"_a, "height"_a, "intialValue"_a=0);
    cls.def(py::init<geom::Extent2I const &, PixelT>(),
            "dimensions"_a=geom::Extent2I(), "initialValue"_a=0);
    cls.def(py::init<geom::Box2I const &, PixelT>(),
            "bbox"_a, "initialValue"_a=0);
    cls.def(py::init<Image<PixelT> const &, geom::Box2I const &, ImageOrigin const, const bool>(),
            "rhs"_a, "bbox"_a, "origin"_a=PARENT, "deep"_a=false);
    cls.def(py::init<std::string const &, int, std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &, ImageOrigin>(),
            "fileName"_a, "hdu"_a=0, "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
            "bbox"_a=geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<fits::MemFileManager &, int, std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &, ImageOrigin>(),
            "manager"_a, "hdu"_a=0, "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
            "bbox"_a=geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<fits::Fits &, std::shared_ptr<daf::base::PropertySet>,
                     geom::Box2I const &, ImageOrigin>(),
            "fitsFile"_a, "metadata"_a=std::shared_ptr<daf::base::PropertySet>(),
            "bbox"_a=geom::Box2I(), "origin"_a=PARENT);
    cls.def(py::init<ndarray::Array<PixelT,2,1> const &, bool, geom::Point2I const &>(),
            "array"_a, "deep"_a=false, "xy0"_a=geom::Point2I());

    /* Operators */
    cls.def("__iadd__", [](Image<PixelT> & self, PixelT const & other) {
                return self += other;
            }, py::is_operator());
    cls.def("__iadd__", [](Image<PixelT> & self, Image<PixelT> const & other) {
                return self += other;
            }, py::is_operator());
    cls.def("__iadd__", [](Image<PixelT> & self, lsst::afw::math::Function2<double> const & other) {
                return self += other;
            }, py::is_operator());
    cls.def("__isub__", [](Image<PixelT> & self, PixelT const & other) {
                return self -= other;
            }, py::is_operator());
    cls.def("__isub__", [](Image<PixelT> & self, Image<PixelT> const & other) {
                return self -= other;
            }, py::is_operator());
    cls.def("__isub__", [](Image<PixelT> & self, lsst::afw::math::Function2<double> const & other) {
                return self -= other;
            }, py::is_operator());
    cls.def("__imul__", [](Image<PixelT> & self, PixelT const & other) {
                return self *= other;
            }, py::is_operator());
    cls.def("__imul__", [](Image<PixelT> & self, Image<PixelT> const & other) {
                return self *= other;
            }, py::is_operator());
    cls.def("__itruediv__", [](Image<PixelT> & self, PixelT const & other) {
                return self /= other;
            }, py::is_operator());
    cls.def("__itruediv__", [](Image<PixelT> & self, Image<PixelT> const & other) {
                return self /= other;
            }, py::is_operator());

    /* Members */
    cls.def("scaledPlus", &Image<PixelT>::scaledPlus);
    cls.def("scaledMinus", &Image<PixelT>::scaledMinus);
    cls.def("scaledMultiplies", &Image<PixelT>::scaledMultiplies);
    cls.def("scaledDivides", &Image<PixelT>::scaledDivides);

    cls.def("writeFits",
            (void (Image<PixelT>::*)(std::string const&, std::shared_ptr<daf::base::PropertySet const>,
                                     std::string const&) const) &Image<PixelT>::writeFits,
            "fileName"_a, "metadata"_a=std::shared_ptr<daf::base::PropertySet const>(), "mode"_a="w");
    cls.def("writeFits",
            (void (Image<PixelT>::*)(fits::MemFileManager &,
                                     std::shared_ptr<daf::base::PropertySet const>,
                                     std::string const&) const) &Image<PixelT>::writeFits,
            "manager"_a, "metadata"_a=std::shared_ptr<daf::base::PropertySet const>(), "mode"_a="w");
    cls.def("writeFits",
            (void (Image<PixelT>::*)(fits::Fits &,
                                     std::shared_ptr<daf::base::PropertySet const>) const)
                &Image<PixelT>::writeFits,
            "fitsfile"_a, "metadata"_a=std::shared_ptr<daf::base::PropertySet const>());
    cls.def_static("readFits", (Image<PixelT> (*)(std::string const &, int)) Image<PixelT>::readFits,
                   "filename"_a, "hdu"_a=0);
    cls.def_static("readFits",
                   (Image<PixelT> (*)(fits::MemFileManager &, int)) Image<PixelT>::readFits,
                   "manager"_a, "hdu"_a=0);
    cls.def("sqrt", &Image<PixelT>::sqrt);

    /* Add-ons for Python interface only */
    cls.def("set", [](Image<PixelT> &img, double val) { img=val; });
    cls.def("set", [](Image<PixelT> &img, int x, int y, double val) { img(x, y, lsst::afw::image::CheckIndices(true))=val; });
    cls.def("get", [](Image<PixelT> &img, int x, int y) { return img(x, y, lsst::afw::image::CheckIndices(true)); });
    cls.def("set0", [](Image<PixelT> &img, int x, int y, double val) { img.set0(x, y, val, lsst::afw::image::CheckIndices(true)); });
    cls.def("get0", [](Image<PixelT> &img, int x, int y) { return img.get0(x, y, lsst::afw::image::CheckIndices(true)); });

    return cls;
}

template <typename PixelT>
void declareDecoratedImage(py::module & mod, std::string const & suffix) {
    using Class = DecoratedImage<PixelT>;

    py::class_<Class, std::shared_ptr<Class>, lsst::daf::base::Persistable> cls(mod, ("DecoratedImage" + suffix).c_str());

    cls.def(py::init<const lsst::afw::geom::Extent2I &>(),
            "dimensions"_a=lsst::afw::geom::Extent2I());
    cls.def(py::init<const lsst::afw::geom::Box2I &>(),
            "bbox"_a);
    cls.def(py::init<PTR(Image<PixelT>)>(),
            "rhs"_a);
    cls.def(py::init<Class const&, const bool>(),
            "rhs"_a, "deep"_a=false);
    cls.def(py::init<std::string const&,
            const int,
            lsst::afw::geom::Box2I const&,
            ImageOrigin const>(),
            "fileName"_a,
            "hdu"_a=0,
            "bbox"_a=lsst::afw::geom::Box2I(),
            "origin"_a=PARENT);

    cls.def("getMetadata", &Class::getMetadata);
    cls.def("setMetadata", &Class::setMetadata);
    cls.def("getWidth", &Class::getWidth);
    cls.def("getHeight", &Class::getHeight);
    cls.def("getX0", &Class::getX0);
    cls.def("getY0", &Class::getY0);
    cls.def("getDimensions", &Class::getDimensions);
    cls.def("swap", &Class::swap);
    cls.def("writeFits", &Class::writeFits,
            "fileName"_a, "metadata"_a=CONST_PTR(lsst::daf::base::PropertySet)(), "mode"_a="w");
    cls.def("getImage", (typename Class::ImagePtr (Class::*)()) &Class::getImage);
    cls.def("getGain", &Class::getGain);
    cls.def("setGain", &Class::setGain);
}

/* Declare ImageSlice operators separately since they are only instantiated for float double */
template <typename PixelT>
void addImageSliceOperators(py::class_<Image<PixelT>,
                                       std::shared_ptr<Image<PixelT>>,
                                       ImageBase<PixelT>> & cls) {
    cls.def("__add__",
            [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) {return self + other; },
            py::is_operator());
    cls.def("__sub__",
            [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self - other; },
            py::is_operator());
    cls.def("__mul__",
            [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self * other; },
            py::is_operator());
    cls.def("__truediv__",
            [](Image<PixelT> const & self, ImageSlice<PixelT> const & other) { return self / other; },
            py::is_operator());
    cls.def("__iadd__",
            [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self += other; return self; },
            py::is_operator());
    cls.def("__isub__",
            [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self -= other; return self; },
            py::is_operator());
    cls.def("__imul__",
            [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self *= other; return self; },
            py::is_operator());
    cls.def("__itruediv__",
            [](Image<PixelT> & self, ImageSlice<PixelT> const & other) { self /= other; return self; },
            py::is_operator());
}

template <typename PixelT, typename PyClass>
void addGeneralizedCopyConstructors(PyClass & cls) {
    cls.def(py::init<Image<int> const &, const bool>(), "rhs"_a, "deep"_a=false);
    cls.def(py::init<Image<float> const &, const bool>(), "rhs"_a, "deep"_a=false);
    cls.def(py::init<Image<double> const &, const bool>(), "rhs"_a, "deep"_a=false);
    cls.def(py::init<Image<std::uint16_t> const &, const bool>(), "rhs"_a, "deep"_a=false);
    cls.def(py::init<Image<std::uint64_t> const &, const bool>(), "rhs"_a, "deep"_a=false);

	cls.def("convertI", [](Image<PixelT> const & self) { return Image<int>(self, true); });
	cls.def("convertF", [](Image<PixelT> const & self) { return Image<float>(self, true); });
	cls.def("convertD", [](Image<PixelT> const & self) { return Image<double>(self, true); });
	cls.def("convertU", [](Image<PixelT> const & self) { return Image<std::uint16_t>(self, true); });
	cls.def("convertL", [](Image<PixelT> const & self) { return Image<std::uint64_t>(self, true); });

	cls.def("convertFloat", [](Image<PixelT> const & self) { return Image<float>(self, true); });
	cls.def("convertDouble", [](Image<PixelT> const & self) { return Image<double>(self, true); });
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

    auto clsImageI = declareImage<int>(mod, "I");
    auto clsImageF = declareImage<float>(mod, "F");
    auto clsImageD = declareImage<double>(mod, "D");
    auto clsImageU = declareImage<std::uint16_t>(mod, "U");
    auto clsImageL = declareImage<std::uint64_t>(mod, "L");

    // Add generalized copy constructors
    addGeneralizedCopyConstructors<int>(clsImageI);
    addGeneralizedCopyConstructors<float>(clsImageF);
    addGeneralizedCopyConstructors<double>(clsImageD);
    addGeneralizedCopyConstructors<std::uint16_t>(clsImageU);
    addGeneralizedCopyConstructors<std::uint64_t>(clsImageL);

    // Add slice operators only for float and double
    addImageSliceOperators<float>(clsImageF);
    addImageSliceOperators<double>(clsImageD);

    declareDecoratedImage<int>(mod, "I");
    declareDecoratedImage<float>(mod, "F");
    declareDecoratedImage<double>(mod, "D");
    declareDecoratedImage<std::uint16_t>(mod, "U");
    declareDecoratedImage<std::uint64_t>(mod, "L");

    // Declare constructors for casting all exposure types to to float and double
    // (the only two types of casts that Python supports)
    declareCastConstructor<int, float>(clsImageF);
    declareCastConstructor<int, double>(clsImageD);

    declareCastConstructor<float, double>(clsImageD);

    declareCastConstructor<double, float>(clsImageF);

    declareCastConstructor<std::uint16_t, float>(clsImageF);
    declareCastConstructor<std::uint16_t, double>(clsImageD);

    declareCastConstructor<std::uint64_t, float>(clsImageF);
    declareCastConstructor<std::uint64_t, double>(clsImageD);

    return mod.ptr();
}

}}}  // namespae image
