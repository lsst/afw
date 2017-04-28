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

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Background.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
template <typename ImageT>
void declareMakeBackground(py::module &mod) {
    mod.def("makeBackground", makeBackground<ImageT>, "img"_a, "bgCtrl"_a);
}

template <typename PixelT, typename PyClass>
void declareGetImage(PyClass &cls, std::string const &suffix) {
    cls.def(("getImage" + suffix).c_str(),
            (std::shared_ptr<lsst::afw::image::Image<PixelT>> (BackgroundMI::*)(
                    Interpolate::Style const, UndersampleStyle const) const) &
                    BackgroundMI::getImage<PixelT>,
            "interpStyle"_a, "undersampleStyle"_a = THROW_EXCEPTION);
    cls.def(("getImage" + suffix).c_str(),
            (std::shared_ptr<lsst::afw::image::Image<PixelT>> (BackgroundMI::*)(std::string const &,
                                                                                std::string const &) const) &
                    BackgroundMI::getImage<PixelT>,
            "interpStyle"_a, "undersampleStyle"_a = "THROW_EXCEPTION");
    cls.def(("getImage" + suffix).c_str(),
            (std::shared_ptr<lsst::afw::image::Image<PixelT>> (BackgroundMI::*)(
                    lsst::afw::geom::Box2I const &, Interpolate::Style const, UndersampleStyle const) const) &
                    BackgroundMI::getImage<PixelT>,
            "bbox"_a, "interpStyle"_a, "undersampleStyle"_a = THROW_EXCEPTION);
    cls.def(("getImage" + suffix).c_str(),
            (std::shared_ptr<lsst::afw::image::Image<PixelT>> (BackgroundMI::*)(
                    lsst::afw::geom::Box2I const &, std::string const &, std::string const &) const) &
                    BackgroundMI::getImage<PixelT>,
            "bbox"_a, "interpStyle"_a, "undersampleStyle"_a = "THROW_EXCEPTION");
    cls.def(("getImage" + suffix).c_str(),
            (std::shared_ptr<lsst::afw::image::Image<PixelT>> (BackgroundMI::*)() const) &
                    BackgroundMI::getImage<PixelT>);
}
}

PYBIND11_PLUGIN(_background) {
    py::module mod("_background", "Python wrapper for afw _background library");

    /* Member types and enums */
    py::enum_<UndersampleStyle>(mod, "UndersampleStyle")
            .value("THROW_EXCEPTION", UndersampleStyle::THROW_EXCEPTION)
            .value("REDUCE_INTERP_ORDER", UndersampleStyle::REDUCE_INTERP_ORDER)
            .value("INCREASE_NXNYSAMPLE", UndersampleStyle::INCREASE_NXNYSAMPLE)
            .export_values();

    py::class_<BackgroundControl, std::shared_ptr<BackgroundControl>> clsBackgroundControl(
            mod, "BackgroundControl");

    /* Constructors */
    clsBackgroundControl.def(py::init<int const, int const, StatisticsControl const, Property const,
                                      ApproximateControl const>(),
                             "nxSample"_a, "nySample"_a, "sctrl"_a = StatisticsControl(), "prop"_a = MEANCLIP,
                             "actrl"_a = ApproximateControl(ApproximateControl::UNKNOWN, 1));
    clsBackgroundControl.def(py::init<int const, int const, StatisticsControl const, std::string const &,
                                      ApproximateControl const>(),
                             "nxSample"_a, "nySample"_a, "sctrl"_a, "prop"_a,
                             "actrl"_a = ApproximateControl(ApproximateControl::UNKNOWN, 1));
    clsBackgroundControl.def(py::init<Interpolate::Style const, int const, int const, UndersampleStyle const,
                                      StatisticsControl const, Property const, ApproximateControl const>(),
                             "style"_a, "nxSample"_a = 10, "nySample"_a = 10,
                             "undersampleStyle"_a = THROW_EXCEPTION, "sctrl"_a = StatisticsControl(),
                             "prop"_a = MEANCLIP,
                             "actrl"_a = ApproximateControl(ApproximateControl::UNKNOWN, 1));
    clsBackgroundControl.def(
            py::init<std::string const &, int const, int const, std::string const &, StatisticsControl const,
                     std::string const &, ApproximateControl const>(),
            "style"_a, "nxSample"_a = 10, "nySample"_a = 10, "undersampleStyle"_a = "THROW_EXCEPTION",
            "sctrl"_a = StatisticsControl(), "prop"_a = "MEANCLIP",
            "actrl"_a = ApproximateControl(ApproximateControl::UNKNOWN, 1));

    /* Members */
    clsBackgroundControl.def("setNxSample", &BackgroundControl::setNxSample);
    clsBackgroundControl.def("setNySample", &BackgroundControl::setNySample);
    clsBackgroundControl.def("setInterpStyle", (void (BackgroundControl::*)(Interpolate::Style const)) &
                                                       BackgroundControl::setInterpStyle);
    clsBackgroundControl.def("setInterpStyle", (void (BackgroundControl::*)(std::string const &)) &
                                                       BackgroundControl::setInterpStyle);
    clsBackgroundControl.def("setUndersampleStyle", (void (BackgroundControl::*)(UndersampleStyle const)) &
                                                            BackgroundControl::setUndersampleStyle);
    clsBackgroundControl.def("setUndersampleStyle", (void (BackgroundControl::*)(std::string const &)) &
                                                            BackgroundControl::setUndersampleStyle);
    clsBackgroundControl.def("getNxSample", &BackgroundControl::getNxSample);
    clsBackgroundControl.def("getNySample", &BackgroundControl::getNySample);
    clsBackgroundControl.def("getInterpStyle", &BackgroundControl::getInterpStyle);
    clsBackgroundControl.def("getUndersampleStyle", &BackgroundControl::getUndersampleStyle);
    clsBackgroundControl.def("getStatisticsControl",
                             (std::shared_ptr<StatisticsControl> (BackgroundControl::*)()) &
                                     BackgroundControl::getStatisticsControl);
    clsBackgroundControl.def("getStatisticsProperty", &BackgroundControl::getStatisticsProperty);
    clsBackgroundControl.def("setStatisticsProperty", (void (BackgroundControl::*)(Property)) &
                                                              BackgroundControl::setStatisticsProperty);
    clsBackgroundControl.def("setStatisticsProperty", (void (BackgroundControl::*)(std::string)) &
                                                              BackgroundControl::setStatisticsProperty);
    clsBackgroundControl.def("setApproximateControl", &BackgroundControl::setApproximateControl);
    clsBackgroundControl.def("getApproximateControl",
                             (std::shared_ptr<ApproximateControl> (BackgroundControl::*)()) &
                                     BackgroundControl::getApproximateControl);

    /* Note that, in this case, the holder type must be unique_ptr to enable usage
     * of py::nodelete, which in turn is needed because Background has a protected
     * destructor. Adding py::nodelete prevents pybind11 from calling the destructor
     * when the pointer is destroyed. Thus care needs to be taken to prevent leaks.
     * Basically Background should only ever be used as a base class (without data
     * members). */
    py::class_<Background, std::unique_ptr<Background, py::nodelete>> clsBackground(mod, "Background");

    /* Members */
    declareGetImage<float>(clsBackground, "F");

    clsBackground.def("getAsUsedInterpStyle", &Background::getAsUsedInterpStyle);
    clsBackground.def("getAsUsedUndersampleStyle", &Background::getAsUsedUndersampleStyle);
    clsBackground.def("getApproximate", &Background::getApproximate, "actrl"_a,
                      "undersampleStyle"_a = THROW_EXCEPTION);
    clsBackground.def("getBackgroundControl", (std::shared_ptr<BackgroundControl> (Background::*)()) &
                                                      Background::getBackgroundControl);

    py::class_<BackgroundMI, std::shared_ptr<BackgroundMI>, Background> clsBackgroundMI(mod, "BackgroundMI");

    /* Constructors */
    clsBackgroundMI.def(
            py::init<geom::Box2I const, image::MaskedImage<typename Background::InternalPixelT> const &>(),
            "imageDimensions"_a, "statsImage"_a);

    /* Operators */
    clsBackgroundMI.def("__iadd__", &BackgroundMI::operator+=);
    clsBackgroundMI.def("__isub__", &BackgroundMI::operator-=);

    /* Members */
    declareGetImage<float>(clsBackgroundMI, "F");

    clsBackgroundMI.def("getPixel",
                        (double (BackgroundMI::*)(Interpolate::Style const, int const, int const) const) &
                                BackgroundMI::getPixel);
    clsBackgroundMI.def("getPixel",
                        (double (BackgroundMI::*)(int const, int const) const) & BackgroundMI::getPixel);
    clsBackgroundMI.def("getStatsImage", &BackgroundMI::getStatsImage);
    clsBackgroundMI.def("getImageBBox", &BackgroundMI::getImageBBox);

    // Yes, really only float
    declareMakeBackground<image::Image<float>>(mod);
    declareMakeBackground<image::MaskedImage<float>>(mod);

    mod.def("stringToUndersampleStyle", stringToUndersampleStyle, "style"_a);

    return mod.ptr();
}
}
}
}  // lsst::afw::math