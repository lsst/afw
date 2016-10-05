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

#include "lsst/afw/math/Stack.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::math;

template <typename PixelT>
void declareStatisticsStack(py::module & mod) {
    mod.def("statisticsStack", (typename lsst::afw::image::MaskedImage<PixelT>::Ptr (*)(
        lsst::afw::image::Image<PixelT> const &,
        Property,
        char,
        StatisticsControl const &
        )) statisticsStack<PixelT>,
        "image"_a,
        "flags"_a,
        "dimensions"_a,
        "sctrl"_a=StatisticsControl()
    );
    mod.def("statisticsStack", (typename lsst::afw::image::MaskedImage<PixelT>::Ptr (*)(
        lsst::afw::image::MaskedImage<PixelT> const &,
        Property,
        char,
        StatisticsControl const &
        )) statisticsStack<PixelT>,
        "image"_a,
        "flags"_a,
        "dimensions"_a,
        "sctrl"_a=StatisticsControl()
    );
}

template <typename PixelT>
void declareStatisticsStackVectorOverloads(py::module & mod) {
    mod.def("statisticsStack", (typename lsst::afw::image::Image<PixelT>::Ptr (*)(
        std::vector<typename lsst::afw::image::Image<PixelT>::Ptr > &,
        Property,
        StatisticsControl const &,
        std::vector<lsst::afw::image::VariancePixel> const &
        )) statisticsStack<PixelT>,
        "images"_a,
        "flags"_a,
        "sctrl"_a=StatisticsControl(),
        "wvector"_a=std::vector<lsst::afw::image::VariancePixel>(0)
    );
    mod.def("statisticsStack", (void (*)(
        lsst::afw::image::Image<PixelT> &,
        std::vector<typename lsst::afw::image::Image<PixelT>::Ptr > &,
        Property,
        StatisticsControl const &,
        std::vector<lsst::afw::image::VariancePixel> const &
        )) statisticsStack<PixelT>,
        "out"_a,
        "images"_a,
        "flags"_a,
        "sctrl"_a=StatisticsControl(),
        "wvector"_a=std::vector<lsst::afw::image::VariancePixel>(0)
    );
    mod.def("statisticsStack", (typename lsst::afw::image::MaskedImage<PixelT>::Ptr (*)(
        std::vector<typename lsst::afw::image::MaskedImage<PixelT>::Ptr > &,
        Property,
        StatisticsControl const &,
        std::vector<lsst::afw::image::VariancePixel> const &
        )) statisticsStack<PixelT>,
        "images"_a,
        "flags"_a,
        "sctrl"_a=StatisticsControl(),
        "wvector"_a=std::vector<lsst::afw::image::VariancePixel>(0)
    );
    mod.def("statisticsStack", (void (*)(
        lsst::afw::image::MaskedImage<PixelT> &,
        std::vector<typename lsst::afw::image::MaskedImage<PixelT>::Ptr > &,
        Property,
        StatisticsControl const &,
        std::vector<lsst::afw::image::VariancePixel> const &
        )) statisticsStack<PixelT>,
        "out"_a,
        "images"_a,
        "flags"_a,
        "sctrl"_a=StatisticsControl(),
        "wvector"_a=std::vector<lsst::afw::image::VariancePixel>(0)
    );
    mod.def("statisticsStack", (std::shared_ptr<std::vector<PixelT>> (*)(
        std::vector<std::shared_ptr<std::vector<PixelT>>> &,
        Property,
        StatisticsControl const &,
        std::vector<lsst::afw::image::VariancePixel> const &
        )) statisticsStack<PixelT>,
        "vectors"_a,
        "flags"_a,
        "sctrl"_a=StatisticsControl(),
        "wvector"_a=std::vector<lsst::afw::image::VariancePixel>(0)
    );
}

PYBIND11_PLUGIN(_stack) {
    py::module mod("_stack", "Python wrapper for afw _stack library");

    /* Module level */
    declareStatisticsStack<float>(mod);
    declareStatisticsStack<double>(mod);
    declareStatisticsStackVectorOverloads<float>(mod);
    declareStatisticsStackVectorOverloads<double>(mod);

    return mod.ptr();
}