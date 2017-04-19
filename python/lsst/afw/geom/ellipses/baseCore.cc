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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Convolution.h"
#include "lsst/afw/geom/ellipses/Transformer.h"

namespace py = pybind11;

using namespace lsst::afw::geom::ellipses;

PYBIND11_PLUGIN(_baseCore) {
    py::module mod("_baseCore", "Python wrapper for afw _baseCore library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    /* Module level */
    py::class_<BaseCore, std::shared_ptr<BaseCore>> clsBaseCore(mod, "BaseCore");

    /* Member types and enums */
    py::class_<BaseCore::Convolution> clsBaseCoreConvolution(clsBaseCore, "Convolution");
    py::class_<BaseCore::Transformer> clsBaseCoreTransformer(clsBaseCore, "Transformer");

    //    clsBaseCoreTransformer.def(py::init<BaseCore &, LinearTransform const &>());
    //
    //    clsBaseCoreTransformer.def("inPlace", &BaseCore::Transformer::inPlace);
    //    clsBaseCoreTransformer.def("apply", &BaseCore::Transformer::apply);
    //    clsBaseCoreTransformer.def("d", &BaseCore::Transformer::d);
    //    clsBaseCoreTransformer.def("dTransform", &BaseCore::Transformer::dTransform);
    //
    //    clsBaseCoreTransformer.def_readwrite("input", &BaseCore::Transformer::input);
    //    clsBaseCoreTransformer.def_readonly("transform", &BaseCore::Transformer::transform);

    /* Constructors */

    /* Operators */
    clsBaseCore.def("__eq__", &BaseCore::operator==, py::is_operator());
    clsBaseCore.def("__nq__", &BaseCore::operator!=, py::is_operator());

    /* Members */
    clsBaseCore.def("getName", &BaseCore::getName);
    clsBaseCore.def("clone", &BaseCore::clone);
    clsBaseCore.def("normalize", &BaseCore::normalize);
    clsBaseCore.def("grow", &BaseCore::grow);
    clsBaseCore.def("scale", &BaseCore::scale);
    clsBaseCore.def("getArea", &BaseCore::getArea);
    clsBaseCore.def("getDeterminantRadius", &BaseCore::getDeterminantRadius);
    clsBaseCore.def("getTraceRadius", &BaseCore::getTraceRadius);
    //    clsBaseCore.def("transform", (typename BaseCore::Transformer const
    //    (BaseCore::*)(lsst::afw::geom::LinearTransform const &) const) &BaseCore::transform);
    //    clsBaseCore.def("getGridTransform", &BaseCore::getGridTransform);
    clsBaseCore.def("convolve", (BaseCore::Convolution (BaseCore::*)(BaseCore const &)) & BaseCore::convolve);
    clsBaseCore.def("computeDimensions", &BaseCore::computeDimensions);
    clsBaseCore.def("getParameterVector", &BaseCore::getParameterVector);
    clsBaseCore.def("setParameterVector", &BaseCore::setParameterVector);
    //    clsBaseCore.def("transformInPlace", [](BaseCore & self, lsst::afw::geom::LinearTransform const & t)
    //    {
    //       self.transform(t).inPlace();
    //    });

    return mod.ptr();
}