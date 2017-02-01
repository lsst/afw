/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
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


#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/table/io/pybind11.h" // for declarePersistableFacade
#include "lsst/afw/detection/Bootprint.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst { namespace afw { namespace detection {

PYBIND11_PLUGIN(_bootprint) {
    py::module mod("_bootprint", "Python wrapper for afw Bootprint library");
    table::io::declarePersistableFacade<Bootprint>(mod, "Bootprint");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    /* Bootprint Constructors */
    py::class_<Bootprint,
               std::shared_ptr<Bootprint>,
               daf::base::Citizen,
               table::io::Persistable,
               table::io::PersistableFacade<Bootprint>> clsBootprint(mod, "Bootprint");
    clsBootprint.def(py::init<std::shared_ptr<geom::SpanSet>, geom::Box2I const &>(),
                     "inputSpans"_a, "region"_a = geom::Box2I());

    clsBootprint.def(py::init<std::shared_ptr<geom::SpanSet>,
                              afw::table::Schema const &,
                              geom::Box2I const &>(),
                     "inputSpans"_a, "peakSchema"_a, "region"_a = geom::Box2I());

    clsBootprint.def(py::init<>());

    /* Bootprint Methods */
    clsBootprint.def("getSpans", &Bootprint::getSpans);
    clsBootprint.def("setSpans", &Bootprint::setSpans);
    clsBootprint.def("getPeaks", (PeakCatalog & (Bootprint::*)()) &Bootprint::getPeaks,
                     py::return_value_policy::reference_internal);
    clsBootprint.def("addPeak", &Bootprint::addPeak);
    clsBootprint.def("sortPeaks", &Bootprint::sortPeaks, "key"_a = afw::table::Key<float>());
    clsBootprint.def("setPeakSchema", &Bootprint::setPeakSchema);
    clsBootprint.def("getArea", &Bootprint::getArea);
    clsBootprint.def("getCentroid", &Bootprint::getCentroid);
    clsBootprint.def("getShape", &Bootprint::getShape);
    clsBootprint.def("shift", (void (Bootprint::*)(int, int)) &Bootprint::shift);
    clsBootprint.def("shift", (void (Bootprint::*)(geom::ExtentI const &)) &Bootprint::shift);
    clsBootprint.def("getBBox", &Bootprint::getBBox);
    clsBootprint.def("getRegion", &Bootprint::getRegion);
    clsBootprint.def("setRegion", &Bootprint::setRegion);
    clsBootprint.def("clipTo", &Bootprint::clipTo);
    clsBootprint.def("contains", &Bootprint::contains);
    clsBootprint.def("transform", &Bootprint::transform,
                     "source"_a, "target"_a, "region"_a, "doClip"_a = true);
    clsBootprint.def("dilate", (void (Bootprint::*)(int, geom::Stencil)) &Bootprint::dilate,
                     "r"_a, "stencil"_a = geom::Stencil::CIRCLE);
    clsBootprint.def("dilate", (void (Bootprint::*)(geom::SpanSet const &)) &Bootprint::dilate);
    clsBootprint.def("erode", (void (Bootprint::*)(int, geom::Stencil)) &Bootprint::erode,
                     "r"_a, "stencil"_a = geom::Stencil::CIRCLE);
    clsBootprint.def("erode", (void (Bootprint::*)(geom::SpanSet const &)) &Bootprint::erode);
    clsBootprint.def("removeOrphanPeaks", &Bootprint::removeOrphanPeaks);
    clsBootprint.def("isContiguous", &Bootprint::isContiguous);
    clsBootprint.def("split", []
                              (Bootprint const & self) -> py::list
                              {
                                  // This is a work around for pybind not properly
                                  // handling converting a vector of unique pointers
                                  // to python lists of shared pointers
                                  py::list l;
                                  for (auto & ptr: self.split()) {
                                      l.append(py::cast(std::shared_ptr<Bootprint>(std::move(ptr))));
                                  }
                                  return l;
                              });

    /* Define python level properties */
    clsBootprint.def_property("spans", &Bootprint::getSpans, &Bootprint::setSpans);
    clsBootprint.def_property_readonly("peaks", (PeakCatalog & (Bootprint::*)()) &Bootprint::getPeaks,
                                       py::return_value_policy::reference);
    clsBootprint.def_property_readonly("isHeavy", &Bootprint::isHeavy);


    /* Python Operators functions */
    clsBootprint.def("__contains__", []
                                     (Bootprint const & self,
                                      geom::Point2I const & point)->bool {
                                          return self.contains(point);
                                      });
    clsBootprint.def("__eq__", []
                               (Bootprint const & self,
                                Bootprint const & other)->bool {
                                    return self == other;
                                }, py::is_operator());

    return mod.ptr();
}
}}} // close lsst::afw::detection
