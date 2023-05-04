/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "lsst/utils/python.h"

#include <string>
#include <vector>

#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/typehandling/Storable.h"
#include "lsst/afw/image/ImageSummary.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyImageSummary = py::class_<ImageSummary, std::shared_ptr<ImageSummary>, typehandling::Storable>;
}  // namespace

void wrapImageSummary(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.addSignatureDependency("lsst.afw.table"); // for Schema
    wrappers.addInheritanceDependency("lsst.afw.table.io");
    wrappers.addInheritanceDependency("lsst.afw.typehandling");
    wrappers.wrapType(PyImageSummary(wrappers.module, "ImageSummary"), [](auto &mod, auto &cls) {
        cls.def(py::init<table::Schema const &>(), "schema"_a);
        cls.def(py::init<ImageSummary const &>(), "imageSummary"_a);

        table::io::python::addPersistableMethods<ImageSummary>(cls);

        /* Members */
        cls.def("__setitem__", &ImageSummary::setBool);
        cls.def("__setitem__", &ImageSummary::setInt64);
        cls.def("__setitem__", &ImageSummary::setDouble);

        cls.def("__getitem__", [] (ImageSummary const& self, std::string const& key){
            py::object result;
            self.getSchema().findAndApply(key, [&self, &key, &result](auto const& item){
                result = py::cast(self.getKey(item.key));
            });
            return result;
        });
    });
}

}  // namespace image
}  // namespace afw
}  // namespace lsst
