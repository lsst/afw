/*
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

#include "lsst/utils/python.h"

#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/fits.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace io {

using PyInputArchive = py::class_<InputArchive, std::shared_ptr<InputArchive>>;

void wrapInputArchive(utils::python::WrapperCollection &wrappers) {
    // TODO: uncomment once afw.fits uses WrapperCollection
    // wrappers.addSignatureDependency("lsst.afw.fits");

    wrappers.wrapType(PyInputArchive(wrappers.module, "InputArchive"), [](auto &mod, auto &cls) {
      /*cls.def("readFits",
              py::overload_cast<fits::Fits &, int>(&InputArchive::readFits, py::const_),
              "fitsFile"_a, "hdu"_a
              );*/
        cls.def("readFits", &InputArchive::readFits);
        // lambda version; also works to disambiguate with the templated version
        cls.def("get", [](const InputArchive& ia, int id){
            return ia.get(id);
            });
        //cls.def("get", &InputArchive::get);
        /*cls.def("get",
                py::overload_cast<int>(&InputArchive::get, py::const_),
                "id"_a
                );*/
    });
}

}  // namespace io
}  // namespace table
}  // namespace afw
}  // namespace lsst
