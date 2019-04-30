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

#include "lsst/afw/table/io/FitsSchemaInputMapper.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace io {

PYBIND11_MODULE(fits, mod) {

    mod.def("setNRowsToPrep", [](std::size_t n) { FitsSchemaInputMapper::N_ROWS_TO_PREP = n; });
    mod.def("getNRowsToPrep", []() { return FitsSchemaInputMapper::N_ROWS_TO_PREP; });

}

}
}
}
}  // namespace lsst::afw::table::io