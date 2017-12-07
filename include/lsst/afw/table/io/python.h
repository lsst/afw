// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_TABLE_IO_PYTHON_H
#define LSST_AFW_TABLE_IO_PYTHON_H

#include "pybind11/pybind11.h"

#include <memory>
#include <string>

#include "lsst/afw/fits.h"
#include "lsst/afw/table/io/Persistable.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace table {
namespace io {
namespace python {

/**
 * Wraps an instantiation of @ref PersistableFacade.
 *
 * @deprecated Use addPersistableMethods for all new code.
 *
 * Pybind11 shall assume that `PersistableFacade` is managed using
 * `std::shared_ptr`, as this is required for compatibility with
 * existing subclasses of `PersistableFacade`. This means that wrapping
 * will only work if new classes also use `std::shared_ptr` as their
 * holder type.
 *
 * @tparam T The type of object this `PersistableFacade` is for.
 *
 * @param module The pybind11 module that shall contain `PersistableFacade<T>`
 * @param suffix A string to disambiguate this class from other
 *               `PersistableFacades`. The Python name of this class shall be
 *               `PersistableFacade<suffix>`.
 */
template <typename T>
void declarePersistableFacade(pybind11::module &module, std::string const &suffix) {
    using namespace pybind11::literals;

    pybind11::class_<PersistableFacade<T>, std::shared_ptr<PersistableFacade<T>>> cls(
            module, ("PersistableFacade" + suffix).c_str());
    cls.def_static("readFits",
                   (std::shared_ptr<T>(*)(std::string const &, int)) & PersistableFacade<T>::readFits,
                   "fileName"_a, "hdu"_a = INT_MIN);
    cls.def_static("readFits",
                   (std::shared_ptr<T>(*)(fits::MemFileManager &, int)) & PersistableFacade<T>::readFits,
                   "manager"_a, "hdu"_a = INT_MIN);
}

/**
 * Add table::io::Persistable and PersistableFacade methods to the pybind11 wrapper for a class
 *
 * Use this instead of declarePersistableFacade to avoid circular import issues in Python;
 * it allows your class to be used without importing lsst.afw.table.
 *
 * Use as follows:
 * - When declaring the pybind11 class that wraps your Class do *not* list
 *   table::io::PersistableFacade<Class> and table::io::Persistable as subclasses.
 * - Call this function to wrap the methods that make your object persistable.
 */
template <typename Class, typename PyClass>
void addPersistableMethods(PyClass &cls) {
    cls.def_static("readFits",
                   (std::shared_ptr<Class>(*)(std::string const &, int)) & PersistableFacade<Class>::readFits,
                   "fileName"_a, "hdu"_a = INT_MIN);
    cls.def_static(
            "readFits",
            (std::shared_ptr<Class>(*)(fits::MemFileManager &, int)) & PersistableFacade<Class>::readFits,
            "manager"_a, "hdu"_a = INT_MIN);
    cls.def("writeFits", (void (Class::*)(std::string const &, std::string const &) const) & Class::writeFits,
            "fileName"_a, "mode"_a = "w");
    cls.def("writeFits",
            (void (Class::*)(fits::MemFileManager &, std::string const &) const) & Class::writeFits,
            "manager"_a, "mode"_a = "w");
    cls.def("isPersistable", &Class::isPersistable);
}
}
}
}
}
}  // lsst::afw::table::io::python

#endif
