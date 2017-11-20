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

namespace lsst {
namespace afw {
namespace table {
namespace io {
namespace python {

/**
 * Add Python readFits overrides that work with hidden subclasses.
 *
 * Most classes that inherit from PersistableFacade do not need to do anything
 * to expose its interface to Python, and PersistableFacade itself should *never*
 * be included in the list of base classes exposed to pybind11.
 *
 * The only time this function is needed is when some subclasses of the class
 * being wrapped are directly wrapped (i.e. they are constructed by a factory,
 * and only available via the base class interface).  In all other cases,
 * pybind11 should be able to do the downcasting itself.
 *
 * @param  The pybind11 class wrapper to add methods to.
 */
template <typename T, typename ...E>
void declarePersistableFacade(pybind11::class_<T, E...> & cls) {
    using namespace pybind11::literals;
    cls.def_static("readFits",
                   (std::shared_ptr<T>(*)(std::string const &, int)) &PersistableFacade<T>::readFits,
                   "fileName"_a, "hdu"_a = INT_MIN);
    cls.def_static("readFits",
                   (std::shared_ptr<T>(*)(fits::MemFileManager &, int)) &PersistableFacade<T>::readFits,
                   "manager"_a, "hdu"_a = INT_MIN);
}

}
}
}
}
}  // lsst::afw::table::io::python

#endif
