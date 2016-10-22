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

#ifndef LSST_AFW_TABLE_IO_PYBIND11_H
#define LSST_AFW_TABLE_IO_PYBIND11_H

#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace afw {
namespace table {
namespace io {

    /**
     * @brief Wraps an instantiation of @ref PersistableFacade.
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
    void declarePersistableFacade(pybind11::module & module, std::string const & suffix) {
        using namespace pybind11::literals;
        // shared_ptr is used by subclasses of PersistableFacade
        pybind11::class_<PersistableFacade<T>, std::shared_ptr<PersistableFacade<T>>>
                clsFacade(module, ("PersistableFacade" + suffix).c_str());
        clsFacade.def_static("readFits",
                             (PTR(T) (*)(std::string const &, int)) &PersistableFacade<T>::readFits,
                             "fileName"_a, "hdu"_a=0);
    }

}}}}     // lsst::afw::table::io

#endif

