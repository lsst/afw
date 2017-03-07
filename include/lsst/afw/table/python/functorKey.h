#ifndef AFW_TABLE_PYBIND11_FUNCTORKEY_H_INCLUDED
#define AFW_TABLE_PYBIND11_FUNCTORKEY_H_INCLUDED
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

#include "pybind11/pybind11.h"

#include "lsst/afw/table/FunctorKey.h"

namespace lsst {
namespace afw {
namespace table {
namespace python {

/**
Declare OutputFunctorKey<T>, InputFunctorKey<T> and FunctorKey<T> bases for a given type T.

@tparam T key type, e.g. IcrsCoord, FluxResult

@param[in] mod pybind11 module.
@param[in] suffix suffix for class name in Python.
*/
template <typename T>
void declareFunctorKeys(pybind11::module & mod, std::string const & suffix) {
    pybind11::class_<OutputFunctorKey<T>, std::shared_ptr<OutputFunctorKey<T>>>
        clsOutputFunctorKey(mod, ("OutputFunctorKey"+suffix).c_str());
    pybind11::class_<InputFunctorKey<T>, std::shared_ptr<InputFunctorKey<T>>>
        clsInputFunctorKey(mod, ("InputFunctorKey"+suffix).c_str());
    pybind11::class_<FunctorKey<T>, std::shared_ptr<FunctorKey<T>>, OutputFunctorKey<T>, InputFunctorKey<T>>
        clsFunctorKey(mod, ("FunctorKey"+suffix).c_str());
    
};

}}}}  // namespace lsst::afw::table::python
#endif
