// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2014 LSST Corporation.
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
#ifndef AFW_TABLE_FunctorKey_h_INCLUDED
#define AFW_TABLE_FunctorKey_h_INCLUDED

#include "lsst/afw/table/fwd.h"
#include "lsst/afw/table/Key.h"

namespace lsst { namespace afw { namespace table {

template <typename T>
class OutputFunctorKey {
public:

    virtual T get(BaseRecord const & record) const = 0;

    virtual ~OutputFunctorKey() {}

};

template <typename T>
class InputFunctorKey {
public:

    virtual void set(BaseRecord & record, T const & value) const = 0;

    virtual ~InputFunctorKey() {}

};

template <typename T>
class FunctorKey : public OutputFunctorKey<T>, public InputFunctorKey<T> {};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_FunctorKey_h_INCLUDED
