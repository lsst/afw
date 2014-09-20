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

/*
 * Wrappers for FunctorKeys that map arrays (ndarray/NumPy) to consecutive fields of the same type
 */

%include "lsst/afw/table/aggregates.i"

%{
#include "lsst/afw/table/arrays.h"
%}

%template(KeyFVector) std::vector< lsst::afw::table::Key<float> >;
%template(KeyDVector) std::vector< lsst::afw::table::Key<double> >;

%define %declareArrayKey1(SUFFIX, T)
%declareFunctorKey(Array ## SUFFIX, ndarray::Array<T const,1,1>)
%declareReferenceFunctorKey(Array ## SUFFIX, ndarray::ArrayRef<T,1,1>)
%declareConstReferenceFunctorKey(Array ## SUFFIX, ndarray::ArrayRef<T const,1,1>)
%shared_ptr(lsst::afw::table::ArrayKey<T>)
%declareNumPyConverters(ndarray::Array<T const,1,1>);
%declareNumPyConverters(ndarray::Array<T,1,1>);
%extend lsst::afw::table::ArrayKey<T> {
lsst::afw::table::Key<T> _get(int i) {
    return (*self)[i];
}
ndarray::Array<T,1,1> getReference(lsst::afw::table::BaseRecord & record) {
    return self->getReference(record).shallow();
}
%pythoncode %{
def __getitem__(self, index):
    if isinstance(index, slice):
        start, stop, stride = index.indices(self.getSize())
        if stride != 1:
            raise IndexError("Non-unit stride not supported")
        return self.slice(start, stop)
    return self._get(index)
%}
}
%enddef

%define %declareArrayKey2(SUFFIX, T)
%template(Array ## SUFFIX ## Key) lsst::afw::table::ArrayKey<T>;
%useValueEquality(lsst::afw::table::ArrayKey<T>)
%enddef

%declareArrayKey1(F, float)
%declareArrayKey1(D, double)

%include "lsst/afw/table/arrays.h"

%declareArrayKey2(F, float)
%declareArrayKey2(D, double)
