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

%{
#include "lsst/afw/image/ApCorrMap.h"
%}

%include "std_map.i"
%import "lsst/afw/table/io/ioLib.i"

// Using %import to bring this in causes circular dependency problems, and it seems.
// not to be necessary as long as we tell Swig we're using shared_ptr with it.
%shared_ptr(lsst::afw::math::BoundedField);

%declareTablePersistable(ApCorrMap, lsst::afw::image::ApCorrMap);

%rename(__getitem__) lsst::afw::image::ApCorrMap::operator[];
%include "lsst/afw/image/ApCorrMap.h"

%extend lsst::afw::image::ApCorrMap {

    // I couldn't figure out how to get Swig to expose the C++ iterators using std_map.i and (e.g.)
    // code in utils.i; it kept wrapping the iterators as opaque objects I couldn't deference, probably
    // due to some weirdness with the typedefs.  I don't want to sink time into debugging that.
    // So I just wrote this function to return a list of names, and I'll base the Python iterators on
    // that.
    PyObject * keys() const {
        PyObject * r = PyList_New(self->size());
        Py_ssize_t n = 0;
        for (lsst::afw::image::ApCorrMap::Iterator i = self->begin(); i != self->end(); ++i, ++n) {
            PyList_SET_ITEM(r, n, PyBytes_FromStringAndSize(i->first.data(), i->first.size()));
        }
        return r;
    }

    %pythoncode %{

        def values(self):
            return [self[name] for name in self.keys()]

        def items(self):
            return [(name, self[name]) for name in self.keys()]

        def __iter__(self):
            return iter(self.keys())

        def __setitem__(self, name, value):
            return self.set(name, value)

        def __contains__(self, name):
            return self.get(name) is not None

        def __len__(self):
            return self.size()

        #
        # Provide return value for C++ "void operator op=()" or it will magically end up as None
        #
        def __imul__(*args):
            """__imul__(self, double scale) -> self"""
            _imageLib.ApCorrMap___imul__(*args)
            return args[0]
    
        def __idiv__(*args):
            """__idiv__(self, double scale) -> self"""
            _imageLib.ApCorrMap___idiv__(*args)
            return args[0]
%}
}
