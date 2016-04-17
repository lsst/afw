// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
#include "lsst/afw/geom/Span.h"
%}
%shared_ptr(lsst::afw::geom::Span);

%warnfilter(509) lsst::afw::geom::Span;

%include "lsst/afw/geom/Span.h"

%ignore lsst::afw::geom::Span::begin;
%ignore lsst::afw::geom::Span::end;

%newobject lsst::afw::geom::Span::__iter__;
%extend lsst::afw::geom::Span {

    %fragment("SwigPyIterator_T");

    // This is the same recipe used to support the STL iterators in SWIG.
    swig::SwigPyIterator * __iter__(PyObject **PYTHON_SELF) {
        return swig::make_output_iterator(self->begin(), self->begin(), self->end(), *PYTHON_SELF);
    }

    %pythoncode %{

    def __len__(self):
        return self.getWidth()

    def __str__(self):
        """Print this Span"""
        return self.toString()
    %}
}
