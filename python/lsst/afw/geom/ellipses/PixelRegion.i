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
#include "lsst/afw/geom/ellipses/PixelRegion.h"
%}

%import "lsst/afw/geom/Span.i"

#if 0 // this doesn't work because of a SWIG bug related to the fact that we've done %shared_ptr(Span)

%include "std_vector.i"
// We don't actually need this vector, but we do need SWIG to generate
// the traits classes needed to support iterators that yield Spans.
// Presumably there's a more minimal way to do that, but it's the
// opposite of well-documented.
%template(SpanVector) std::vector<lsst::afw::geom::Span>;

%newobject lsst::afw::geom::ellipses::__iter__;
%extend lsst::afw::geom::ellipses::PixelRegion {

    %fragment("SwigPyIterator_T");

    // This is the same recipe used to support the STL iterators in SWIG.
    swig::SwigPyIterator * __iter__(PyObject **PYTHON_SELF) {
        return swig::make_output_iterator(self->begin(), self->begin(), self->end(), *PYTHON_SELF);
    }

}

#else // So here's more workaround for the SWIG bug, where we re-implement some of its iterator stuff
      // without templates.

%{
#include "lsst/afw/geom/ellipses/PyPixelRegion.h"
%}
%include "lsst/afw/geom/ellipses/PyPixelRegion.h"

%extend PyPixelRegionIterator {
%pythoncode %{
def __next__(self):
    if self.atEnd():
        raise StopIteration()
    current = self.get()
    self.increment()
    return current

next = __next__

def __iter__(self):
    return self
%}
}

%newobject lsst::afw::geom::ellipses::PixelRegion::__iter__;
%extend lsst::afw::geom::ellipses::PixelRegion {

    PyPixelRegionIterator * __iter__(PyObject **PYTHON_SELF) const {
        return new PyPixelRegionIterator(self->begin(), self->end(), *PYTHON_SELF);
    }

}

#endif // workaround for SWIG bug

%returnCopy(lsst::afw::geom::ellipses::PixelRegion::getBBox)

%include "lsst/afw/geom/ellipses/PixelRegion.h"
