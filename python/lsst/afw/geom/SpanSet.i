
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

%template (SpanVector) std::vector<lsst::afw::geom::Span>;
%{
#include "lsst/afw/geom/SpanSet.h"
%}
%shared_ptr(lsst::afw::geom::SpanSet);

%warnfilter(509) lsst::afw::geom::SpanSet;

%include "lsst/afw/geom/SpanSet.h"

%ignore lsst::afw::geom::SpanSet::begin;
%ignore lsst::afw::geom::SpanSet::cbegin;
%ignore lsst::afw::geom::SpanSet::end;
%ignore lsst::afw::geom::SpanSet::cend;


%template(flattenI) lsst::afw::geom::SpanSet::flatten<int, 1>;
%template(flattenIOutParam) lsst::afw::geom::SpanSet::flatten<int, 1, 2>;
%template(flattenD) lsst::afw::geom::SpanSet::flatten<double, 1>;
%template(flattenDOutParam) lsst::afw::geom::SpanSet::flatten<double, 1, 2>;

%template(unflattenI) lsst::afw::geom::SpanSet::unflatten<int, 1>;
%template(unflattenIOutParam) lsst::afw::geom::SpanSet::unflatten<int, 2, 1>;
%template(unflattenD) lsst::afw::geom::SpanSet::unflatten<double, 1>;
%template(unflattenDOutParam) lsst::afw::geom::SpanSet::unflatten<double, 2, 1>;

%template(setMaskI) lsst::afw::geom::SpanSet::setMask<unsigned short>;

%template(clearMaskI) lsst::afw::geom::SpanSet::clearMask<unsigned short>;



%newobject lsst::afw::geom::SpanSet::__iter__;
%extend lsst::afw::geom::SpanSet {

    %fragment("SwigPyIterator_T");

    swig::SwigPyIterator * __iter__(PyObject **PYTHON_SELF) {
        return swig::make_output_iterator(self->begin(), self->begin(), self->end(), *PYTHON_SELF);
    }

    %pythoncode %{
    def __len__(self):
        return self.size()
    %}
}
