/*
 * LSST Data Management System
 * Copyright 2014 LSST Corporation.
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

%include "std_pair.i"
%include "std_vector.i"

%{
#include "lsst/afw/geom/CartesianPolygon.h"
%}

// Using typemaps because "%template(VectorFoo) std::vector<Foo>" requires a default constructor for Foo
// but CartesianPolygon doesn't have one.
%typemap(typecheck) std::vector<lsst::afw::geom::CartesianPolygon>& {
    $1 = PySequence_Check($input) ? 1 : 0;
}
%typemap(in) std::vector<lsst::afw::geom::CartesianPolygon>& (std::vector<lsst::afw::geom::CartesianPolygon> mapped) {
    PyObject* seq = PySequence_Fast($input, "expected a sequence");
    size_t len = PySequence_Fast_GET_SIZE(seq);
    for (size_t i = 0; i < len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        void* vPoly = 0;
        int res = SWIG_ConvertPtr(item, &vPoly, $descriptor(lsst::afw::geom::CartesianPolygon*), 0);
        if (!SWIG_IsOK(res)) {
            SWIG_exception_fail(SWIG_ArgError(res), "while converting CartesianPolygon");
        }
        if (!vPoly) {
            SWIG_exception_fail(SWIG_ValueError, "while converting CartesianPolygon");
        }
        lsst::afw::geom::CartesianPolygon* poly =
            reinterpret_cast<lsst::afw::geom::CartesianPolygon*>(vPoly);
        mapped.push_back(*poly);
    }
    $1 = &mapped;
}
%typemap(out) std::vector<lsst::afw::geom::CartesianPolygon> {
    $result = PyList_New($1.size());
    for (size_t i = 0; i < $1.size(); ++i) {
        PyList_SetItem($result, i,
                       SWIG_NewPointerObj(new lsst::afw::geom::CartesianPolygon($1.operator[](i)),
                                          $descriptor(lsst::afw::geom::CartesianPolygon*),
                                          SWIG_POINTER_OWN | 0 ));
    }
}

%include "lsst/afw/geom/CartesianPolygon.h"

%template(VectorPoint) std::vector<lsst::afw::geom::Point2D>;
%template(PairPoint) std::pair<lsst::afw::geom::Point2D, lsst::afw::geom::Point2D>;
%template(VectorPairPoint) std::vector<std::pair<lsst::afw::geom::Point2D, lsst::afw::geom::Point2D> >;
%template(VectorWcs) std::vector<CONST_PTR(lsst::afw::image::Wcs)>;

%extend lsst::afw::geom::CartesianPolygon {
%pythoncode %{
    union = union_ # Because this is python, not C++

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, [p for p in self.getVertices()])
    def __reduce__(self):
        return self.__class__, (self.getVertices(),)

    def display(self, xy0=None, frame=1, ctype=None):
        """Display polygon on existing frame in ds9"""
        import lsst.afw.geom as afwGeom
        import lsst.afw.display.ds9 as ds9
        xy0 = afwGeom.Extent2D(0,0) if xy0 is None else afwGeom.Extent2D(xy0)
        with ds9.Buffering():
            for p1, p2 in self.getEdges():
                ds9.line((p1 - xy0, p2 - xy0), frame=frame, ctype=ctype)
%}
}
%extend std::vector<lsst::afw::geom::Point2D> {
%pythoncode %{
    def __reduce__(self):
        return self.__class__, ([p for p in self],)
%}
}

