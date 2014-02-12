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

%include "lsst/p_lsstSwig.i"
%include "lsst/afw/utils.i"
%include "lsst/afw/geom/Point.i"

%{
#include "lsst/afw/geom/Polygon.h"
%}

// Using typemaps because "%template(VectorFoo) std::vector<Foo>" requires a default constructor for Foo
// but Polygon doesn't have one.
// An alternative is:
//     %ignore std::vector<lsst::afw::geom::Polygon>::vector;
//     %ignore std::vector<lsst::afw::geom::Polygon>::resize;
//     %template(VectorPolygon) std::vector<lsst::afw::geom::Polygon>;
// However, this makes VectorPolygon a separate SWIG object, rather than a list which is easily manipulated.
%typemap(typecheck) std::vector<lsst::afw::geom::Polygon>& {
    $1 = PySequence_Check($input) ? 1 : 0;
}
%typemap(in) std::vector<lsst::afw::geom::Polygon>& (std::vector<lsst::afw::geom::Polygon> mapped) {
    PyObject* seq = PySequence_Fast($input, "expected a sequence");
    size_t len = PySequence_Fast_GET_SIZE(seq);
    for (size_t i = 0; i < len; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        void* vPoly = 0;
        int res = SWIG_ConvertPtr(item, &vPoly, $descriptor(lsst::afw::geom::Polygon*), 0);
        if (!SWIG_IsOK(res)) {
            SWIG_exception_fail(SWIG_ArgError(res), "while converting Polygon");
        }
        if (!vPoly) {
            SWIG_exception_fail(SWIG_ValueError, "while converting Polygon");
        }
        lsst::afw::geom::Polygon* poly = reinterpret_cast<lsst::afw::geom::Polygon*>(vPoly);
        mapped.push_back(*poly);
    }
    $1 = &mapped;
}
%typemap(out) std::vector<lsst::afw::geom::Polygon> {
    $result = PyList_New($1.size());
    for (size_t i = 0; i < $1.size(); ++i) {
        PyList_SetItem($result, i,
                       SWIG_NewPointerObj(new lsst::afw::geom::Polygon($1.operator[](i)),
                                          $descriptor(lsst::afw::geom::Polygon*), SWIG_POINTER_OWN));
    }
}

%include "lsst/afw/geom/Polygon.h"

%template(VectorPoint) std::vector<lsst::afw::geom::Point2D>;
%template(PairPoint) std::pair<lsst::afw::geom::Point2D, lsst::afw::geom::Point2D>;
%template(VectorPairPoint) std::vector<std::pair<lsst::afw::geom::Point2D, lsst::afw::geom::Point2D> >;

%useValueEquality(lsst::afw::geom::Polygon);
%definePythonIterator(lsst::afw::geom::Polygon);
%ignore lsst::afw::geom::Polygon::swap(lsst::afw::geom::Polygon&); // not needed in python

%extend lsst::afw::geom::Polygon {
%pythoncode %{
    union = union_ # Because this is python, not C++

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, [p for p in self.getVertices()])
    def __reduce__(self):
        return self.__class__, (self.getVertices(),)

    def __iter__(self):
        """Iterator over vertices"""
        vertices = self.getVertices()
        return iter(vertices)
    def __len__(self):
        return self.getNumEdges()

    def __contains__(self, point):
        """point in polygon?"""
        return self.contains(point)

    def display(self, xy0=None, frame=1, ctype=None):
        """Display polygon on existing frame in ds9"""
        import lsst.afw.geom as afwGeom
        import lsst.afw.display.ds9 as ds9
        xy0 = afwGeom.Extent2D(0,0) if xy0 is None else afwGeom.Extent2D(xy0)
        with ds9.Buffering():
            for p1, p2 in self.getEdges():
                ds9.line((p1 - xy0, p2 - xy0), frame=frame, ctype=ctype)

    def plot(self, axes=None, **kwargs):
        """Plot polygon with matplotlib

        @param axes: Matplotlib axes to use, or None
        @param kwargs: Additional arguments for plotting (e.g., color, line type)
        @return Matplotlib axes
        """
        import numpy
        if axes is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = plt.axes()
        for p1, p2 in self.getEdges():
            x = (p1.getX(), p2.getX())
            y = (p1.getY(), p2.getY())
            axes.plot(x, y, **kwargs)
        vertices = self.getVertices()
        x = numpy.array([p[0] for p in vertices])
        y = numpy.array([p[1] for p in vertices])
        axes.scatter(x, y, **kwargs)
        return axes
%}
}
%extend std::vector<lsst::afw::geom::Point2D> {
%pythoncode %{
    def __reduce__(self):
        return self.__class__, ([p for p in self],)
%}
}

