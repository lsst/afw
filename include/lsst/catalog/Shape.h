// -*- c++ -*-
#ifndef CATALOG_Shape_h_INCLUDED
#define CATALOG_Shape_h_INCLUDED

namespace lsst { namespace catalog {

namespace shape {

enum { XX=0, YY=1, XY=2 };

} // namespace point

template <typename T>
struct Shape {
    T xx;
    T yy;
    T xy;

    Shape(T xx_, T yy_, T xy_) : xx(xx_), yy(yy_), xy(xy_) {}
};

}} // namespace lsst::catalog

#endif // !CATALOG_Shape_h_INCLUDED
