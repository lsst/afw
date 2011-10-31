// -*- c++ -*-
#ifndef CATALOG_Point_h_INCLUDED
#define CATALOG_Point_h_INCLUDED

namespace lsst { namespace catalog {

namespace point {

enum { X=0, Y=1 };

} // namespace point

template <typename T>
struct Point {
    T x;
    T y;

    Point(T x_, T y_) : x(x_), y(y_) {}
};

}} // namespace lsst::catalog

#endif // !CATALOG_Point_h_INCLUDED
