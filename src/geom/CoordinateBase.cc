#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace geom = lsst::afw::geom;

template <typename Derived, typename T, int N>
bool geom::allclose(
    CoordinateBase<Derived,T,N> const & a,
    CoordinateBase<Derived,T,N> const & b, 
    T rtol, T atol
) {
    Eigen::Matrix<T,N,1> diff = (a.asVector() - b.asVector()).cwise().abs();
    Eigen::Matrix<T,N,1> rhs = (0.5*(a.asVector() + b.asVector())).cwise().abs();
    rhs *= rtol;
    rhs.cwise() += atol;
    return (diff.cwise() <= rhs).all();
}

template bool geom::allclose(
    CoordinateBase<Point2D,double,2> const &,
    CoordinateBase<Point2D,double,2> const &,
    double, double
);
template bool geom::allclose(
    CoordinateBase<Point3D,double,3> const &,
    CoordinateBase<Point3D,double,3> const &,
    double, double
);
template bool geom::allclose(
    CoordinateBase<Extent2D,double,2> const &,
    CoordinateBase<Extent2D,double,2> const &,
    double, double
);
template bool geom::allclose(
    CoordinateBase<Extent3D,double,3> const &,
    CoordinateBase<Extent3D,double,3> const &,
    double, double
);
