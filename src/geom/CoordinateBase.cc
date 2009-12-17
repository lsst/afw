#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace geom = lsst::afw::geom;

/**
 *  \brief Floating-point comparison with tolerance.
 *  
 *  Interface, naming, and default tolerances matches Numpy.
 *
 *  \relatesalso CoordinateBase
 */
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

template bool geom::allclose<geom::Point2D,double,2>(
    CoordinateBase<geom::Point2D,double,2> const &,
    CoordinateBase<geom::Point2D,double,2> const &,
    double, double
);
template bool geom::allclose<geom::Point3D,double,3>(
    CoordinateBase<geom::Point3D,double,3> const &,
    CoordinateBase<geom::Point3D,double,3> const &,
    double, double
);
template bool geom::allclose<geom::Extent2D,double,2>(
    CoordinateBase<geom::Extent2D,double,2> const &,
    CoordinateBase<geom::Extent2D,double,2> const &,
    double, double
);
template bool geom::allclose<geom::Extent3D,double,3>(
    CoordinateBase<geom::Extent3D,double,3> const &,
    CoordinateBase<geom::Extent3D,double,3> const &,
    double, double
);
