// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include "lsst/afw/geom/ellipses/GridTransform.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Separable.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/radii.h"

namespace lsst { namespace afw { namespace geom {
namespace ellipses {

BaseCore::GridTransform::GridTransform(BaseCore const & input) :
    _input(input),
    _eig(Quadrupole(input).getMatrix())
{}

LinearTransform::Matrix BaseCore::GridTransform::getMatrix() const {
    return _eig.operatorInverseSqrt();
}

BaseCore::GridTransform::operator LinearTransform () const {
    return LinearTransform(_eig.operatorInverseSqrt());
}

BaseCore::GridTransform::DerivativeMatrix
BaseCore::GridTransform::d() const {
    /*
       Grid transform is easiest to differentiate in the ReducedShear/DeterminantRadius parametrization.
       But we actually differentiate the inverse of the transform, and then use
       $dM^{-1}/dt = -M^{-1} dM/dt M^{-1} to compute the derivative of the inverse.

       The inverse of the grid transform in ReducedShear/DeterminantRadius is:
       $\frac{r}{\sqrt{1-g^2}}(\sigma_x + g_1 \sigma_z + g2 \sigma_y)$, where $\sigma_i$ are the
       Pauli spin matrices.
    */
    typedef Separable<ReducedShear,DeterminantRadius> C;
    C core;
    Jacobian rhs = core.dAssign(_input);
    double g1 = core.getE1();
    double g2 = core.getE2();
    double g = core.getEllipticity().getE();
    double r = core.getRadius();
    double beta = 1.0 - g*g;
    double alpha = r / std::sqrt(beta);

    Eigen::Matrix2d sigma_z, sigma_y;
    sigma_z <<
        1.0, 0.0,
        0.0,-1.0;
    sigma_y <<
        0.0, 1.0,
        1.0, 0.0;
    Eigen::Matrix2d t = _eig.operatorSqrt();
    Eigen::Matrix2d tInv = _eig.operatorInverseSqrt();
    Eigen::Matrix2d dt_dg1 = t * g1 / beta + alpha * sigma_z;
    Eigen::Matrix2d dt_dg2 = t * g2 / beta + alpha * sigma_y;
    Eigen::Matrix2d dt_dr = t * (1.0 / r);
    Eigen::Matrix2d dtInv_dg1 = -tInv * dt_dg1 * tInv;
    Eigen::Matrix2d dtInv_dg2 = -tInv * dt_dg2 * tInv;
    Eigen::Matrix2d dtInv_dr = -tInv * dt_dr * tInv;

    GridTransform::DerivativeMatrix mid;
    mid(LinearTransform::XX, C::E1) = dtInv_dg1(0,0);
    mid(LinearTransform::XY, C::E1) = mid(LinearTransform::YX, C::E1) = dtInv_dg1(0,1);
    mid(LinearTransform::YY, C::E1) = dtInv_dg1(1,1);
    mid(LinearTransform::XX, C::E2) = dtInv_dg2(0,0);
    mid(LinearTransform::XY, C::E2) = mid(LinearTransform::YX, C::E2) = dtInv_dg2(0,1);
    mid(LinearTransform::YY, C::E2) = dtInv_dg2(1,1);
    mid(LinearTransform::XX, C::RADIUS) = dtInv_dr(0,0);
    mid(LinearTransform::XY, C::RADIUS) = mid(LinearTransform::YX, C::RADIUS) = dtInv_dr(0,1);
    mid(LinearTransform::YY, C::RADIUS) = dtInv_dr(1,1);
    return mid * rhs;
}

double BaseCore::GridTransform::getDeterminant() const {
    return sqrt(1.0 / _eig.eigenvalues().prod());
}

LinearTransform BaseCore::GridTransform::invert() const {
    return LinearTransform(_eig.operatorSqrt());
}

Ellipse::GridTransform::GridTransform(Ellipse const & input) : _input(input), _coreGt(input.getCore()) {}

AffineTransform::Matrix Ellipse::GridTransform::getMatrix() const {
    AffineTransform::Matrix r = AffineTransform::Matrix::Zero();
    r.block<2,2>(0,0) = _coreGt.getMatrix();
    r.block<2,1>(0,2) = -r.block<2,2>(0,0) * _input.getCenter().asEigen();
    r(2,2) = 1.0;
    return r;
}

Ellipse::GridTransform::DerivativeMatrix 
Ellipse::GridTransform::d() const {
    DerivativeMatrix r = DerivativeMatrix::Zero();
    LinearTransform linear = _coreGt;
    r.block<4,3>(0,0) = _coreGt.d();
    double x = -_input.getCenter().getX();
    double y = -_input.getCenter().getY();
    r(AffineTransform::X, Ellipse::X) = -linear[LinearTransform::XX];
    r(AffineTransform::Y, Ellipse::X) = -linear[LinearTransform::YX];
    r(AffineTransform::X, Ellipse::Y) = -linear[LinearTransform::XY];
    r(AffineTransform::Y, Ellipse::Y) = -linear[LinearTransform::YY];
    r(AffineTransform::X, 0) = x * r(AffineTransform::XX, 0) + y * r(AffineTransform::XY, 0);
    r(AffineTransform::Y, 0) = x * r(AffineTransform::YX, 0) + y * r(AffineTransform::YY, 0);
    r(AffineTransform::X, 1) = x * r(AffineTransform::XX, 1) + y * r(AffineTransform::XY, 1);
    r(AffineTransform::Y, 1) = x * r(AffineTransform::YX, 1) + y * r(AffineTransform::YY, 1);
    r(AffineTransform::X, 2) = x * r(AffineTransform::XX, 2) + y * r(AffineTransform::XY, 2);
    r(AffineTransform::Y, 2) = x * r(AffineTransform::YX, 2) + y * r(AffineTransform::YY, 2);
    return r;
}

double Ellipse::GridTransform::getDeterminant() const {
    return _coreGt.getDeterminant();
}

Ellipse::GridTransform::operator AffineTransform () const {
    LinearTransform linear = _coreGt;
    return AffineTransform(linear, linear(Point2D() - _input.getCenter()));
}

}}}} // namespace lsst::afw::geom::ellipses
