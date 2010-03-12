// -*- lsst-c++ -*-
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/LU>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Ellipse 

#include "boost/test/unit_test.hpp"
#include "boost/format.hpp"

#include "lsst/afw/geom/ellipses.h"

using namespace std;
namespace geom = lsst::afw::geom;
namespace ellipses = lsst::afw::geom::ellipses;

typedef geom::PointD PointD;
typedef geom::ExtentD ExtentD;
typedef geom::AffineTransform AffineTransform;
typedef ellipses::Quadrupole Quadrupole;
typedef ellipses::LogShear LogShear;
typedef ellipses::Axes Axes;
typedef ellipses::Distortion Distortion;
typedef ellipses::BaseCore BaseCore;
typedef ellipses::Quadrupole::Ellipse QuadrupoleEllipse;
typedef ellipses::LogShear::Ellipse LogShearEllipse;
typedef ellipses::Axes::Ellipse AxesEllipse;
typedef ellipses::Distortion::Ellipse DistortionEllipse;

static const double eps = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

inline bool approx(double a, double b, double tol=1E-8) {
    return std::fabs(a-b) <= tol;
}

inline bool approx(BaseCore const & a, BaseCore const & b, double tol=1E-8) {
    return approx(a[0], b[0], tol) && approx(a[1], b[1], tol) && approx(a[2], b[2], tol);
}

void testQuadrupole(Quadrupole const & core, bool =true) {
    Quadrupole copy(core);
    BOOST_CHECK(approx(core, copy, 0));

    Axes axes(core);
    copy = axes;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(axes);
    BOOST_CHECK(approx(core, copy, 1E-12));

    Distortion distortion(core);
    copy = distortion;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(distortion);
    BOOST_CHECK(approx(core, copy, 1E-12));

    LogShear log_shear(core);
    copy = log_shear;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(log_shear);
    BOOST_CHECK(approx(core, copy, 1E-12));
}

BOOST_AUTO_TEST_CASE(QuadrupoleTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testQuadrupole(Quadrupole(1.5, 2.0, -0.75));
    testQuadrupole(Quadrupole(200.0, 200.0, 0.0), false);
}

void testAxes(Axes const & core) {
    Axes copy(core);
    BOOST_CHECK(approx(core, copy, 0));

    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(quadrupole);
    BOOST_CHECK(approx(core, copy, 1E-12));
  
    Distortion distortion(core);
    copy = distortion;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(distortion);
    BOOST_CHECK(approx(core, copy, 1E-12));

    LogShear log_shear(core);
    copy = log_shear;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(log_shear);
    BOOST_CHECK(approx(core, copy, 1E-12));

}

BOOST_AUTO_TEST_CASE(AxesTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testAxes(Axes(2.5, 1.3, -0.75));
    testAxes(Axes(40, 40, 0.0));
}

void testDistortion(Distortion const & core) {
        
    Distortion copy(core);
    BOOST_CHECK(approx(core, copy, 0));
  
    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(quadrupole);
    BOOST_CHECK(approx(core, copy, 1E-12));

    Axes axes(core);
    copy = axes;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(axes);
    BOOST_CHECK(approx(core, copy, 1E-12));

    LogShear log_shear(core);
    copy = log_shear;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(log_shear);
    BOOST_CHECK(approx(core, copy, 1E-12));
}

BOOST_AUTO_TEST_CASE(DistortionTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testDistortion(Distortion(0.8, 0.3, 1.5));
    testDistortion(Distortion(0.0, 0.0, 1.0));
}

void testLogShear(LogShear const & core) {
    LogShear copy(core);
    BOOST_CHECK(approx(core, copy, 0));
  
    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(quadrupole);
    BOOST_CHECK(approx(core, copy, 1E-12));

    Axes axes(core);
    copy = axes;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(axes);
    BOOST_CHECK(approx(core, copy, 1E-12));

    Distortion distortion(core);
    copy = distortion;
    BOOST_CHECK(approx(core, copy, 1E-12));
    copy.dAssign(distortion);
    BOOST_CHECK(approx(core, copy, 1E-12));
}

BOOST_AUTO_TEST_CASE(LogShearTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testLogShear(LogShear(0.8, 0.3, 1.5));
    testLogShear(LogShear(0.0, 0.0, 1.0));
}

template <typename TCore1, typename TCore2>
void testEllipseJacobian(TCore2 core2) {
  TCore1 core1;
  Eigen::Matrix3d jac_analytic = core1.dAssign(core2);
  Eigen::Matrix3d jac_analytic_inv = core2.dAssign(core1);
  Eigen::Matrix3d jac_numeric_inv = jac_analytic.inverse();
  BOOST_CHECK_MESSAGE(
      jac_analytic_inv.isApprox(jac_numeric_inv, 1E-4),
      boost::str(
          boost::format("Jacobian inversion failure for %s -> %s") % core2.getName() % core1.getName()
      )
  );
  Eigen::Matrix3d jac_numeric;
  for (int i=0; i<3; ++i) {
      core2[i] += eps;
      core1 = core2;
      for (int j=0; j<3; ++j)
          jac_numeric(j, i) = core1[j];
      core2[i] -= 2*eps;
      core1 = core2;
      for (int j=0; j<3; ++j)
          jac_numeric(j, i) -= core1[j];
      core2[i] += eps;
      core1 = core2;
  }
  jac_numeric /= (2*eps);

  BOOST_CHECK_MESSAGE(
      jac_analytic.isApprox(jac_numeric, 1E-4),
      boost::str(
          boost::format("Jacobian computation failure for %s -> %s:\nAnalytic:\n%s\nNumeric:\n%s\n") 
          % core2.getName() % core1.getName()
          % jac_analytic % jac_numeric
      )
  );
}

BOOST_AUTO_TEST_CASE(EllipseJacobian) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testEllipseJacobian<Quadrupole>(Quadrupole(3.0, 2.0, 0.89));
    testEllipseJacobian<Quadrupole>(Axes(3.0, 2.0, 1.234));
    testEllipseJacobian<Quadrupole>(Distortion(std::complex<double>(0.5, 0.65), 2.5));
    testEllipseJacobian<Quadrupole>(LogShear(3.0, 2.0, 1.234));
    
    testEllipseJacobian<Axes>(Quadrupole(3.0, 2.0, 0.89));
    testEllipseJacobian<Axes>(Axes(3.0, 2.0, 1.234));
    testEllipseJacobian<Axes>(Distortion(std::complex<double>(0.5, 0.65), 2.5));
    testEllipseJacobian<Axes>(LogShear(3.0, 2.0, 1.234));
    
    testEllipseJacobian<Distortion>(Quadrupole(3.0, 2.0, 0.89));
    testEllipseJacobian<Distortion>(Axes(3.0, 2.0, 1.234));
    testEllipseJacobian<Distortion>(Distortion(std::complex<double>(0.5, 0.65), 2.5));
    testEllipseJacobian<Distortion>(LogShear(3.0, 2.0, 1.234));

    testEllipseJacobian<LogShear>(Quadrupole(3.0, 2.0, 0.89));
    testEllipseJacobian<LogShear>(Axes(3.0, 2.0, 1.234));
    testEllipseJacobian<LogShear>(Distortion(std::complex<double>(0.5, 0.65), 2.5));
    testEllipseJacobian<LogShear>(LogShear(3.0, 2.0, 1.234));

}

template <typename TCore>
void testEllipseTransformer(TCore core) {
    typename TCore::Ellipse input(core, PointD(Eigen::Vector2d::Random()));
    Eigen::Matrix2d linear(Eigen::Matrix2d::Random());
    AffineTransform transform(linear);
    typename TCore::Ellipse output(*input.transform(transform).copy());
    Eigen::Matrix<double, 5, 5> e_d_analytic = input.transform(transform).d();
    Eigen::Matrix<double, 5, 5> e_d_numeric = Eigen::Matrix<double, 5, 5>::Zero();
    
    for (int i=0; i<5; ++i) {
        input[i] += eps;
        output = *input.transform(transform).copy();
        for (int j=0; j<5; ++j)
            e_d_numeric(j, i) = output[j];
        input[i] -= 2*eps;
        output = *input.transform(transform).copy();
        for (int j=0; j<5; ++j)
            e_d_numeric(j, i) -= output[j];
        input[i] += eps;
    }
    e_d_numeric /= (2*eps);
    BOOST_CHECK_MESSAGE(
        e_d_analytic.isApprox(e_d_numeric, 1E-4),
        boost::str(
            boost::format("Transformer::d failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
            % core.getName() % e_d_analytic % e_d_numeric
        )
    );

    Eigen::Matrix<double, 5, 6> t_d_analytic = input.transform(transform).dTransform();
    Eigen::Matrix<double, 5, 6> t_d_numeric = Eigen::Matrix<double, 5, 6>::Zero();
    for (int i=0; i<6; ++i) {
        transform[i] += eps;
        output = *input.transform(transform).copy();
        for (int j=0; j<5; ++j)
            t_d_numeric(j, i) = output[j];
        transform[i] -= 2*eps;
        output = *input.transform(transform).copy();
        for (int j=0; j<5; ++j)
            t_d_numeric(j, i) -= output[j];
        transform[i] += eps;
    }
    t_d_numeric /= (2*eps);
    
    BOOST_CHECK_MESSAGE(
        t_d_analytic.isApprox(t_d_numeric, 1E-4),
        boost::str(
            boost::format("Transformer::dTransform failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
            % core.getName() % t_d_analytic % t_d_numeric
        )
    );
}

BOOST_AUTO_TEST_CASE(EllipseTransformDerivative) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    testEllipseTransformer<Quadrupole>(Quadrupole(3.0, 2.0, 0.89));
    testEllipseTransformer<Axes>(Axes(3.0, 2.0, 1.234));
    testEllipseTransformer<Distortion>(Distortion(std::complex<double>(0.5, 0.65), 2.5));
    testEllipseTransformer<LogShear>(LogShear(3.0, 2.0, 1.234));
}

template <typename TCore>
void testRadialFraction(TCore const & input) {
    typename TCore::Ptr ptr(input.clone());
    TCore & core = *ptr;
    BaseCore::RadialFraction rf(core);
    PointD p = PointD::make(1.25, 0.85);
    ExtentD epsX = ExtentD::make(eps, 0.0);
    ExtentD epsY = ExtentD::make(0.0, eps);
    Eigen::RowVector2d grad_analytic = rf.d(p);
    Eigen::RowVector2d grad_numeric;
    grad_numeric << (rf(p+epsX) - rf(p-epsX)) / (2*eps), (rf(p+epsY) - rf(p-epsY)) / (2*eps);
    BOOST_CHECK(grad_analytic.isApprox(grad_numeric, 1E-4));
    Eigen::RowVector3d jac_analytic = rf.dCore(p);
    Eigen::RowVector3d jac_numeric;
    for (int i = 0; i<3; ++i) {
        core[i] += eps;
        rf = BaseCore::RadialFraction(core);
        jac_numeric[i] = rf(p);
        core[i] -= 2*eps;
        rf = BaseCore::RadialFraction(core);
        jac_numeric[i] -= rf(p);
        core[i] += eps;
        rf = BaseCore::RadialFraction(core);
        jac_numeric[i] /= (2*eps);
    }
    BOOST_CHECK(jac_analytic.isApprox(jac_numeric, 1E-4));
}

BOOST_AUTO_TEST_CASE(RadialFractionTest) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    BaseCore::RadialFraction rf(Quadrupole(1.5, 2.0, -0.75));
    BOOST_CHECK(approx(rf(PointD::make(-6.75, 2.375)), 5.5669008088329193, 1E-8));
    BOOST_CHECK(approx(rf(PointD::make(3.25, -4.375)), 3.4198702929369733, 1E-8));
    testRadialFraction(Quadrupole(3.0, 2.0, 0.89));
    testRadialFraction(Axes(3.0, 2.0, 1.234));
    testRadialFraction(Distortion(0.5, 0.65, 2.5));
    testRadialFraction(LogShear(3.0, 2.0, 1.234));
}
