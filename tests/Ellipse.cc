#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Ellipse 

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"
#include "boost/format.hpp"

#include "lsst/afw/math/ellipses.h"

using namespace std;
namespace math = lsst::afw::math;
namespace ellipses = lsst::afw::math::ellipses;

typedef ellipses::Quadrupole Quadrupole;
typedef ellipses::LogShear LogShear;
typedef ellipses::Axes Axes;
typedef ellipses::Distortion Distortion;
typedef ellipses::RadialFraction RadialFraction;
typedef ellipses::Core Core;
typedef ellipses::Parametric Parametric;
typedef ellipses::Quadrupole::Ellipse QuadrupoleEllipse;
typedef ellipses::LogShear::Ellipse LogShearEllipse;
typedef ellipses::Axes::Ellipse AxesEllipse;
typedef ellipses::Distortion::Ellipse DistortionEllipse;

const static double eps = std::pow(std::numeric_limits<double>::epsilon(),0.25);

inline bool approx(double a, double b, double tol=1E-8) {
    return std::fabs(a-b) <= tol;
}

inline bool approx(Core const & a, Core const & b, double tol=1E-8) {
    return approx(a[0],b[0],tol) && approx(a[1],b[1],tol) && approx(a[2],b[2],tol);
}

void testQuadrupole(Quadrupole const & core, bool test_rf=true) {
  Quadrupole copy(core);
  BOOST_CHECK(approx(core,copy,0));
  
  Axes axes(core);
  copy = axes;
  BOOST_CHECK(approx(core,copy,1E-12));
  copy.differentialAssign(axes);
  BOOST_CHECK(approx(core,copy,1E-12));

  Distortion distortion(core);
  copy = distortion;
  BOOST_CHECK(approx(core,copy,1E-12));
  copy.differentialAssign(distortion);
  BOOST_CHECK(approx(core,copy,1E-12));

  LogShear log_shear(core);
  copy = log_shear;
  BOOST_CHECK(approx(core,copy,1E-12));
  copy.differentialAssign(log_shear);
  BOOST_CHECK(approx(core,copy,1E-12));

  RadialFraction rf(copy);

  if (test_rf) {
      BOOST_CHECK(approx(rf(math::Coordinate(-6.75,2.375)),std::pow(5.5669008088329193,2),1E-8));
      BOOST_CHECK(approx(rf(math::Coordinate(3.25,-4.375)),std::pow(3.4198702929369733,2),1E-8));
  }

  math::Coordinate p(1.25,0.85);
  Eigen::RowVector2d grad_analytic = rf.differentiateCoordinate(p);
  Eigen::RowVector2d grad_numeric;
  grad_numeric << (rf(math::Coordinate(p.x()+eps,p.y()))-rf(math::Coordinate(p.x()-eps,p.y())))/(2*eps),
      (rf(math::Coordinate(p.x(),p.y()+eps))-rf(math::Coordinate(p.x(),p.y()-eps)))/(2*eps);
  BOOST_CHECK(grad_analytic.isApprox(grad_numeric,1E-4));

  Eigen::RowVector3d jac_analytic = rf.differentiateCore(p);
  Eigen::RowVector3d jac_numeric;
  copy[Quadrupole::IXX] += eps; rf = RadialFraction(copy);
  jac_numeric[0] = rf(p);
  copy[Quadrupole::IXX] -= 2*eps; rf = RadialFraction(copy);
  jac_numeric[0] -= rf(p);
  copy[Quadrupole::IXX] += eps; rf = RadialFraction(copy);
  jac_numeric[0] /= (2*eps);
  copy[Quadrupole::IYY] += eps; rf = RadialFraction(copy);
  jac_numeric[1] = rf(p);
  copy[Quadrupole::IYY] -= 2*eps; rf = RadialFraction(copy);
  jac_numeric[1] -= rf(p);
  copy[Quadrupole::IYY] += eps; rf = RadialFraction(copy);
  jac_numeric[1] /= (2*eps);
  copy[Quadrupole::IXY] += eps; rf = RadialFraction(copy);
  jac_numeric[2] = rf(p);
  copy[Quadrupole::IXY] -= 2*eps; rf = RadialFraction(copy);
  jac_numeric[2] -= rf(p);
  copy[Quadrupole::IXY] += eps; rf = RadialFraction(copy);
  jac_numeric[2] /= (2*eps);
  BOOST_CHECK(jac_analytic.isApprox(jac_numeric,1E-4));
}

BOOST_AUTO_TEST_CASE(run_test_Quadrupole) {
    testQuadrupole(Quadrupole(1.5,2.0,-0.75));
    testQuadrupole(Quadrupole(200.0,200.0,0.0),false);
}

void testAxes(Axes const & core) {
    Axes copy(core);
    BOOST_CHECK(approx(core,copy,0));

    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(quadrupole);
    BOOST_CHECK(approx(core,copy,1E-12));
  
    Distortion distortion(core);
    copy = distortion;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(distortion);
    BOOST_CHECK(approx(core,copy,1E-12));

    LogShear log_shear(core);
    copy = log_shear;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(log_shear);
    BOOST_CHECK(approx(core,copy,1E-12));

}

BOOST_AUTO_TEST_CASE(AxesTest) {
    testAxes(Axes(2.5,1.3,-0.75));
    testAxes(Axes(40,40,0.0));
}

void testDistortion(Distortion const & core) {
        
    Distortion copy(core);
    BOOST_CHECK(approx(core,copy,0));
  
    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(quadrupole);
    BOOST_CHECK(approx(core,copy,1E-12));

    Axes axes(core);
    copy = axes;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(axes);
    BOOST_CHECK(approx(core,copy,1E-12));

    LogShear log_shear(core);
    copy = log_shear;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(log_shear);
    BOOST_CHECK(approx(core,copy,1E-12));
}

BOOST_AUTO_TEST_CASE(DistortionTest) {
    testDistortion(Distortion(0.8,0.3,1.5));
    testDistortion(Distortion(0.0,0.0,1.0));
}

void testLogShear(LogShear const & core) {
    LogShear copy(core);
    BOOST_CHECK(approx(core,copy,0));
  
    Quadrupole quadrupole(core);
    copy = quadrupole;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(quadrupole);
    BOOST_CHECK(approx(core,copy,1E-12));

    Axes axes(core);
    copy = axes;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(axes);
    BOOST_CHECK(approx(core,copy,1E-12));

    Distortion distortion(core);
    copy = distortion;
    BOOST_CHECK(approx(core,copy,1E-12));
    copy.differentialAssign(distortion);
    BOOST_CHECK(approx(core,copy,1E-12));
}

BOOST_AUTO_TEST_CASE(LogShearTest) {
    testLogShear(LogShear(0.8,0.3,1.5));
    testLogShear(LogShear(0.0,0.0,1.0));
}

BOOST_AUTO_TEST_CASE(ParametricTest) {
    Parametric p(QuadrupoleEllipse(Quadrupole(3,2,-0.65)));
    BOOST_CHECK(p(1.45).isApprox(math::Coordinate(0.76537615289287353, 1.0573336496088439)));
    BOOST_CHECK(p(-2.56).isApprox(math::Coordinate(-1.6804596457433354, 0.03378847788858419)));
}

template <typename TCore1, typename TCore2>
void testEllipseJacobian(TCore2 core2) {
  TCore1 core1;
  Eigen::Matrix3d jac_analytic = core1.differentialAssign(core2);
  Eigen::Matrix3d jac_analytic_inv = core2.differentialAssign(core1);
  Eigen::Matrix3d jac_numeric_inv = jac_analytic.inverse();
  BOOST_CHECK_MESSAGE(
      jac_analytic_inv.isApprox(jac_numeric_inv,1E-4),
      (boost::format("Jacobian inversion failure for %s -> %s") % core2.getName() % core1.getName()).str()
  );
  Eigen::Matrix3d jac_numeric;
  for (int i=0; i<3; ++i) {
      core2[i] += eps;
      core1 = core2;
      for (int j=0; j<3; ++j)
          jac_numeric(j,i) = core1[j];
      core2[i] -= 2*eps;
      core1 = core2;
      for (int j=0; j<3; ++j)
          jac_numeric(j,i) -= core1[j];
      core2[i] += eps;
      core1 = core2;
  }
  jac_numeric /= (2*eps);

  BOOST_CHECK_MESSAGE(
      jac_analytic.isApprox(jac_numeric,1E-4),
      boost::str(
          boost::format("Jacobian computation failure for %s -> %s:\nAnalytic:\n%s\nNumeric:\n%s\n") 
          % core2.getName() % core1.getName()
          % jac_analytic % jac_numeric
      )
  );
}

BOOST_AUTO_TEST_CASE(EllipseJacobian) {
    testEllipseJacobian<Quadrupole>(Quadrupole(3.0,2.0,0.89));
    testEllipseJacobian<Quadrupole>(Axes(3.0,2.0,1.234));
    testEllipseJacobian<Quadrupole>(Distortion(std::complex<double>(0.5,0.65),2.5));
    testEllipseJacobian<Quadrupole>(LogShear(3.0,2.0,1.234));
    
    testEllipseJacobian<Axes>(Quadrupole(3.0,2.0,0.89));
    testEllipseJacobian<Axes>(Axes(3.0,2.0,1.234));
    testEllipseJacobian<Axes>(Distortion(std::complex<double>(0.5,0.65),2.5));
    testEllipseJacobian<Axes>(LogShear(3.0,2.0,1.234));
    
    testEllipseJacobian<Distortion>(Quadrupole(3.0,2.0,0.89));
    testEllipseJacobian<Distortion>(Axes(3.0,2.0,1.234));
    testEllipseJacobian<Distortion>(Distortion(std::complex<double>(0.5,0.65),2.5));
    testEllipseJacobian<Distortion>(LogShear(3.0,2.0,1.234));

    testEllipseJacobian<LogShear>(Quadrupole(3.0,2.0,0.89));
    testEllipseJacobian<LogShear>(Axes(3.0,2.0,1.234));
    testEllipseJacobian<LogShear>(Distortion(std::complex<double>(0.5,0.65),2.5));
    testEllipseJacobian<LogShear>(LogShear(3.0,2.0,1.234));

}

template <typename TCore>
void testEllipseTransformDerivative(TCore core) {
    typename TCore::Ellipse input(core, math::Coordinate::Random());
    math::AffineTransform transform(Eigen::Matrix2d::Random(),Eigen::Vector2d::Random());
    typename TCore::Ellipse output = input;
    output.transform(transform);
    ellipses::Ellipse::TransformDerivative td(input,transform);
    Eigen::Matrix<double,5,5> e_d_analytic = td.dInput();
    Eigen::Matrix<double,5,5> e_d_numeric = Eigen::Matrix<double,5,5>::Zero();
    
    for (int i=0; i<5; ++i) {
        input[i] += eps;
        output = input;
        output.transform(transform);
        for (int j=0; j<5; ++j)
            e_d_numeric(j,i) = output[j];
        input[i] -= 2*eps;
        output = input;
        output.transform(transform);
        for (int j=0; j<5; ++j)
            e_d_numeric(j,i) -= output[j];
        input[i] += eps;
    }
    e_d_numeric /= (2*eps);
    BOOST_CHECK_MESSAGE(
        e_d_analytic.isApprox(e_d_numeric,1E-4),
        boost::str(
            boost::format("TransformDerivative::dInput failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
            % core.getName() % e_d_analytic % e_d_numeric
        )
    );

    Eigen::Matrix<double,5,6> t_d_analytic = td.dTransform();
    Eigen::Matrix<double,5,6> t_d_numeric = Eigen::Matrix<double,5,6>::Zero();
    for (int i=0; i<6; ++i) {
        transform[i] += eps;
        output = input;
        output.transform(transform);
        for (int j=0; j<5; ++j)
            t_d_numeric(j,i) = output[j];
        transform[i] -= 2*eps;
        output = input;
        output.transform(transform);
        for (int j=0; j<5; ++j)
            t_d_numeric(j,i) -= output[j];
        transform[i] += eps;
    }
    t_d_numeric /= (2*eps);
    
    BOOST_CHECK_MESSAGE(
        t_d_analytic.isApprox(t_d_numeric,1E-4),
        boost::str(
            boost::format("TransformDerivative::dTransform failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
            % core.getName() % t_d_analytic % t_d_numeric
        )
    );
}

BOOST_AUTO_TEST_CASE(EllipseTransformDerivative) {
    testEllipseTransformDerivative<Quadrupole>(Quadrupole(3.0,2.0,0.89));
    testEllipseTransformDerivative<Axes>(Axes(3.0,2.0,1.234));
    testEllipseTransformDerivative<Distortion>(Distortion(std::complex<double>(0.5,0.65),2.5));
    testEllipseTransformDerivative<LogShear>(LogShear(3.0,2.0,1.234));
}

