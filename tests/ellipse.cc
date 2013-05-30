// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ellipses
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/format.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/afw/geom/ellipses.h"
#include "Eigen/LU"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

const static double eps = std::pow(std::numeric_limits<double>::epsilon(),0.5);

inline bool approx(double a, double b, double tol=1E-8) {
    return std::fabs(a - b) <= tol;
}

inline bool approx(EllipseCore const & a, EllipseCore const & b, double tol=1E-8) {
    return a.getParameterVector().isApprox(b.getParameterVector(), tol);
}

inline bool approx(detail::EllipticityBase const & a, detail::EllipticityBase const & b, double tol=1E-8) {
    return Eigen::Vector2d(a.getE1(), a.getE2()).isApprox(Eigen::Vector2d(b.getE1(), b.getE2()), tol);
}

template <typename Function>
Eigen::Matrix<double,Function::M,Function::N>
computeJacobian(Function f, Eigen::Matrix<double,Function::N,1> const & initial) {
    Eigen::Matrix<double,Function::N,1> x(initial);
    Eigen::Matrix<double,Function::M,Function::N> result;
    for (int n = 0; n < Function::N; ++n) {
        x[n] += eps;
        result.col(n) = f(x);
        x[n] -= 2.0 * eps;
        result.col(n) -= f(x);
        x[n] += eps;
        result.col(n) /= (2.0 * eps);
    }
    return result;
}

template <typename TestCase>
void invokeCoreTest() {
    TestCase::apply(Quadrupole(1.5,2.0,-0.75));
    TestCase::apply(Axes(2.5,1.3,-0.75*radians));
    TestCase::apply(Separable<Distortion>(0.4,-0.25,2.3));
    TestCase::apply(Separable<ConformalShear>(0.4,-0.25,2.3));
    TestCase::apply(Separable<ReducedShear>(0.4,-0.25,2.3));

    TestCase::apply(Quadrupole(200.0,200.0,0.0));
    TestCase::apply(Axes(40,40));
    TestCase::apply(Separable<Distortion>(0.0, 0.0, 2.3));
    TestCase::apply(Separable<ConformalShear>(0.0, 0.0, 2.3));
    TestCase::apply(Separable<ReducedShear>(0.0, 0.0, 2.3));
}

template <typename TestCase>
void invokeEllipticityTest() {
    TestCase::apply(Distortion(0.6, -0.3));
    TestCase::apply(ReducedShear(0.35, -0.12));
    TestCase::apply(ConformalShear(0.23, -0.31));
    TestCase::apply(Distortion(0.0, 0.0));
    TestCase::apply(ReducedShear(0.0, 0.0));
    TestCase::apply(ConformalShear(0.0, 0.0));
}

struct EllipticityConversionTest {

    template <typename T1, typename T2>
    struct Functor {

        static int const M = 2;
        static int const N = 2;

        Eigen::Vector2d operator()(Eigen::Vector2d const & x) {
            T1 c1(x[0], x[1]);
            T2 c2(c1);
            return Eigen::Vector2d(c2.getE1(), c2.getE2());
        }
    };

    template <typename T2, typename T1>
    static void testEllipticityConversion(T1 const & orig) {
        T1 copy(orig);
        BOOST_CHECK(approx(orig, copy, 0.0));
        if (!approx(orig, copy, 0.0)) {
            std::cerr << copy.getName() << "\n";
            std::cerr << orig.getComplex() << " ---- " << copy.getComplex() << "\n";
        }
        T2 other(orig);
        copy = other;
        BOOST_CHECK(approx(orig, copy, 1E-12));
        typename T1::Jacobian a1 = copy.dAssign(other);
        typename T1::Jacobian a2 = other.dAssign(copy);
        Functor<T2,T1> f1;
        Functor<T1,T2> f2;
        typename T1::Jacobian b1 = computeJacobian(f1, Eigen::Vector2d(other.getE1(), other.getE2()));
        typename T1::Jacobian b2 = computeJacobian(f2, Eigen::Vector2d(copy.getE1(), copy.getE2()));
        if (!(a1 - b1).isMuchSmallerThan(1.0, 1E-4)) {
            std::cerr << copy.getName() << " <- " << other.getName() << "\n";
            std::cerr << "Input: " << other.getComplex() << "\n";
            std::cerr << "Output: " << copy.getComplex() << "\n";
            std::cerr << "Analytic:\n" << a1 << "\n";
            std::cerr << "Numeric:\n" << b1 << "\n\n";
        }
        if (!(a2 - b2).isMuchSmallerThan(1.0, 1E-4)) {
            std::cerr << other.getName() << " <- " << copy.getName() << "\n";
            std::cerr << "Input: " << copy.getComplex() << "\n";
            std::cerr << "Output: " << other.getComplex() << "\n";
            std::cerr << "Analytic:\n" << a2 << "\n";
            std::cerr << "Numeric:\n" << b2 << "\n\n";
        }
        BOOST_CHECK(approx(orig, copy, 1E-12));
        BOOST_CHECK((a1 * a2).isIdentity(1E-5));
        BOOST_CHECK((a2 * a1).isIdentity(1E-5));
        BOOST_CHECK((a1 - b1).isMuchSmallerThan(1.0, 1E-4));
        BOOST_CHECK((a2 - b2).isMuchSmallerThan(1.0, 1E-4));

    }

    template <typename T1>
    static void apply(T1 const & core) {
        testEllipticityConversion<Distortion>(core);
        testEllipticityConversion<ReducedShear>(core);
        testEllipticityConversion<ConformalShear>(core);
    }

};

struct TransformerTest {

    struct Functor1 {

        static int const M = 5;
        static int const N = 5;

        Ellipse ellipse;
        AffineTransform transform;

        Eigen::Matrix<double,5,1> operator()(Eigen::Matrix<double,5,1> const & x) {
            ellipse.setParameterVector(x);
            ellipse.transform(transform).inPlace();
            return ellipse.getParameterVector();
        }

        Functor1(Ellipse const & ellipse_, AffineTransform const & transform_) :
            ellipse(ellipse_), transform(transform_)
        {}

    };

    struct Functor2 {

        static int const M = 5;
        static int const N = 6;

        Ellipse ellipse;
        AffineTransform transform;

        Eigen::Matrix<double,5,1> operator()(Eigen::Matrix<double,6,1> const & x) {
            transform.setParameterVector(x);
            return ellipse.transform(transform).copy().getParameterVector();
        }

        Functor2(Ellipse const & ellipse_, AffineTransform const & transform_) :
            ellipse(ellipse_), transform(transform_)
        {}

    };

    template <typename Core>
    static void apply(Core const & core) {
        Ellipse input(core, Point2D(Eigen::Vector2d::Random()));
        Eigen::Matrix2d tm;
        tm <<
            -0.2704311, 0.9044595,
            0.0268018, 0.8323901;
        AffineTransform transform(tm, Eigen::Vector2d::Random());
        Functor1 f1(input, transform);
        Functor2 f2(input, transform);
        Ellipse output = input.transform(transform);
        Eigen::Matrix<double,5,5> e_d_analytic = input.transform(transform).d();
        Eigen::Matrix<double,5,5> e_d_numeric = computeJacobian(f1, input.getParameterVector());
        BOOST_CHECK_MESSAGE(
            e_d_analytic.isApprox(e_d_numeric,1E-4),
            boost::str(
                boost::format("Transformer::d failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
                % core.getName() % e_d_analytic % e_d_numeric
            )
        );
        Eigen::Matrix<double,5,6> t_d_analytic = input.transform(transform).dTransform();
        Eigen::Matrix<double,5,6> t_d_numeric = computeJacobian(f2, transform.getParameterVector());
        BOOST_CHECK_MESSAGE(
            t_d_analytic.isApprox(t_d_numeric,1E-4),
            boost::str(
                boost::format("Transformer::dTransform failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
                % core.getName() % t_d_analytic % t_d_numeric
            )
        );
    }

};

struct GridTransformTest {

    struct Functor {

        static int const M = 6;
        static int const N = 5;

        Ellipse ellipse;

        Eigen::Matrix<double,M,1> operator()(Eigen::Matrix<double,N,1> const & x) {
            ellipse.setParameterVector(x);
            return AffineTransform(ellipse.getGridTransform()).getParameterVector();
        }

        Functor(Ellipse const & ellipse_) : ellipse(ellipse_) {}

    };


    template <typename T>
    static void apply(T const & core) {
        Ellipse input(core, Point2D(Eigen::Vector2d::Random()));
        AffineTransform output = input.getGridTransform();
        BOOST_CHECK_MESSAGE(
            output.getMatrix().isApprox(input.getGridTransform().getMatrix(), 1E-12),
            boost::str(
                boost::format("GridTransform::getMatrix incorrect %s:\ngetMatrix:\n%s\nTransform:\n%s\n")
                % core.getName() % input.getGridTransform().getMatrix() % output.getMatrix()
            )
        );
        BOOST_CHECK_CLOSE(output.getLinear().getMatrix()(0,1), output.getLinear().getMatrix()(1,0), 1E-8);
        Ellipse unit_circle = input.transform(output);
        Axes unit_circle_axes(unit_circle.getCore());
        BOOST_CHECK_CLOSE(unit_circle_axes.getA(), 1.0, 1E-8);
        BOOST_CHECK_CLOSE(unit_circle_axes.getB(), 1.0, 1E-8);
        Functor f(input);
        Ellipse::GridTransform::DerivativeMatrix d_analytic = input.getGridTransform().d();
        Ellipse::GridTransform::DerivativeMatrix d_numeric
            = computeJacobian(f, input.getParameterVector());
        BOOST_CHECK_MESSAGE(
            d_analytic.isApprox(d_numeric,1E-4),
            boost::str(
                boost::format("GridTransform::d failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
                % core.getName() % d_analytic % d_numeric
            )
        );

    }

};

struct ConvolutionTest {

    template <typename T>
    struct Functor {

        static int const M = 3;
        static int const N = 3;

        EllipseCore::ParameterVector operator()(EllipseCore::ParameterVector const & x) {
            T c1(x);
            c1.convolve(other).inPlace();
            return c1.getParameterVector();
        }

        Quadrupole other;

        Functor(Quadrupole const & other_) : other(other_) {}

    };

    template <typename T>
    static void apply(T core) {
        Quadrupole other(1.2, 0.8, -0.25);
        T input(core);
        T output = input.convolve(other);
        Functor<T> f(other);
        EllipseCore::Convolution::DerivativeMatrix d_analytic = input.convolve(other).d();
        EllipseCore::Convolution::DerivativeMatrix d_numeric
            = computeJacobian(f, input.getParameterVector());
        BOOST_CHECK_MESSAGE(
            d_analytic.isApprox(d_numeric,1E-4),
            boost::str(
                boost::format("Convolve::d failed for %s:\nAnalytic:\n%s\nNumeric:\n%s\n")
                % core.getName() % d_analytic % d_numeric
            )
        );

    }
};

}}}} // namespace lsst::afw::geom::ellipses

namespace afwEllipses = lsst::afw::geom::ellipses;
namespace afwGeom = lsst::afw::geom;

BOOST_AUTO_TEST_CASE(EllipticityTest) {
    afwEllipses::invokeEllipticityTest<afwEllipses::EllipticityConversionTest>();
}

BOOST_AUTO_TEST_CASE(ParametricTest) {
    afwEllipses::Parametric p(afwEllipses::Ellipse(afwEllipses::Quadrupole(3,2,-0.65)));
    BOOST_CHECK(
        p(1.45).asEigen().isApprox(afwGeom::Point2D::EigenVector(0.76537615289287353, 1.0573336496088439))
    );
    BOOST_CHECK(
        p(-2.56).asEigen().isApprox(afwGeom::Point2D::EigenVector(-1.6804596457433354, 0.03378847788858419))
    );
}

BOOST_AUTO_TEST_CASE(Transformer) {
    afwEllipses::invokeCoreTest<afwEllipses::TransformerTest>();
}

BOOST_AUTO_TEST_CASE(GridTransform) {
    afwEllipses::invokeCoreTest<afwEllipses::GridTransformTest>();
}

BOOST_AUTO_TEST_CASE(Convolution) {
    afwEllipses::invokeCoreTest<afwEllipses::ConvolutionTest>();
}
