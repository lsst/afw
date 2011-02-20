#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ellipses
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>

#include "lsst/afw/geom/ellipses.h"
#include <Eigen/LU>

namespace lsst { namespace afw { namespace geom { namespace ellipses {

const static double eps = std::pow(std::numeric_limits<double>::epsilon(),0.25);

inline bool approx(double a, double b, double tol=1E-8) {
    return std::fabs(a - b) <= tol;
}

inline bool approx(BaseCore const & a, BaseCore const & b, double tol=1E-8) {
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
void invokeTest(bool no_circles) {
    TestCase::apply(Quadrupole(1.5,2.0,-0.75));
    TestCase::apply(Axes(2.5,1.3,-0.75));
    TestCase::apply(DistortionAndDeterminantRadius(0.4,-0.25,2.3));
    TestCase::apply(DistortionAndTraceRadius(0.4,-0.25,2.3));
    TestCase::apply(DistortionAndLogDeterminantRadius(0.4,-0.25,2.3));
    TestCase::apply(DistortionAndLogTraceRadius(0.4,-0.25,2.3));
    TestCase::apply(LogShearAndDeterminantRadius(0.4,-0.25,2.3));
    TestCase::apply(LogShearAndTraceRadius(0.4,-0.25,2.3));
    TestCase::apply(LogShearAndLogDeterminantRadius(0.4,-0.25,2.3));
    TestCase::apply(LogShearAndLogTraceRadius(0.4,-0.25,2.3));
    if (no_circles) return;
    TestCase::apply(Quadrupole(200.0,200.0,0.0));
    TestCase::apply(Axes(40,40,0.0));
    TestCase::apply(DistortionAndDeterminantRadius(0.0, 0.0, 2.3));
    TestCase::apply(DistortionAndTraceRadius(0.0, 0.0, 2.3));
    TestCase::apply(DistortionAndLogDeterminantRadius(0.0, 0.0, 2.3));
    TestCase::apply(DistortionAndLogTraceRadius(0.0, 0.0, 2.3));
    TestCase::apply(LogShearAndDeterminantRadius(0.0, 0.0, 2.3));
    TestCase::apply(LogShearAndTraceRadius(0.0, 0.0, 2.3));
    TestCase::apply(LogShearAndLogDeterminantRadius(0.0, 0.0, 2.3));
    TestCase::apply(LogShearAndLogTraceRadius(0.0, 0.0, 2.3));
}

template <typename T1, typename T2>
struct EllipticityConversionFunctor {
    
    static int const M = 2;
    static int const N = 2;

    Eigen::Vector2d operator()(Eigen::Vector2d const & x) {
        T1 c1(x[0], x[1]);
        T2 c2(c1);
        return Eigen::Vector2d(c2.getE1(), c2.getE2());
    }
};

struct CoreConversionTest {

    template <typename T1, typename T2>
    struct Functor {

        static int const M = 3;
        static int const N = 3;

        BaseCore::ParameterVector operator()(BaseCore::ParameterVector const & x) {
            T1 c1(x);
            T2 c2(c1);
            return c2.getParameterVector();
        }

    };

    template <typename T2, typename T1>
    static void testCoreConversion(T1 const & core) {
        T1 copy(core);
        BOOST_CHECK(approx(core, copy, 0.0));
        if (!approx(core, copy, 0.0)) {
            std::cerr << copy.getName() << "\n";
            std::cerr << core.getParameterVector().transpose() << " ---- "
                      << copy.getParameterVector().transpose() << "\n";
        }
        T2 other(core);
        copy = other;
        BOOST_CHECK(approx(core, copy, 1E-12));
        BaseCore::Jacobian a1 = copy.dAssign(other);
        BaseCore::Jacobian a2 = other.dAssign(copy);
        if (copy.getName() != "Axes" && other.getName() != "Axes") {
            Functor<T2,T1> f1;
            Functor<T1,T2> f2;
            BaseCore::Jacobian b1 = computeJacobian(f1, other.getParameterVector());
            BaseCore::Jacobian b2 = computeJacobian(f2, copy.getParameterVector());
            if (!(a1 - b1).isMuchSmallerThan(1.0, 1E-4)) {
                std::cerr << copy.getName() << ", " << other.getName() << "\n";
                std::cerr << (a1 - b1) << "\n\n";
            }
            if (!(a2 - b2).isMuchSmallerThan(1.0, 1E-4)) {
                std::cerr << other.getName() << ", " << copy.getName() << "\n";
                std::cerr << (a2 - b2) << "\n\n";
            }
            BOOST_CHECK(approx(core, copy, 1E-12));
            BOOST_CHECK((a1 * a2).isIdentity(1E-5));
            BOOST_CHECK((a2 * a1).isIdentity(1E-5));
            BOOST_CHECK((a1 - b1).isMuchSmallerThan(1.0, 1E-4));
            BOOST_CHECK((a2 - b2).isMuchSmallerThan(1.0, 1E-4));
        }
    }

    template <typename T1>
    static void apply(T1 const & core) {
        testCoreConversion<Axes>(core);
        testCoreConversion<Quadrupole>(core);
        testCoreConversion<DistortionAndDeterminantRadius>(core);
        testCoreConversion<DistortionAndTraceRadius>(core);
        testCoreConversion<DistortionAndLogDeterminantRadius>(core);
        testCoreConversion<DistortionAndLogTraceRadius>(core);
        testCoreConversion<LogShearAndDeterminantRadius>(core);
        testCoreConversion<LogShearAndTraceRadius>(core);
        testCoreConversion<LogShearAndLogDeterminantRadius>(core);
        testCoreConversion<LogShearAndLogTraceRadius>(core);
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
            return ellipse.transform(transform).copy()->getParameterVector();
        }

        Functor2(Ellipse const & ellipse_, AffineTransform const & transform_) :
            ellipse(ellipse_), transform(transform_) 
        {}

    };

    template <typename Core>
    static void apply(Core const & core) {
        Ellipse input(core, PointD(Eigen::Vector2d::Random()));
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
        Ellipse input(core, PointD(Eigen::Vector2d::Random()));
        AffineTransform output = input.getGridTransform();
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

        BaseCore::ParameterVector operator()(BaseCore::ParameterVector const & x) {
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
        BaseCore::Convolution::DerivativeMatrix d_analytic = input.convolve(other).d();
        BaseCore::Convolution::DerivativeMatrix d_numeric 
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
    afwEllipses::Distortion delta1a(0.6, -0.3);
    afwEllipses::LogShear gamma1a(delta1a);
    afwEllipses::Distortion delta2a(gamma1a);
    afwEllipses::LogShear gamma2a;
    Eigen::Matrix2d d2g_analytic = gamma2a.dAssign(delta1a);
    afwEllipses::LogShear gamma1b(-0.25, 0.55);
    afwEllipses::Distortion delta1b(gamma1b);
    afwEllipses::LogShear gamma2b(delta1b);
    afwEllipses::Distortion delta2b;
    Eigen::Matrix2d g2d_analytic = delta2b.dAssign(gamma1b);
    BOOST_CHECK(approx(delta1a, delta2a));
    BOOST_CHECK(approx(gamma1b, gamma2b));
    afwEllipses::EllipticityConversionFunctor<afwEllipses::Distortion,afwEllipses::LogShear> d2g;
    afwEllipses::EllipticityConversionFunctor<afwEllipses::LogShear,afwEllipses::Distortion> g2d;
    Eigen::Matrix2d d2g_numeric = afwEllipses::computeJacobian(d2g, Eigen::Vector2d(0.6, -0.3));
    Eigen::Matrix2d g2d_numeric = afwEllipses::computeJacobian(g2d, Eigen::Vector2d(-0.25, 0.55));
    BOOST_CHECK(d2g_numeric.isApprox(d2g_analytic, 1E-5));
    BOOST_CHECK(g2d_numeric.isApprox(g2d_analytic, 1E-5));
}

BOOST_AUTO_TEST_CASE(ParametricTest) {
    afwEllipses::Parametric p(afwEllipses::Ellipse(afwEllipses::Quadrupole(3,2,-0.65)));
    BOOST_CHECK(
        p(1.45).asEigen().isApprox(afwGeom::PointD::EigenVector(0.76537615289287353, 1.0573336496088439))
    );
    BOOST_CHECK(
        p(-2.56).asEigen().isApprox(afwGeom::PointD::EigenVector(-1.6804596457433354, 0.03378847788858419))
    );
}

BOOST_AUTO_TEST_CASE(CoreConversion) {
    afwEllipses::invokeTest<afwEllipses::CoreConversionTest>(false);
}

BOOST_AUTO_TEST_CASE(Transformer) {
    afwEllipses::invokeTest<afwEllipses::TransformerTest>(false);
}

BOOST_AUTO_TEST_CASE(GridTransform) {
    afwEllipses::invokeTest<afwEllipses::GridTransformTest>(true);
}

BOOST_AUTO_TEST_CASE(Convolution) {
    afwEllipses::invokeTest<afwEllipses::ConvolutionTest>(false);
}

BOOST_AUTO_TEST_CASE(Radii) {
    afwEllipses::DeterminantRadius gr;
    afwEllipses::LogDeterminantRadius lgr;
    lgr = gr;
    afwEllipses::TraceRadius ar;
    //ar = gr; // this line should fail to compile
}
