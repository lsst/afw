#if !defined(LSST_AFW_MATH_DETAIL_INTERPOLATE_TAUT)
#define LSST_AFW_MATH_DETAIL_INTERPOLATE_TAUT
#include <cmath>
#include <vector>
#include "lsst/afw/math/Interpolate.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

#if !defined(SWIG)
/**
 * \brief A class that adapts the SDSS spline code (such as the taut splines) to Interpolate
 *
 * Rather than rewrite the SDSS code ab initio, it seemed simpler to provide this class which
 * isn't visible in the external interface (but it has to be in the header as InterpolateTautSpline
 * inherits from it)
 */
class InterpolateSdssSpline : public Interpolate {
public:
    /// dtor
    virtual ~InterpolateSdssSpline() {}
    virtual double interpolate(double const x) const;
    virtual void interpolate(std::vector<double> const& x, std::vector<double> &y) const;
    virtual double derivative(double const x) const;
    virtual void derivative(std::vector<double>  const& x, std::vector<double> &dydx) const;
    virtual std::vector<double> roots(double const value, double const x0, double const x1) const;
protected:
    /// \brief ctor
    InterpolateSdssSpline(std::vector<double> const &x, ///< the ordinates of points
                          std::vector<double> const &y ///< the values at x[]
                         ) : Interpolate(x, y) {}

    void _allocateSpline(int const nknot);
    std::vector<double> _knots;                ///< positions of knots
    std::vector<std::vector<double> > _coeffs; ///< and associated coefficients
};
#endif

class InterpolateControlTautSpline;
    
/**
 * \brief Taut Splines
 *
 * Adapted from
 *      A Practical Guide to Splines
 * by
 *      C. de Boor (N.Y. : Springer-Verlag, 1978)
 * (His routine tautsp converted to C by Robert Lupton and then to C++ by an older and grayer Robert Lupton)
 *
 * If gamma > 0, additional knots are introduced where needed to
 * make the interpolant more flexible locally. This avoids extraneous
 * inflection points typical of cubic spline interpolation at knots to
 * rapidly changing data. Values for gamma are:
 *
 * \li         = 0: no additional knots
 * \li         in [0, 3): under certain conditions on the given data at
 *                points i-1, i, i+1, and i+2, a knot is added in the
 *                i-th interval, i=1,...,ntau-3. see description of method
 *                below. the interpolant gets rounded with increasing
 *                gamma. A value of  2.5  for gamma is typical.
 *
 * \li         in [3,6): as for gamma in [0, 3), except that knots might also be added in
 *                intervals in which an inflection point would be permitted. The value
 *                is then decremented by 3 --- i.e. into [0, 3)
 *                A value of 5.5 for gamma is typical (i.e. 2.5 after this decrement)
 *
 * \note This spline appears to return an interpolant that's almost identical to the Akima spline, but has
 * extra methods provided by InterpolateSdssSpline, and handles even/odd functions
 *
 * Method:
 * \verbatim
On the i-th interval, (tau[i], tau[i+1]), the interpolant is of the form
     f(u(x)) = a + b*u + c*h(u, z) + d*h(1 - u, 1 - z) ,            (##)
with u = u(x) = (x - tau[i])/dtau[i]
Here,
     z = z(i) = addg(i + 1)/(addg(i) + addg(i + 1))   (= 1/2 if the denominator vanishes)
     ddg(j)   = dg(j + 1) - dg(j),
     addg(j)  = abs(ddg(j)),
     dg(j)    = divdif(j) = (gtau[j + 1] - gtau[j])/dtau[j]
and
     h(u,z)   = alpha*u**3 + (1 - alpha)*max((u - zeta)/(1 - zeta), 0)**3
with
     alpha(z) = (1 - gamma/3)/zeta
     zeta(z)  = 1 - gamma*min((1 - z), 1/3)
thus, for 1/3 <= z <= 2/3, f is just a cubic polynomial on the interval i.
Otherwise, it has one additional knot, at
       tau[i] + zeta*dtau[i]
As z approaches 1, h(.,z) has an increasingly sharp bend near 1, thus allowing f to turn rapidly near the
additional knot.

In terms of f(j) = gtau[j] and fsecnd[j] = 2*derivative of f at tau[j], the coefficients in (##) are given by
     a = f(i) - d
     b = (f(i + 1) - f(i)) - (c - d)
     c = fsecnd[i + 1]*dtau[i]**2/hsecnd(1, z)           # RHL: what's hsecnd?
     d = fsecnd[i]*dtau[i]**2/hsecnd(1, 1 - z)
hence can be computed once fsecnd[i],i=0,...,ntau-1, is fixed.
\endverbatim
 *
 *  f is automatically continuous and has a continuous second derivative
 * (except when z = 0 or 1 for some i). We determine fscnd from
 * the requirement that also the first derivative of f be continuous.
 * in addition, we require that the third derivative be continuous
 * across  tau[1] and across tau[ntau-2] . this leads to a strictly
 * diagonally dominant tridiagonal linear system for the fsecnd[i]
 * which we solve by gauss elimination without pivoting.
 *
 *  There must be at least 4 interpolation points for us to fit a taut
 * cubic spline, but if you provide fewer we'll fit a quadratic or linear
 * polynomial (but you must provide at least 2)
 */
class InterpolateTautSpline
#if !defined(SWIG)
    : public InterpolateSdssSpline
#endif
{
    friend PTR(Interpolate) math::makeInterpolate(std::vector<double> const &x, std::vector<double> const &y,
                                                  InterpolateControl const& ictrl);
public:
    /// Specify the desired symmetry for the spline
    enum Symmetry { UNKNOWN,            ///< No particular symmetry
                    ODD,                ///< Spline interpolation will obey y(-x) = -y(x)
                    EVEN                ///< Spline interpolation will obey y(-x) =  y(x)
    };
    /// dtor
    virtual ~InterpolateTautSpline() {}
private:
    InterpolateTautSpline(std::vector<double> const &x, std::vector<double> const &y,
                          InterpolateControlTautSpline const& ictrl);

    void _calculateTautSpline(std::vector<double> const& x,
                             std::vector<double> const& y,
                             double const gamma0);
    void _calculateTautSplineEvenOdd(std::vector<double> const& x,
                                     std::vector<double> const& y,
                                     double const gamma0,
                                     bool even);
    PTR(InterpolateControlTautSpline) _ictrl;
};
    
/*****************************************************************************/
/**
 * \brief Control the creation of an InterpolateTautSpline
 *
 */
class InterpolateControlTautSpline : public InterpolateControl {
    friend class InterpolateTautSpline;
public:
    InterpolateControlTautSpline(float gamma=0.0,
                                 InterpolateTautSpline::Symmetry symmetry=InterpolateTautSpline::UNKNOWN);
private:
    float _gamma;                     ///< How taut should I be?
    InterpolateTautSpline::Symmetry _symmetry; ///< Desired symmetry
};
    
}}}}
#endif
