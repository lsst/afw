#if !defined(LSST_AFW_MATH_DETAIL_SPLINE)
#define LSST_AFW_MATH_DETAIL_SPLINE 1
#include <cmath>
#include <vector>

namespace lsst {
namespace afw {
namespace math {
namespace detail {

/*
 * Splines
 */
class Spline {
public:
    virtual ~Spline() = default;

    Spline(Spline const &) = default;
    Spline(Spline &&) = default;
    Spline & operator=(Spline const &) = default;
    Spline & operator=(Spline &&) = default;

    /**
     * Interpolate a Spline.
     *
     * @param[in] x points to interpolate at
     * @param[out] y values of spline interpolation at x
     */
    void interpolate(std::vector<double> const& x, std::vector<double>& y) const;
    /**
     * Find the derivative of a Spline.
     *
     * @param[in] x points to evaluate derivative at
     * @param[out] dydx derivatives at x
     */
    void derivative(std::vector<double> const& x, std::vector<double>& dydx) const;

    /**
     * Find the roots of
     *   Spline - val = 0
     * in the range [x0, x1). Return a vector of all the roots found
     *
     * @param value desired value
     * @param x0, x1 specify desired range is [x0,x1)
     */
    std::vector<double> roots(double const value, double const x0, double const x1) const;

protected:
    Spline() = default;
    /**
     * Allocate the storage a Spline needs
     */
    void _allocateSpline(int const nknot);

    std::vector<double> _knots;                 // positions of knots
    std::vector<std::vector<double> > _coeffs;  // and associated coefficients
};

class TautSpline : public Spline {
public:
    enum Symmetry { Unknown, Odd, Even };

    /**
     * Construct cubic spline interpolant to given data.
     *
     * Adapted from <i>A Practical Guide to Splines</i> by C. de Boor (N.Y. : Springer-Verlag, 1978).
     * (His routine tautsp converted to C by Robert Lupton and then to C++ by an older and grayer Robert
     * Lupton)
     *
     * If `gamma` > 0, additional knots are introduced where needed to
     * make the interpolant more flexible locally. This avoids extraneous
     * inflection points typical of cubic spline interpolation at knots to
     * rapidly changing data. Values for gamma are:
     *
     * - = 0, no additional knots
     * - in (0, 3), under certain conditions on the given data at
     *                points i-1, i, i+1, and i+2, a knot is added in the
     *                i-th interval, i=1,...,ntau-3. See notes
     *                below. The interpolant gets rounded with increasing
     *                gamma. A value of  2.5  for gamma is typical.
     * - in (3, 6), same as above, except that knots might also be added in
     *                intervals in which an inflection point would be permitted.
     *                A value of  5.5  for gamma is typical.
     *
     * @param x points where function's specified
     * @param y values of function at `tau[]`
     * @param gamma control extra knots. See main description for details.
     * @param type specify the desired symmetry (e.g. Even)
     *
     * @throws pex::exceptions::InvalidParameterError Thrown if `x` and `y` do not have
     *              the same length or do not have at least two points
     *
     * @note on the i-th interval, (tau[i], tau[i+1]), the interpolant is of the
     *  form
     *  (*)  f(u(x)) = a + b*u + c*h(u,z) + d*h(1-u,1-z) ,
     *  with u = u(x) = (x - tau[i])/dtau[i].  Here,
     *       z = z(i) = addg(i+1)/(addg(i) + addg(i+1))
     *  (= .5, in case the denominator vanishes). with
     *       ddg(j) = dg(j+1) - dg(j),
     *       addg(j) = abs(ddg(j)),
     *       dg(j) = divdif(j) = (gtau[j+1] - gtau[j])/dtau[j]
     *  and
     *       h(u,z) = alpha*u**3 + (1 - alpha)*(max(((u-zeta)/(1-zeta)),0)**3
     *  with
     *       alpha(z) = (1-gamma/3)/zeta
     *       zeta(z) = 1 - gamma*min((1 - z), 1/3)
     *  thus, for 1/3 .le. z .le. 2/3,  f  is just a cubic polynomial on
     *  the interval i. otherwise, it has one additional knot, at
     *         tau[i] + zeta*dtau[i] .
     *  as  z  approaches  1, h(.,z) has an increasingly sharp bend  near 1,
     *  thus allowing  f  to turn rapidly near the additional knot.
     *     in terms of f(j) = gtau[j] and
     *       fsecnd[j] = 2*derivative of f at tau[j],
     *  the coefficients for (*) are given as
     *       a = f(i) - d
     *       b = (f(i+1) - f(i)) - (c - d)
     *       c = fsecnd[i+1]*dtau[i]**2/hsecnd(1,z)
     *       d = fsecnd[i]*dtau[i]**2/hsecnd(1,1-z)
     *  hence can be computed once fsecnd[i],i=0,...,ntau-1, is fixed.
     *
     * @note f  is automatically continuous and has a continuous second derivat-
     *  ive (except when z = 0 or 1 for some i). we determine fscnd(.) from
     *  the requirement that also the first derivative of  f  be continuous.
     *  in addition, we require that the third derivative be continuous
     *  across  tau[1] and across tau[ntau-2] . this leads to a strictly
     *  diagonally dominant tridiagonal linear system for the fsecnd[i]
     *  which we solve by gauss elimination without pivoting.
     *
     * @note  There must be at least 4 interpolation points for us to fit a taut
     * cubic spline, but if you provide fewer we'll fit a quadratic or linear
     * polynomial (but you must provide at least 2)
     */
    TautSpline(std::vector<double> const& x, std::vector<double> const& y, double const gamma = 0,
               Symmetry type = Unknown);

private:
    /**
     * Here's the worker routine for the TautSpline ctor
     *
     * @param x points where function's specified
     * @param y values of function at tau[]
     * @param gamma0 control extra knots
     */
    void calculateTautSpline(std::vector<double> const& x, std::vector<double> const& y, double const gamma0);
    /**
     * Fit a taut spline to a set of data, forcing the resulting spline to
     * obey S(x) = +-S(-x). The input points must have tau[] >= 0.
     *
     * See TautSpline::TautSpline() for a discussion of the algorithm, and
     * the meaning of the parameter gamma
     *
     * This is done by duplicating the input data for -ve x, so consider
     * carefully before using this function on many-thousand-point datasets
     */
    void calculateTautSplineEvenOdd(std::vector<double> const& x, std::vector<double> const& y,
                                    double const gamma0, bool even);
};

class SmoothedSpline : public Spline {
public:
    /**
     * Cubic spline data smoother
     *
     * Algorithm 642 collected algorithms from ACM. Algorithm appeared in
     * Acm-Trans. Math. Software, vol.12, no. 2, Jun., 1986, p. 150.
     *
     * Translated from fortran by a combination of f2c and RHL.
     *
     * @verbatim
         Author              - M.F.Hutchinson
                               CSIRO Division of Mathematics and Statistics
                               P.O. Box 1965
                               Canberra, ACT 2601
                               Australia
       @endverbatim
     *   latest revision     - 15 August 1985
     *
     * @param[in] x array of length n containing the abscissae of the n data points
     * (x(i),f(i)) i=0..n-1.  x must be ordered so that x(i) < x(i+1)
     * @param[in] y vector of length >= 3 containing the ordinates (or function values)
     * of the data points
     * @param[in] dy vector of standard deviations of `y`
     * the error associated with the data point; each dy[] must be positive.
     *
     * @param[in] s desired chisq
     * @param[out] chisq final chisq (if non-NULL)
     * @param[out] errs error estimates, (if non-NULL).  You'll need to delete it
     *
     * @note y,c: spline coefficients (output). y is an array of length n; c is
     * an n-1 by 3 matrix. The value of the spline approximation at t is
     *    s(t) = c[2][i]*d^3 + c[1][i]*d^2 + c[0][i]*d + y[i]
     * where x[i] <= t < x[i+1] and d = t - x[i].
     *
     * @note var: error variance. If var is negative (i.e. unknown) then the
     * smoothing parameter is determined by minimizing the generalized
     * cross validation and an estimate of the error variance is returned.
     * If var is non-negative (i.e. known) then the smoothing parameter is
     * determined to minimize an estimate, which depends on var, of the true
     * mean square error. In particular, if var is zero, then an interpolating
     * natural cubic spline is calculated. Set var to 1 if absolute standard
     * deviations have been provided in dy (see above).
     *
     * @note Additional information on the fit is available in the stat array.
     on normal exit the values are assigned as follows:
     *   stat[0] = smoothing parameter (= rho/(rho + 1))
     *   stat[1] = estimate of the number of degrees of freedom of the
     * residual sum of squares; this reduces to the usual value of n-2
     * when a least squares regression line is calculated.
     *   stat[2] = generalized cross validation
     *   stat[3] = mean square residual
     *   stat[4] = estimate of the true mean square error at the data points
     *   stat[5] = estimate of the error variance; chi^2/nu in the case
     *             of linear regression
     *
     * @note If stat[0]==0 (rho==0) an interpolating natural cubic spline has been
     * calculated; if stat[0]==1 (rho==infinite) a least squares regression
     * line has been calculated.
     *
     * @note Returns stat[4], an estimate of the true rms error
     *
     * @note precision/hardware  - double (originally VAX double)
     *
     * @note the number of arithmetic operations required by the subroutine is
     * proportional to n.  The subroutine uses an algorithm developed by
     * M.F. Hutchinson and F.R. de Hoog, 'Smoothing Noisy Data with Spline
     * Functions', Numer. Math. 47 p.99 (1985)
     */
    SmoothedSpline(std::vector<double> const& x, std::vector<double> const& y, std::vector<double> const& dy,
                   double s, double* chisq = NULL, std::vector<double>* errs = NULL);
};
}
}
}
}
#endif
