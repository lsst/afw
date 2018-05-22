/*
 * A number of non-trivial splines
 *
 * @note These should be merged into lsst::afw::math::Interpolate, but its current implementation
 * (and to some degree interface) uses gsl explicitly
 */
#include <limits>

#include "boost/format.hpp"
#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/afw/math/detail/Spline.h"
#include "lsst/geom/Angle.h"

namespace lsst {
namespace afw {
namespace math {
namespace detail {

static int search_array(double z, double const *x, int n, int i);

void Spline::_allocateSpline(int const nknot) {
    _knots.resize(nknot);
    _coeffs.resize(4);
    for (unsigned int i = 0; i != _coeffs.size(); ++i) {
        _coeffs[i].reserve(nknot);
    }
}

void Spline::interpolate(std::vector<double> const &x, std::vector<double> &y) const {
    int const nknot = _knots.size();
    int const n = x.size();

    y.resize(n);   // may default-construct elements which is a little inefficient
                   /*
                    * For _knots[i] <= x <= _knots[i+1], the interpolant
                    * has the form
                    *    val = _coeff[0][i] +dx*(_coeff[1][i] + dx*(_coeff[2][i]/2 + dx*_coeff[3][i]/6))
                    * with
                    *    dx = x - knots[i]
                    */
    int ind = -1;  // no idea initially
    for (int i = 0; i != n; ++i) {
        ind = search_array(x[i], &_knots[0], nknot, ind);

        if (ind < 0) {  // off bottom
            ind = 0;
        } else if (ind >= nknot) {  // off top
            ind = nknot - 1;
        }

        double const dx = x[i] - _knots[ind];
        y[i] = _coeffs[0][ind] +
               dx * (_coeffs[1][ind] + dx * (_coeffs[2][ind] / 2 + dx * _coeffs[3][ind] / 6));
    }
}

void Spline::derivative(std::vector<double> const &x, std::vector<double> &dydx) const {
    int const nknot = _knots.size();
    int const n = x.size();

    dydx.resize(n);  // may default-construct elements which is a little inefficient
                     /*
                      * For _knots[i] <= x <= _knots[i+1], the * interpolant has the form
                      *    val = _coeff[0][i] +dx*(_coeff[1][i] + dx*(_coeff[2][i]/2 + dx*_coeff[3][i]/6))
                      * with
                      *    dx = x - knots[i]
                      * so the derivative is
                      *    val = _coeff[1][i] + dx*(_coeff[2][i] + dx*_coeff[3][i]/2))
                      */

    int ind = -1;  // no idea initially
    for (int i = 0; i != n; ++i) {
        ind = search_array(x[i], &_knots[0], nknot, ind);

        if (ind < 0) {  // off bottom
            ind = 0;
        } else if (ind >= nknot) {  // off top
            ind = nknot - 1;
        }

        double const dx = x[i] - _knots[ind];
        dydx[i] = _coeffs[1][ind] + dx * (_coeffs[2][ind] + dx * _coeffs[3][ind] / 2);
    }
}

TautSpline::TautSpline(std::vector<double> const &x, std::vector<double> const &y, double const gamma0,
                       Symmetry type) {
    if (x.size() != y.size()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          (boost::format("TautSpline: x and y must have the same size; saw %d %d\n") %
                           x.size() % y.size())
                                  .str());
    }

    int const ntau = x.size(); /* size of tau and gtau, must be >= 2*/
    if (ntau < 2) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          (boost::format("TautSpline: ntau = %d, should be >= 2\n") % ntau).str());
    }

    switch (type) {
        case Unknown:
            calculateTautSpline(x, y, gamma0);
            break;
        case Even:
        case Odd:
            calculateTautSplineEvenOdd(x, y, gamma0, type == Even);
            break;
    }
}

void TautSpline::calculateTautSpline(std::vector<double> const &x, std::vector<double> const &y,
                                     double const gamma0) {
    const double *tau = &x[0];
    const double *gtau = &y[0];
    int const ntau = x.size();  // size of tau and gtau, must be >= 2

    if (ntau < 4) {  // use a single quadratic
        int const nknot = ntau;

        _allocateSpline(nknot);

        _knots[0] = tau[0];
        for (int i = 1; i < nknot; i++) {
            _knots[i] = tau[i];
            if (tau[i - 1] >= tau[i]) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                                  (boost::format("point %d and the next, %f %f, are out of order") % (i - 1) %
                                   tau[i - 1] %
                                   tau[i]).str());
            }
        }

        if (ntau == 2) {
            _coeffs[0][0] = gtau[0];
            _coeffs[1][0] = (gtau[1] - gtau[0]) / (tau[1] - tau[0]);
            _coeffs[2][0] = _coeffs[3][0] = 0;

            _coeffs[0][1] = gtau[1];
            _coeffs[1][1] = (gtau[1] - gtau[0]) / (tau[1] - tau[0]);
            _coeffs[2][1] = _coeffs[3][1] = 0;
        } else { /* must be 3 */
            double tmp = (tau[2] - tau[0]) * (tau[2] - tau[1]) * (tau[1] - tau[0]);
            _coeffs[0][0] = gtau[0];
            _coeffs[1][0] = ((gtau[1] - gtau[0]) * pow(tau[2] - tau[0], 2) -
                             (gtau[2] - gtau[0]) * pow(tau[1] - tau[0], 2)) /
                            tmp;
            _coeffs[2][0] =
                    -2 * ((gtau[1] - gtau[0]) * (tau[2] - tau[0]) - (gtau[2] - gtau[0]) * (tau[1] - tau[0])) /
                    tmp;
            _coeffs[3][0] = 0;

            _coeffs[0][1] = gtau[1];
            _coeffs[1][1] = _coeffs[1][0] + (tau[1] - tau[0]) * _coeffs[2][0];
            _coeffs[2][1] = _coeffs[2][0];
            _coeffs[3][1] = 0;

            _coeffs[0][2] = gtau[2];
            _coeffs[1][2] = _coeffs[1][0] + (tau[2] - tau[0]) * _coeffs[2][0];
            _coeffs[2][2] = _coeffs[2][0];
            _coeffs[3][2] = 0;
        }

        return;
    }
    /*
     * Allocate scratch space
     *     s[0][...] = dtau = tau(.+1) - tau
     *     s[1][...] = diag = diagonal in linear system
     *     s[2][...] = u = upper diagonal in linear system
     *     s[3][...] = r = right side for linear system (initially)
     *               = fsecnd = solution of linear system , namely the second
     *                          derivatives of interpolant at  tau
     *     s[4][...] = z = indicator of additional knots
     *     s[5][...] = 1/hsecnd(1,x) with x = z or = 1-z. see below.
     */
    std::vector<std::vector<double> > s(6);  // scratch space

    for (int i = 0; i != 6; i++) {
        s[i].resize(ntau);
    }
    /*
     * Construct delta tau and first and second (divided) differences of data
     */

    for (int i = 0; i < ntau - 1; i++) {
        s[0][i] = tau[i + 1] - tau[i];
        if (s[0][i] <= 0.) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              (boost::format("point %d and the next, %f %f, are out of order") % i % tau[i] %
                               tau[i + 1])
                                      .str());
        }
        s[3][i + 1] = (gtau[i + 1] - gtau[i]) / s[0][i];
    }
    for (int i = 1; i < ntau - 1; ++i) {
        s[3][i] = s[3][i + 1] - s[3][i];
    }
    /*
     * Construct system of equations for second derivatives at tau. At each
     * interior data point, there is one continuity equation, at the first
     * and the last interior data point there is an additional one for a
     * total of ntau equations in ntau unknowns.
     */
    s[1][1] = s[0][0] / 3;

    int method;
    double gamma = gamma0;  // control the smoothing
    if (gamma <= 0) {
        method = 1;
    } else if (gamma > 3) {
        gamma -= 3;
        method = 3;
    } else {
        method = 2;
    }
    double const onemg3 = 1 - gamma / 3;

    int nknot = ntau;  // count number of knots
                       /*
                        * Some compilers don't realise that the flow of control always initialises
                        * these variables; so initialise them to placate e.g. gcc
                        */
    double zeta = 0.0;
    double alpha = 0.0;
    double ratio = 0.0;

    // double c, d;
    // double z, denom, factr2;
    // double onemzt, zt2, del;

    double entry3 = 0.0;
    double factor = 0.0;
    double onemzt = 0;
    double zt2 = 0;
    double z_half = 0;

    for (int i = 1; i < ntau - 2; ++i) {
        /*
         * construct z[i] and zeta[i]
         */
        double z = .5;
        if ((method == 2 && s[3][i] * s[3][i + 1] >= 0) || method == 3) {
            double const temp = fabs(s[3][i + 1]);
            double const denom = fabs(s[3][i]) + temp;
            if (denom != 0) {
                z = temp / denom;
                if (fabs(z - 0.5) <= 1.0 / 6.0) {
                    z = 0.5;
                }
            }
        }

        s[4][i] = z;
        /*
          set up part of the i-th equation which depends on the i-th interval
        */
        z_half = z - 0.5;
        if (z_half < 0) {
            zeta = gamma * z;
            onemzt = 1 - zeta;
            zt2 = zeta * zeta;
            double temp = onemg3 / onemzt;
            alpha = (temp < 1 ? temp : 1);
            factor = zeta / (alpha * (zt2 - 1) + 1);
            s[5][i] = zeta * factor / 6;
            s[1][i] += s[0][i] * ((1 - alpha * onemzt) * factor / 2 - s[5][i]);
            /*
             * if z = 0 and the previous z = 1, then d[i] = 0. Since then
             * also u[i-1] = l[i+1] = 0, its value does not matter. Reset
             * d[i] = 1 to insure nonzero pivot in elimination.
             */
            if (s[1][i] <= 0.) {
                s[1][i] = 1;
            }
            s[2][i] = s[0][i] / 6;

            if (z != 0) { /* we'll get a new knot */
                nknot++;
            }
        } else if (z_half == 0) {
            s[1][i] += s[0][i] / 3;
            s[2][i] = s[0][i] / 6;
        } else {
            onemzt = gamma * (1 - z);
            zeta = 1 - onemzt;
            double const temp = onemg3 / zeta;
            alpha = (temp < 1 ? temp : 1);
            factor = onemzt / (1 - alpha * zeta * (onemzt + 1));
            s[5][i] = onemzt * factor / 6;
            s[1][i] += s[0][i] / 3;
            s[2][i] = s[5][i] * s[0][i];

            if (onemzt != 0) { /* we'll get a new knot */
                nknot++;
            }
        }
        if (i == 1) {
            s[4][0] = 0.5;
            /*
             * the first two equations enforce continuity of the first and of
             * the third derivative across tau[1].
             */
            s[1][0] = s[0][0] / 6;
            s[2][0] = s[1][1];
            entry3 = s[2][1];
            if (z_half < 0) {
                const double factr2 = zeta * (alpha * (zt2 - 1.) + 1.) / (alpha * (zeta * zt2 - 1.) + 1.);
                ratio = factr2 * s[0][1] / s[1][0];
                s[1][1] = factr2 * s[0][1] + s[0][0];
                s[2][1] = -factr2 * s[0][0];
            } else if (z_half == 0) {
                ratio = s[0][1] / s[1][0];
                s[1][1] = s[0][1] + s[0][0];
                s[2][1] = -s[0][0];
            } else {
                ratio = s[0][1] / s[1][0];
                s[1][1] = s[0][1] + s[0][0];
                s[2][1] = -s[0][0] * 6 * alpha * s[5][1];
            }
            /*
             * at this point, the first two equations read
             *              diag[0]*x0 +    u[0]*x1 + entry3*x2 = r[1]
             *       -ratio*diag[0]*x0 + diag[1]*x1 +   u[1]*x2 = 0
             * set r[0] = r[1] and eliminate x1 from the second equation
             */
            s[3][0] = s[3][1];

            s[1][1] += ratio * s[2][0];
            s[2][1] += ratio * entry3;
            s[3][1] = ratio * s[3][1];
        } else {
            /*
             * the i-th equation enforces continuity of the first derivative
             * across tau[i]; it reads
             *         -ratio*diag[i-1]*x_{i-1} + diag[i]*x_i + u[i]*x_{i+1} = r[i]
             * eliminate x_{i-1} from this equation
             */
            s[1][i] += ratio * s[2][i - 1];
            s[3][i] += ratio * s[3][i - 1];
        }
        /*
         * Set up the part of the next equation which depends on the i-th interval.
         */
        if (z_half < 0) {
            ratio = -s[5][i] * s[0][i] / s[1][i];
            s[1][i + 1] = s[0][i] / 3;
        } else if (z_half == 0) {
            ratio = -(s[0][i] / 6) / s[1][i];
            s[1][i + 1] = s[0][i] / 3;
        } else {
            ratio = -(s[0][i] / 6) / s[1][i];
            s[1][i + 1] = s[0][i] * ((1 - zeta * alpha) * factor / 2 - s[5][i]);
        }
    }

    s[4][ntau - 2] = 0.5;
    /*
     * last two equations, which enforce continuity of third derivative and
     * of first derivative across tau[ntau - 2]
     */
    double const entry_ = ratio * s[2][ntau - 3] + s[1][ntau - 2] + s[0][ntau - 2] / 3;
    s[1][ntau - 1] = s[0][ntau - 2] / 6;
    s[3][ntau - 1] = ratio * s[3][ntau - 3] + s[3][ntau - 2];
    if (z_half < 0) {
        ratio = s[0][ntau - 2] * 6 * s[5][ntau - 3] * alpha / s[1][ntau - 3];
        s[1][ntau - 2] = ratio * s[2][ntau - 3] + s[0][ntau - 2] + s[0][ntau - 3];
        s[2][ntau - 2] = -s[0][ntau - 3];
    } else if (z_half == 0) {
        ratio = s[0][ntau - 2] / s[1][ntau - 3];
        s[1][ntau - 2] = ratio * s[2][ntau - 3] + s[0][ntau - 2] + s[0][ntau - 3];
        s[2][ntau - 2] = -s[0][ntau - 3];
    } else {
        const double factr2 =
                onemzt * (alpha * (onemzt * onemzt - 1) + 1) / (alpha * (onemzt * onemzt * onemzt - 1) + 1);
        ratio = factr2 * s[0][ntau - 2] / s[1][ntau - 3];
        s[1][ntau - 2] = ratio * s[2][ntau - 3] + factr2 * s[0][ntau - 3] + s[0][ntau - 2];
        s[2][ntau - 2] = -factr2 * s[0][ntau - 3];
    }
    /*
     * at this point, the last two equations read
     *             diag[i]*x_i +      u[i]*x_{i+1} = r[i]
     *      -ratio*diag[i]*x_i + diag[i+1]*x_{i+1} = r[i+1]
     *     eliminate x_i from last equation
     */
    s[3][ntau - 2] = ratio * s[3][ntau - 3];
    ratio = -entry_ / s[1][ntau - 2];
    s[1][ntau - 1] += ratio * s[2][ntau - 2];
    s[3][ntau - 1] += ratio * s[3][ntau - 2];

    /*
     * back substitution
     */
    s[3][ntau - 1] /= s[1][ntau - 1];
    for (int i = ntau - 2; i > 0; --i) {
        s[3][i] = (s[3][i] - s[2][i] * s[3][i + 1]) / s[1][i];
    }

    s[3][0] = (s[3][0] - s[2][0] * s[3][1] - entry3 * s[3][2]) / s[1][0];
/*
 * construct polynomial pieces; first allocate space for the coefficients
 */
#if 1
    /*
     * Start by counting the knots
     */
    {
        int const nknot0 = nknot;
        int nknot = ntau;

        for (int i = 0; i < ntau - 1; ++i) {
            double const z = s[4][i];
            if ((z < 0.5 && z != 0.0) || (z > 0.5 && (1 - z) != 0.0)) {
                nknot++;
            }
        }
        assert(nknot == nknot0);
    }
#endif
    _allocateSpline(nknot);

    _knots[0] = tau[0];
    int j = 0;
    for (int i = 0; i < ntau - 1; ++i) {
        _coeffs[0][j] = gtau[i];
        _coeffs[2][j] = s[3][i];
        double const divdif = (gtau[i + 1] - gtau[i]) / s[0][i];
        double z = s[4][i];
        double const z_half = z - 0.5;
        if (z_half < 0) {
            if (z == 0) {
                _coeffs[1][j] = divdif;
                _coeffs[2][j] = 0;
                _coeffs[3][j] = 0;
            } else {
                zeta = gamma * z;
                onemzt = 1 - zeta;
                double const c = s[3][i + 1] / 6;
                double const d = s[3][i] * s[5][i];
                j++;

                double const del = zeta * s[0][i];
                _knots[j] = tau[i] + del;
                zt2 = zeta * zeta;
                double temp = onemg3 / onemzt;
                alpha = (temp < 1 ? temp : 1);
                factor = onemzt * onemzt * alpha;
                temp = s[0][i];
                _coeffs[0][j] = gtau[i] + divdif * del +
                                temp * temp * (d * onemzt * (factor - 1) + c * zeta * (zt2 - 1));
                _coeffs[1][j] = divdif + s[0][i] * (d * (1 - 3 * factor) + c * (3 * zt2 - 1));
                _coeffs[2][j] = (d * alpha * onemzt + c * zeta) * 6;
                _coeffs[3][j] = (c - d * alpha) * 6 / s[0][i];
                if (del * zt2 == 0) {
                    _coeffs[1][j - 1] = 0; /* would be NaN in an */
                    _coeffs[3][j - 1] = 0; /*              0-length interval */
                } else {
                    _coeffs[3][j - 1] = _coeffs[3][j] - d * 6 * (1 - alpha) / (del * zt2);
                    _coeffs[1][j - 1] = _coeffs[1][j] - del * (_coeffs[2][j] - del / 2 * _coeffs[3][j - 1]);
                }
            }
        } else if (z_half == 0) {
            _coeffs[1][j] = divdif - s[0][i] * (s[3][i] * 2 + s[3][i + 1]) / 6;
            _coeffs[3][j] = (s[3][i + 1] - s[3][i]) / s[0][i];
        } else {
            onemzt = gamma * (1 - z);
            if (onemzt == 0) {
                _coeffs[1][j] = divdif;
                _coeffs[2][j] = 0;
                _coeffs[3][j] = 0;
            } else {
                zeta = 1 - onemzt;
                double const temp = onemg3 / zeta;
                alpha = (temp < 1 ? temp : 1);
                double const c = s[3][i + 1] * s[5][i];
                double const d = s[3][i] / 6;
                double const del = zeta * s[0][i];
                _knots[j + 1] = tau[i] + del;
                _coeffs[1][j] = divdif - s[0][i] * (2 * d + c);
                _coeffs[3][j] = (c * alpha - d) * 6 / s[0][i];
                j++;

                _coeffs[3][j] =
                        _coeffs[3][j - 1] + (1 - alpha) * 6 * c / (s[0][i] * (onemzt * onemzt * onemzt));
                _coeffs[2][j] = _coeffs[2][j - 1] + del * _coeffs[3][j - 1];
                _coeffs[1][j] = _coeffs[1][j - 1] + del * (_coeffs[2][j - 1] + del / 2 * _coeffs[3][j - 1]);
                _coeffs[0][j] = _coeffs[0][j - 1] +
                                del * (_coeffs[1][j - 1] +
                                       del / 2 * (_coeffs[2][j - 1] + del / 3 * _coeffs[3][j - 1]));
            }
        }

        j++;
        _knots[j] = tau[i + 1];
    }
    /*
     * If there are discontinuities some of the knots may be at the same
     * position; in this case we generated some NaNs above. As they only
     * occur for 0-length segments, it's safe to replace them by 0s
     *
     * Due to the not-a-knot condition, the last set of coefficients isn't
     * needed (the last-but-one is equivalent), but it makes the book-keeping
     * easier if we _do_ generate them
     */
    double const del = tau[ntau - 1] - _knots[nknot - 2];

    _coeffs[0][nknot - 1] = _coeffs[0][nknot - 2] +
                            del * (_coeffs[1][nknot - 2] +
                                   del * (_coeffs[2][nknot - 2] / 2 + del * _coeffs[3][nknot - 2] / 6));
    _coeffs[1][nknot - 1] =
            _coeffs[1][nknot - 2] + del * (_coeffs[2][nknot - 2] + del * _coeffs[3][nknot - 2] / 2);
    _coeffs[2][nknot - 1] = _coeffs[2][nknot - 2] + del * _coeffs[3][nknot - 2];
    _coeffs[3][nknot - 1] = _coeffs[3][nknot - 2];

    assert(j + 1 == nknot);
}

/*
 * Here's the code to fit smoothing splines through data points
 */
static void spcof1(const double x[], double avh, const double y[], const double dy[], int n, double p,
                   double q, double a[], double *c[3], double u[], const double v[]);

static void sperr1(const double x[], double avh, const double dy[], int n, double *r[3], double p, double var,
                   std::vector<double> *se);

static double spfit1(const double x[], double avh, const double dy[], int n, double rho, double *p, double *q,
                     double var, double stat[], const double a[], double *c[3], double *r[3], double *t[2],
                     double u[], double v[]);

static double spint1(const double x[], double *avh, const double y[], double dy[], int n, double a[],
                     double *c[3], double *r[3], double *t[2]);

SmoothedSpline::SmoothedSpline(std::vector<double> const &x, std::vector<double> const &f,
                               std::vector<double> const &df, double s, double *chisq,
                               std::vector<double> *errs) {
    float var = 1;  // i.e. df is the absolute s.d.  N.B. ADD GCV Variant with var=-1
    int const n = x.size();
    double const ratio = 2.0;
    double const tau = 1.618033989; /* golden ratio */
    double avdf, avar, stat[6];
    double p, q, delta, r1, r2, r3, r4;
    double gf1, gf2, gf3, gf4, avh, err;
    /*
     * allocate scratch space
     */
    _allocateSpline(n);

    double *y = &_coeffs[0][0];
    double *c[3];
    c[0] = &_coeffs[1][0];
    c[1] = &_coeffs[2][0];
    c[2] = &_coeffs[3][0];

    std::vector<double> scratch(7 * (n + 2));  // scratch space

    double *r[3];
    r[0] = &scratch[0] + 1;  // we want indices -1..n
    r[1] = r[0] + (n + 2);
    r[2] = r[1] + (n + 2);
    double *t[2];
    t[0] = r[2] + (n + 2);
    t[1] = t[0] + (n + 2);
    double *u = t[1] + (n + 2);
    double *v = u + (n + 2);
    /*
     * and so to work.
     */
    std::vector<double> sdf = df;  // scaled values of df

    avdf = spint1(&x[0], &avh, &f[0], &sdf[0], n, y, c, r, t);
    avar = var;
    if (var > 0) {
        avar *= avdf * avdf;
    }

    if (var == 0) { /* simply find a natural cubic spline*/
        r1 = 0;

        gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u, v);
    } else { /* Find local minimum of gcv or the
                expected mean square error */
        r1 = 1;
        r2 = ratio * r1;
        gf2 = spfit1(&x[0], avh, &sdf[0], n, r2, &p, &q, avar, stat, y, c, r, t, u, v);
        bool set_r3 = false;  // was r3 set?
        for (;;) {
            gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u, v);
            if (gf1 > gf2) {
                break;
            }

            if (p <= 0) {
                break;
            }
            r2 = r1;
            gf2 = gf1;
            r1 /= ratio;
        }

        if (p <= 0) {
            set_r3 = false;
            r3 = 0; /* placate compiler */
        } else {
            r3 = ratio * r2;
            set_r3 = true;

            for (;;) {
                gf3 = spfit1(&x[0], avh, &sdf[0], n, r3, &p, &q, avar, stat, y, c, r, t, u, v);
                if (gf3 >= gf2) {
                    break;
                }

                if (q <= 0) {
                    break;
                }
                r2 = r3;
                gf2 = gf3;
                r3 = ratio * r3;
            }
        }

        if (p > 0 && q > 0) {
            assert(set_r3);
            r2 = r3;
            gf2 = gf3;
            delta = (r2 - r1) / tau;
            r4 = r1 + delta;
            r3 = r2 - delta;
            gf3 = spfit1(&x[0], avh, &sdf[0], n, r3, &p, &q, avar, stat, y, c, r, t, u, v);
            gf4 = spfit1(&x[0], avh, &sdf[0], n, r4, &p, &q, avar, stat, y, c, r, t, u, v);
            /*
             * Golden section search for local minimum
             */
            do {
                if (gf3 <= gf4) {
                    r2 = r4;
                    gf2 = gf4;
                    r4 = r3;
                    gf4 = gf3;
                    delta /= tau;
                    r3 = r2 - delta;
                    gf3 = spfit1(&x[0], avh, &sdf[0], n, r3, &p, &q, avar, stat, y, c, r, t, u, v);
                } else {
                    r1 = r3;
                    gf1 = gf3;
                    r3 = r4;
                    gf3 = gf4;
                    delta /= tau;
                    r4 = r1 + delta;
                    gf4 = spfit1(&x[0], avh, &sdf[0], n, r4, &p, &q, avar, stat, y, c, r, t, u, v);
                }

                err = (r2 - r1) / (r1 + r2);
            } while (err * err + 1 > 1 && err > 1e-6);

            r1 = (r1 + r2) * .5;
            gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u, v);
        }
    }
    /*
     * Calculate spline coefficients
     */
    spcof1(&x[0], avh, &f[0], &sdf[0], n, p, q, y, c, u, v);

    stat[2] /= avdf * avdf; /* undo scaling */
    stat[3] /= avdf * avdf;
    stat[4] /= avdf * avdf;
    /*
     * Optionally calculate standard error estimates
     */
    if (errs != NULL) {
        sperr1(&x[0], avh, &sdf[0], n, r, p, avar, errs);
    }
    /*
     * clean up
     */
    if (chisq != NULL) {
        *chisq = n * stat[4];
    }
}

/**
 * @internal initializes the arrays c, r and t for one dimensional cubic
 * smoothing spline fitting by subroutine spfit1. The values
 * df[i] are scaled so that the sum of their squares is n.
 * The average of the differences x[i+1] - x[i] is calculated
 * in avh in order to avoid underflow and overflow problems in spfit1.
 *
 * Return the initial rms value of dy.
 */
static double spint1(const double x[], double *avh, const double f[], double df[], int n, double a[],
                     double *c[3], double *r[3], double *t[2]) {
    double avdf;
    double e, ff, g, h;
    int i;

    assert(n >= 3);
    /*
     * Get average x spacing in avh
     */
    g = 0;
    for (i = 0; i < n - 1; ++i) {
        h = x[i + 1] - x[i];
        assert(h > 0);

        g += h;
    }
    *avh = g / (n - 1);
    /*
     * Scale relative weights
     */
    g = 0;
    for (int i = 0; i < n; ++i) {
        assert(df[i] > 0);

        g += df[i] * df[i];
    }
    avdf = sqrt(g / n);

    for (i = 0; i < n; ++i) {
        df[i] /= avdf;
    }
    /*
     * Initialize h,f
     */
    h = (x[1] - x[0]) / (*avh);
    ff = (f[1] - f[0]) / h;
    /*
     * Calculate a,t,r
     */
    for (i = 1; i < n - 1; ++i) {
        g = h;
        h = (x[i + 1] - x[i]) / (*avh);
        e = ff;
        ff = (f[i + 1] - f[i]) / h;
        a[i] = ff - e;
        t[0][i] = (g + h) * 2. / 3.;
        t[1][i] = h / 3.;
        r[2][i] = df[i - 1] / g;
        r[0][i] = df[i + 1] / h;
        r[1][i] = -df[i] / g - df[i] / h;
    }
    /*
     * Calculate c = r'*r
     */
    r[1][n - 1] = 0;
    r[2][n - 1] = 0;
    r[2][n] = 0;

    for (i = 1; i < n - 1; i++) {
        c[0][i] = r[0][i] * r[0][i] + r[1][i] * r[1][i] + r[2][i] * r[2][i];
        c[1][i] = r[0][i] * r[1][i + 1] + r[1][i] * r[2][i + 1];
        c[2][i] = r[0][i] * r[2][i + 2];
    }

    return (avdf);
}

/**
 * @internal Fits a cubic smoothing spline to data with relative
 * weighting dy for a given value of the smoothing parameter
 * rho using an algorithm based on that of C.H. Reinsch (1967),
 * Numer. Math. 10, 177-183.
 *
 * The trace of the influence matrix is calculated using an
 * algorithm developed by M.F.Hutchinson and F.R.De Hoog (numer.
 * math., 47 p.99 (1985)), enabling the generalized cross validation
 * and related statistics to be calculated in order n operations.
 *
 * The arrays a, c, r and t are assumed to have been initialized
 * by the routine spint1.  overflow and underflow problems are
 * avoided by using p=rho/(1 + rho) and q=1/(1 + rho) instead of
 * rho and by scaling the differences x[i+1] - x[i] by avh.
 *
 * The values in dy are assumed to have been scaled so that the
 * sum of their squared values is n.  the value in var, when it is
 * non-negative, is assumed to have been scaled to compensate for
 * the scaling of the values in dy.
 *
 * the value returned in fun is an estimate of the true mean square
 * when var is non-negative, and is the generalized cross validation
 * when var is negative.
 */
static double spfit1(const double x[], double avh, const double dy[], int n, double rho, double *pp,
                     double *pq, double var, double stat[], const double a[], double *c[3], double *r[3],
                     double *t[2], double u[], double v[]) {
    double const eps = std::numeric_limits<double>::epsilon();
    double e, f, g, h;
    double fun;
    int i;
    double p, q; /* == *pp, *pq */
                 /*
                  * Use p and q instead of rho to prevent overflow or underflow
                  */
    if (fabs(rho) < eps) {
        p = 0;
        q = 1;
    } else if (fabs(1 / rho) < eps) {
        p = 1;
        q = 0;
    } else {
        p = rho / (1 + rho);
        q = 1 / (1 + rho);
    }
    /*
     * Rational Cholesky decomposition of p*c + q*t
     */
    f = 0;
    g = 0;
    h = 0;
    r[0][-1] = r[0][0] = 0;

    for (int i = 1; i < n - 1; ++i) {
        r[2][i - 2] = g * r[0][i - 2];
        r[1][i - 1] = f * r[0][i - 1];
        {
            double tmp = p * c[0][i] + q * t[0][i] - f * r[1][i - 1] - g * r[2][i - 2];
            if (tmp == 0.0) {
                r[0][i] = 1e30;
            } else {
                r[0][i] = 1 / tmp;
            }
        }
        f = p * c[1][i] + q * t[1][i] - h * r[1][i - 1];
        g = h;
        h = p * c[2][i];
    }
    /*
     * Solve for u
     */
    u[-1] = u[0] = 0;
    for (int i = 1; i < n - 1; i++) {
        u[i] = a[i] - r[1][i - 1] * u[i - 1] - r[2][i - 2] * u[i - 2];
    }
    u[n - 1] = u[n] = 0;
    for (int i = n - 2; i > 0; i--) {
        u[i] = r[0][i] * u[i] - r[1][i] * u[i + 1] - r[2][i] * u[i + 2];
    }
    /*
     * Calculate residual vector v
     */
    e = h = 0;
    for (int i = 0; i < n - 1; i++) {
        g = h;
        h = (u[i + 1] - u[i]) / ((x[i + 1] - x[i]) / avh);
        v[i] = dy[i] * (h - g);
        e += v[i] * v[i];
    }
    v[n - 1] = -h * dy[n - 1];
    e += v[n - 1] * v[n - 1];
    /*
     * Calculate upper three bands of inverse matrix
     */
    r[0][n - 1] = r[1][n - 1] = r[0][n] = 0;
    for (i = n - 2; i > 0; i--) {
        g = r[1][i];
        h = r[2][i];
        r[1][i] = -g * r[0][i + 1] - h * r[1][i + 1];
        r[2][i] = -g * r[1][i + 1] - h * r[0][i + 2];
        r[0][i] = r[0][i] - g * r[1][i] - h * r[2][i];
    }
    /*
     * Calculate trace
     */
    f = g = h = 0;
    for (i = 1; i < n - 1; ++i) {
        f += r[0][i] * c[0][i];
        g += r[1][i] * c[1][i];
        h += r[2][i] * c[2][i];
    }
    f += 2 * (g + h);
    /*
     * Calculate statistics
     */
    stat[0] = p;
    stat[1] = f * p;
    stat[2] = n * e / (f * f + 1e-20);
    stat[3] = e * p * p / n;
    stat[5] = e * p / ((f < 0) ? f - 1e-10 : f + 1e-10);
    if (var >= 0) {
        fun = stat[3] - 2 * var * stat[1] / n + var;

        stat[4] = fun;
    } else {
        stat[4] = stat[5] - stat[3];
        fun = stat[2];
    }

    *pp = p;
    *pq = q;

    return (fun);
}

/**
 * @internal Calculates coefficients of a cubic smoothing spline from
 * parameters calculated by spfit1()
 */
static void spcof1(const double x[], double avh, const double y[], const double dy[], int n, double p,
                   double q, double a[], double *c[3], double u[], const double v[]) {
    double h;
    int i;
    double qh;

    qh = q / (avh * avh);

    for (i = 0; i < n; ++i) {
        a[i] = y[i] - p * dy[i] * v[i];
        u[i] = qh * u[i];
    }
    /*
     * calculate c
     */
    for (i = 0; i < n - 1; ++i) {
        h = x[i + 1] - x[i];
        c[2][i] = (u[i + 1] - u[i]) / (3 * h);
        c[0][i] = (a[i + 1] - a[i]) / h - (h * c[2][i] + u[i]) * h;
        c[1][i] = u[i];
    }
}

/**
 * @internal Calculates Bayesian estimates of the standard errors of the fitted
 * values of a cubic smoothing spline by calculating the diagonal elements
 * of the influence matrix.
 */
static void sperr1(const double x[], double avh, const double dy[], int n, double *r[3], double p, double var,
                   std::vector<double> *se) {
    double f, g, h;
    int i;
    double f1, g1, h1;
    /*
     * Initialize
     */
    h = avh / (x[1] - x[0]);
    (*se)[0] = 1 - p * dy[0] * dy[0] * h * h * r[0][1];
    r[0][0] = r[1][0] = r[2][0] = 0;
    /*
     * Calculate diagonal elements
     */
    for (i = 1; i < n - 1; ++i) {
        f = h;
        h = avh / (x[i + 1] - x[i]);
        g = -(f + h);
        f1 = f * r[0][i - 1] + g * r[1][i - 1] + h * r[2][i - 1];
        g1 = f * r[1][i - 1] + g * r[0][i] + h * r[1][i];
        h1 = f * r[2][i - 1] + g * r[1][i] + h * r[0][i + 1];
        (*se)[i] = 1 - p * dy[i] * dy[i] * (f * f1 + g * g1 + h * h1);
    }
    (*se)[n - 1] = 1 - p * dy[n - 1] * dy[n - 1] * h * h * r[0][n - 2];
    /*
     * Calculate standard error estimates
     */
    for (int i = 0; i < n; ++i) {
        double const tmp = (*se)[i] * var;
        (*se)[i] = (tmp >= 0) ? sqrt(tmp) * dy[i] : 0;
    }
}

void TautSpline::calculateTautSplineEvenOdd(std::vector<double> const &_tau, std::vector<double> const &_gtau,
                                            double const gamma,
                                            bool const even  // ensure Even symmetry
                                            ) {
    const double *tau = &_tau[0];
    const double *gtau = &_gtau[0];
    int const ntau = _tau.size();  // size of tau and gtau, must be >= 2
    std::vector<double> x, y;      // tau and gtau, extended to -ve tau

    if (tau[0] == 0.0f) {
        int const np = 2 * ntau - 1;
        x.resize(np);
        y.resize(np);

        x[ntau - 1] = tau[0];
        y[ntau - 1] = gtau[0];
        for (int i = 1; i != ntau; ++i) {
            if (even) {
                x[ntau - 1 + i] = tau[i];
                y[ntau - 1 + i] = gtau[i];
                x[ntau - 1 - i] = -tau[i];
                y[ntau - 1 - i] = gtau[i];
            } else {
                x[ntau - 1 + i] = tau[i];
                y[ntau - 1 + i] = gtau[i];
                x[ntau - 1 - i] = -tau[i];
                y[ntau - 1 - i] = -gtau[i];
            }
        }
    } else {
        int const np = 2 * ntau;
        x.resize(np);
        y.resize(np);

        for (int i = 0; i != ntau; ++i) {
            if (even) {
                x[ntau + i] = tau[i];
                y[ntau + i] = gtau[i];
                x[ntau - 1 - i] = -tau[i];
                y[ntau - 1 - i] = gtau[i];
            } else {
                x[ntau + i] = tau[i];
                y[ntau + i] = gtau[i];
                x[ntau - 1 - i] = -tau[i];
                y[ntau - 1 - i] = -gtau[i];
            }
        }
    }

    TautSpline sp(x, y, gamma);  // fit a taut spline to x, y
                                 /*
                                  * Now repackage that spline to reflect the original points
                                  */
    int ii;
    for (ii = sp._knots.size() - 1; ii >= 0; --ii) {
        if (sp._knots[ii] < 0.0f) {
            break;
        }
    }
    int const i0 = ii + 1;
    int const nknot = sp._knots.size() - i0;

    _allocateSpline(nknot);

    for (int i = i0; i != static_cast<int>(sp._knots.size()); ++i) {
        _knots[i - i0] = sp._knots[i];
        for (int j = 0; j != 4; ++j) {
            _coeffs[j][i - i0] = sp._coeffs[j][i];
        }
    }
}

/**
 * @internal returns index i of first element of x >= z; the input i is an initial guess
 *
 * N.b. we could use std::lower_bound except that we use i as an initial hint
 */
static int search_array(double z, double const *x, int n, int i) {
    int lo, hi, mid;
    double xm;

    if (i < 0 || i >= n) { /* initial guess is useless */
        lo = -1;
        hi = n;
    } else {
        unsigned int step = 1; /* how much to step up/down */

        if (z > x[i]) {       /* expand search upwards */
            if (i == n - 1) { /* off top of array */
                return (n - 1);
            }

            lo = i;
            hi = lo + 1;
            while (z >= x[hi]) {
                lo = hi;
                step += step; /* double step size */
                hi = lo + step;
                if (hi >= n) { /* reached top of array */
                    hi = n - 1;
                    break;
                }
            }
        } else {          /* expand it downwards */
            if (i == 0) { /* off bottom of array */
                return (-1);
            }

            hi = i;
            lo = i - 1;
            while (z < x[lo]) {
                hi = lo;
                step += step; /* double step size */
                lo = hi - step;
                if (lo < 0) { /* off bottom of array */
                    lo = -1;
                    break;
                }
            }
        }
    }
    /*
     * perform bisection
     */
    while (hi - lo > 1) {
        mid = (lo + hi) / 2;
        xm = x[mid];
        if (z <= xm) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    if (lo == -1) { /* off the bottom */
        return (lo);
    }
    /*
     * If there's a discontinuity (many knots at same x-value), choose the
     * largest
     */
    xm = x[lo];
    while (lo < n - 1 && x[lo + 1] == xm) lo++;

    return (lo);
}

/**
 * @internal Move the roots that lie in the specified range [x0,x1) from newRoots to roots
 */
static void keep_valid_roots(std::vector<double> &roots, std::vector<double> &newRoots, double x0,
                             double x1) {
    for (unsigned int i = 0; i != newRoots.size(); ++i) {
        if (newRoots[i] >= x0 && newRoots[i] < x1) {  // keep this root
            roots.push_back(newRoots[i]);
        }
    }

    newRoots.clear();
}

/**
 * @internal find the real roots of a quadratic ax^2 + bx + c = 0
 */
namespace {
void do_quadratic(double a, double b, double c, std::vector<double> &roots) {
    if (::fabs(a) < std::numeric_limits<double>::epsilon()) {
        if (::fabs(b) >= std::numeric_limits<double>::epsilon()) {
            roots.push_back(-c / b);
        }
    } else {
        double const tmp = b * b - 4 * a * c;

        if (tmp >= 0) {
            if (b >= 0) {
                roots.push_back((-b - sqrt(tmp)) / (2 * a));
            } else {
                roots.push_back((-b + sqrt(tmp)) / (2 * a));
            }
            roots.push_back(c / (a * roots[0]));
            /*
             * sort roots
             */
            if (roots[0] > roots[1]) {
                double const tmp2 = roots[0];
                roots[0] = roots[1];
                roots[1] = tmp2;
            }
        }
    }
}

/**
 * @internal find the real roots of a cubic ax^3 + bx^2 + cx + d = 0
 */
void do_cubic(double a, double b, double c, double d, std::vector<double> &roots) {
    if (::fabs(a) < std::numeric_limits<double>::epsilon()) {
        do_quadratic(b, c, d, roots);
        return;
    }
    b /= a;
    c /= a;
    d /= a;

    double const q = (b * b - 3 * c) / 9;
    double const r = (2 * b * b * b - 9 * b * c + 27 * d) / 54;
    /*
     * n.b. note that the test for the number of roots is carried out on the
     * same variables as are used in (e.g.) the acos, as it is possible for
     * r*r < q*q*q && r > sq*sq*sq due to rounding.
     */
    double const sq = (q >= 0) ? sqrt(q) : -sqrt(-q);
    double const sq3 = sq * sq * sq;
    if (::fabs(r) < sq3) {                     // three real roots
        double const theta = ::acos(r / sq3);  // sq3 cannot be zero

        roots.push_back(-2 * sq * cos(theta / 3) - b / 3);
        roots.push_back(-2 * sq * cos((theta + lsst::geom::TWOPI) / 3) - b / 3);
        roots.push_back(-2 * sq * cos((theta - lsst::geom::TWOPI) / 3) - b / 3);
        /*
         * sort roots
         */
        if (roots[0] > roots[1]) {
            std::swap(roots[0], roots[1]);
        }
        if (roots[1] > roots[2]) {
            std::swap(roots[1], roots[2]);
        }
        if (roots[0] > roots[1]) {
            std::swap(roots[0], roots[1]);
        }

        return;
    } else if (::fabs(r) == sq3) { /* no more than two real roots */
        double const aa = -((r < 0) ? -::pow(-r, 1.0 / 3.0) : ::pow(r, 1.0 / 3.0));

        if (::fabs(aa) < std::numeric_limits<double>::epsilon()) { /* degenerate case; one real root */
            roots.push_back(-b / 3);
            return;
        } else {
            roots.push_back(2 * aa - b / 3);
            roots.push_back(-aa - b / 3);

            if (roots[0] > roots[1]) {
                std::swap(roots[0], roots[1]);
            }

            return;
        }
    } else { /* only one real root */
        double tmp = ::sqrt(r * r - (q > 0 ? sq3 * sq3 : -sq3 * sq3));
        tmp = r + (r < 0 ? -tmp : tmp);
        double const aa = -((tmp < 0) ? -::pow(-tmp, 1.0 / 3.0) : ::pow(tmp, 1.0 / 3.0));
        double const bb = (fabs(aa) < std::numeric_limits<double>::epsilon()) ? 0 : q / aa;

        roots.push_back((aa + bb) - b / 3);
#if 0
        roots.push_back(-(aa + bb)/2 - b/3);	// the real
        roots.push_back(::sqrt(3)/2*(aa - bb)); //         and imaginary parts of the complex roots
#endif
        return;
    }
}
}

std::vector<double> Spline::roots(double const value, double a, double const b) const {
    /*
     * Strategy: we know that the interpolant has the form
     *    val = coef[0][i] +dx*(coef[1][i] + dx*(coef[2][i]/2 + dx*coef[3][i]/6))
     * so we can use the usual analytic solution for a cubic. Note that the
     * cubic quoted above returns dx, the distance from the previous knot,
     * rather than x itself
     */
    std::vector<double> roots; /* the roots found */
    double x0 = a;             // lower end of current range
    double const x1 = b;
    int const nknot = _knots.size();

    int i0 = search_array(x0, &_knots[0], nknot, -1);
    int const i1 = search_array(x1, &_knots[0], nknot, i0);
    assert(i1 >= i0 && i1 <= nknot - 1);

    std::vector<double> newRoots;  // the roots we find in some interval
                                   /*
                                    * Deal with special case that x0 may be off one end or the other of
                                    * the array of knots.
                                    */
    if (i0 < 0) {                  /* off bottom */
        i0 = 0;
        do_cubic(_coeffs[3][i0] / 6, _coeffs[2][i0] / 2, _coeffs[1][i0], _coeffs[0][i0] - value, newRoots);
        //
        // Could use
        //    std::transform(newRoots.begin(), newRoots.end(), newRoots.begin(),
        //                   std::bind(std::plus<double>(), _1, _knots[i0]));
        // but let's not
        //
        for (unsigned int j = 0; j != newRoots.size(); ++j) {
            newRoots[j] += _knots[i0];
        }
        keep_valid_roots(roots, newRoots, x0, _knots[i0]);

        x0 = _knots[i0];
    } else if (i0 >= nknot) { /* off top */
        i0 = nknot - 1;
        assert(i0 >= 0);
        do_cubic(_coeffs[3][i0] / 6, _coeffs[2][i0] / 2, _coeffs[1][i0], _coeffs[0][i0] - value, newRoots);

        for (unsigned int j = 0; j != newRoots.size(); ++j) {
            newRoots[j] += _knots[i0];
        }
        keep_valid_roots(roots, newRoots, x0, x1);

        return roots;
    }
    /*
     * OK, now search in main body of spline. Note that i1 may be nknot - 1, and
     * in any case the right hand limit of the last segment is at x1, not a knot
     */
    for (int i = i0; i <= i1; i++) {
        do_cubic(_coeffs[3][i] / 6, _coeffs[2][i] / 2, _coeffs[1][i], _coeffs[0][i] - value, newRoots);

        for (unsigned int j = 0; j != newRoots.size(); ++j) {
            newRoots[j] += _knots[i];
        }
        keep_valid_roots(roots, newRoots, ((i == i0) ? x0 : _knots[i]), ((i == i1) ? x1 : _knots[i + 1]));
    }

    return roots;
}
}
}
}
}
