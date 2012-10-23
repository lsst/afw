/**
 * \brief A number of non-trivial splines
 *
 * \note These should be merged into lsst::afw::math::Interpolate, but its current implementation
 * (and to some degree interface) uses gsl explicitly
 */
#include <limits>

#include "boost/format.hpp"
#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/afw/math/detail/Spline.h"
#include "lsst/afw/geom/Angle.h"

namespace afwGeom = lsst::afw::geom;

namespace lsst { namespace afw { namespace math { namespace detail {

/*
 * Allocate the storage a Spline needs
 */
void
Spline::_allocateSpline(int const nknot)
{
    _knots.resize(nknot);
    _coeffs.resize(4);
    for (unsigned int i = 0; i != _coeffs.size(); ++i) {
        _coeffs[i].reserve(nknot);
    }
}

/*****************************************************************************/
/*
 * Here's the code to fit smoothing splines through data points
 */
static void
spcof1(const double x[], double avh, const double y[], const double dy[],
       int n, double p, double q, double a[], double *c[3], double u[],
       const double v[]);

static void
sperr1(const double x[], double avh, const double dy[], int n,
       double *r[3], double p, double var, std::vector<double> *se);

static double
spfit1(const double x[], double avh, const double dy[], int n,
       double rho, double *p, double *q, double var, double stat[],
       const double a[], double *c[3], double *r[3], double *t[2],
       double u[], double v[]);

static double
spint1(const double x[], double *avh, const double y[], double dy[], int n,
       double a[], double *c[3], double *r[3], double *t[2]);

/**
 * Algorithm 642 collected algorithms from ACM.
 *     Algorithm appeared in Acm-Trans. Math. Software, vol.12, no. 2,
 *     Jun., 1986, p. 150.
 *
 * Translated from fortran by a combination of f2c and RHL
 *
 *   Author              - M.F.Hutchinson
 *                         CSIRO Division of Mathematics and Statistics
 *                         P.O. Box 1965
 *                         Canberra, ACT 2601
 *                         Australia
 *
 *   latest revision     - 15 August 1985
 *
 *   purpose             - cubic spline data smoother
 *
 * arguments:
 *   @param x array of length n containing the abscissae of the n data points
 * (x(i),f(i)) i=0..n-1.  x must be ordered so that x(i) .lt. x(i+1)
 *
 *   @param f vector of length >= 3 containing the ordinates (or function values)
 * of the data points
 *
 *   @param df vector of standard deviations of
 * the error associated with the data point; each df[] must be positive.
 *
 *   y,c: spline coefficients (output). y is an array of length n; c is
 * an n-1 by 3 matrix. The value of the spline approximation at t is
 *    s(t) = c[2][i]*d^3 + c[1][i]*d^2 + c[0][i]*d + y[i]
 * where x[i] <= t < x[i+1] and d = t - x[i].
 *
 *   var: error variance. If var is negative (i.e. unknown) then the
 * smoothing parameter is determined by minimizing the generalized
 * cross validation and an estimate of the error variance is returned.
 * If var is non-negative (i.e. known) then the smoothing parameter is
 * determined to minimize an estimate, which depends on var, of the true
 * mean square error. In particular, if var is zero, then an interpolating
 * natural cubic spline is calculated. Set var to 1 if absolute standard
 * deviations have been provided in df (see above).
 *
 * Notes:
 *
 * Additional information on the fit is available in the stat array.
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
 * If stat[0]==0 (rho==0) an interpolating natural cubic spline has been
 * calculated; if stat[0]==1 (rho==infinite) a least squares regression
 * line has been calculated.
 *
 * Returns stat[4], an estimate of the true rms error
 *
 * precision/hardware  - double (originally VAX double)
 *
 * the number of arithmetic operations required by the subroutine is
 * proportional to n.  The subroutine uses an algorithm developed by
 * M.F. Hutchinson and F.R. de Hoog, 'Smoothing Noisy Data with Spline
 * Functions', Numer. Math. 47 p.99 (1985)
 */
SmoothedSpline::SmoothedSpline(
        std::vector<double> const& x,  ///< points where function's specified; monotonic increasing
        std::vector<double> const& f,  ///< values of function at x
        std::vector<double> const& df, ///< error in function at x
        double s,                      ///< desired chisq
        double *chisq,                 ///< final chisq (if non-NULL)
        std::vector<double> *errs      ///< error estimates, (if non-NULL).  You'll need to delete it
                              )
{
    float var = 1;                      // i.e. df is the absolute s.d.  N.B. ADD GCV Variant with var=-1
    int const n = x.size();
    double const ratio = 2.0;
    double const tau = 1.618033989;	/* golden ratio */
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
    
    std::vector<double> scratch(7*(n+2)); // scratch space

    double *r[3];
    r[0] = &scratch[0] + 1;             // we want indices -1..n
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
    std::vector<double> sdf = df;       // scaled values of df

    avdf = spint1(&x[0], &avh, &f[0], &sdf[0], n, y, c, r, t);
    avar = var;
    if (var > 0) {
        avar *= avdf*avdf;
    }

    if (var == 0) {			/* simply find a natural cubic spline*/
        r1 = 0;

        gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u, v);
    } else {				/* Find local minimum of gcv or the
					   expected mean square error */
        r1 = 1;
        r2 = ratio*r1;
        gf2 = spfit1(&x[0], avh, &sdf[0], n, r2, &p, &q, avar, stat, y, c, r, t, u, v);
        bool set_r3 = false;            // was r3 set?
        for (;;) {
            gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u,v);
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
      
        if(p <= 0) {
            set_r3 = false;
            r3 = 0;			/* placate compiler */
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
                r3 = ratio*r3;
            }
        }
      
        if(p > 0 && q > 0) {
            assert (set_r3);
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
            } while(err*err + 1 > 1 && err > 1e-6);

            r1 = (r1 + r2) * .5;
            gf1 = spfit1(&x[0], avh, &sdf[0], n, r1, &p, &q, avar, stat, y, c, r, t, u,v);
        }
    }
/*
 * Calculate spline coefficients
 */
    spcof1(&x[0], avh, &f[0], &sdf[0], n, p, q, y, c, u, v);

    stat[2] /= avdf*avdf;		/* undo scaling */
    stat[3] /= avdf*avdf;
    stat[4] /= avdf*avdf;
/*
 * Optionally calculate standard error estimates
 */
    if(errs != NULL) {
        sperr1(&x[0], avh, &sdf[0], n, r, p, avar, errs);
    }
/*
 * clean up
 */
    if(chisq != NULL) {
        *chisq = n*stat[4];
    }
}

/*****************************************************************************/
/*
 * initializes the arrays c, r and t for one dimensional cubic
 * smoothing spline fitting by subroutine spfit1. The values
 * df[i] are scaled so that the sum of their squares is n.
 * The average of the differences x[i+1] - x[i] is calculated
 * in avh in order to avoid underflow and overflow problems in spfit1.
 *
 * Return the initial rms value of dy.
 */
static double
spint1(const double x[],
       double *avh,
       const double f[],
       double df[],
       int n,
       double a[],
       double *c[3],
       double *r[3],
       double *t[2])
{
   double avdf;
   double e, ff, g, h;
   int i;

   assert (n >= 3);
/*
 * Get average x spacing in avh
 */
   g = 0;
   for (i = 0; i < n - 1; ++i) {
      h = x[i + 1] - x[i];
      assert (h > 0);

      g += h;
   }
   *avh = g/(n - 1);
/*
 * Scale relative weights
 */
   g = 0;
   for (int i = 0; i < n; ++i) {
      assert(df[i] > 0);

      g += df[i]*df[i];
   }
   avdf = sqrt(g / n);

   for (i = 0; i < n; ++i) {
      df[i] /= avdf;
   }
/*
 * Initialize h,f
 */
   h = (x[1] - x[0])/(*avh);
   ff = (f[1] - f[0])/h;
/*
 * Calculate a,t,r
 */
   for (i = 1; i < n - 1; ++i) {
      g = h;
      h = (x[i + 1] - x[i])/(*avh);
      e = ff;
      ff = (f[i + 1] - f[i])/h;
      a[i] = ff - e;
      t[0][i] = (g + h) * 2./3.;
      t[1][i] = h / 3.;
      r[2][i] = df[i - 1]/g;
      r[0][i] = df[i + 1]/h;
      r[1][i] = -df[i]/g - df[i]/h;
   }
/*
 * Calculate c = r'*r
 */
   r[1][n-1] = 0;
   r[2][n-1] = 0;
   r[2][n] = 0;

   for (i = 1; i < n - 1; i++) {
      c[0][i] = r[0][i]*r[0][i]   + r[1][i]*r[1][i] + r[2][i]*r[2][i];
      c[1][i] = r[0][i]*r[1][i+1] + r[1][i]*r[2][i+1];
      c[2][i] = r[0][i]*r[2][i+2];
   }

   return(avdf);
}

/*****************************************************************************/
/*
 * Fits a cubic smoothing spline to data with relative
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
static double
spfit1(const double x[],
       double avh,
       const double dy[],
       int n,
       double rho,
       double *pp,
       double *pq,
       double var,
       double stat[],
       const double a[],
       double *c[3],
       double *r[3],
       double *t[2],
       double u[],
       double v[])
{
    double const eps = std::numeric_limits<double>::epsilon();
   double e, f, g, h;
   double fun;
   int i;
   double p, q;				/* == *pp, *pq */
/*
 * Use p and q instead of rho to prevent overflow or underflow
 */
   if(fabs(rho) < eps) {
      p = 0; q = 1;
   } else if(fabs(1/rho) < eps) {
      p = 1; q = 0;
   } else {
      p = rho/(1 + rho);
      q =   1/(1 + rho);
   }
/*
 * Rational Cholesky decomposition of p*c + q*t
 */
   f = 0;
   g = 0;
   h = 0;
   r[0][-1] = r[0][0] = 0;
   
   for (int i = 1; i < n - 1; ++i) {
      r[2][i-2] = g*r[0][i - 2];
      r[1][i-1] = f*r[0][i - 1];
      {
	 double tmp = p*c[0][i] + q*t[0][i] - f*r[1][i-1] - g*r[2][i-2];
	 if(tmp == 0.0) {
	    r[0][i] = 1e30;
	 } else {
	    r[0][i] = 1/tmp;
	 }
      }
      f = p*c[1][i] + q*t[1][i] - h*r[1][i-1];
      g = h;
      h = p*c[2][i];
   }
/*
 * Solve for u
 */
   u[-1] = u[0] = 0;
   for (int i = 1; i < n - 1; i++) {
      u[i] = a[i] - r[1][i-1]*u[i-1] - r[2][i-2]*u[i-2];
   }
   u[n-1] = u[n] = 0;
   for (int i = n - 2; i > 0; i--) {
      u[i] = r[0][i]*u[i] - r[1][i]*u[i+1] - r[2][i]*u[i+2];
   }
/*
 * Calculate residual vector v
 */
   e = h = 0;
   for (int i = 0; i < n - 1; i++) {
      g = h;
      h = (u[i + 1] - u[i])/((x[i+1] - x[i])/avh);
      v[i] = dy[i]*(h - g);
      e += v[i]*v[i];
   }
   v[n-1] = -h*dy[n-1];
   e += v[n-1]*v[n-1];
/*
 * Calculate upper three bands of inverse matrix
 */
   r[0][n-1] = r[1][n-1] = r[0][n] = 0;
   for (i = n - 2; i > 0; i--) {
      g = r[1][i];
      h = r[2][i];
      r[1][i] = -g*r[0][i+1] - h*r[1][i+1];
      r[2][i] = -g*r[1][i+1] - h*r[0][i+2];
      r[0][i] = r[0][i] - g*r[1][i] - h*r[2][i];
   }
/*
 * Calculate trace
 */
   f = g = h = 0;
   for (i = 1; i < n - 1; ++i) {
      f += r[0][i]*c[0][i];
      g += r[1][i]*c[1][i];
      h += r[2][i]*c[2][i];
   }
   f += 2*(g + h);
/*
 * Calculate statistics
 */
   stat[0] = p;
   stat[1] = f*p;
   stat[2] = n*e/(f*f + 1e-20);
   stat[3] = e*p*p/n;
   stat[5] = e*p/((f < 0) ? f - 1e-10 : f + 1e-10);
   if (var >= 0) {
      fun = stat[3] - 2*var*stat[1]/n + var;

      stat[4] = fun;
   } else {
      stat[4] = stat[5] - stat[3];
      fun = stat[2];
   }
   
   *pp = p; *pq = q;

   return(fun);
}

/*
 * Calculates coefficients of a cubic smoothing spline from
 * parameters calculated by spfit1()
 */
static void
spcof1(const double x[],
       double avh,
       const double y[],
       const double dy[],
       int n,
       double p,
       double q,
       double a[],
       double *c[3],
       double u[],
       const double v[])
{
   double h;
   int i;
   double qh;
   
   qh = q/(avh*avh);
   
   for (i = 0; i < n; ++i) {
      a[i] = y[i] - p * dy[i] * v[i];
      u[i] = qh*u[i];
   }
/*
 * calculate c
 */
   for (i = 0; i < n - 1; ++i) {
      h = x[i+1] - x[i];
      c[2][i] = (u[i + 1] - u[i])/(3*h);
      c[0][i] = (a[i + 1] - a[i])/h - (h*c[2][i] + u[i])*h;
      c[1][i] = u[i];
   }
}

/*****************************************************************************/
/*
 * Calculates Bayesian estimates of the standard errors of the fitted
 * values of a cubic smoothing spline by calculating the diagonal elements
 * of the influence matrix.
 */
static void
sperr1(const double x[],
       double avh,
       const double dy[],
       int n,
       double *r[3],
       double p,
       double var,
       std::vector<double> *se)
{
   double f, g, h;
   int i;
   double f1, g1, h1;
/*
 * Initialize
 */
   h = avh/(x[1] - x[0]);
   (*se)[0] = 1 - p*dy[0]*dy[0]*h*h*r[0][1];
   r[0][0] = r[1][0] = r[2][0] = 0;
/*
 * Calculate diagonal elements
 */
   for (i = 1; i < n - 1; ++i) {
      f = h;
      h = avh/(x[i+1] - x[i]);
      g = -(f + h);
      f1 = f*r[0][i-1] + g*r[1][i-1] + h*r[2][i-1];
      g1 = f*r[1][i-1] + g*r[0][i] + h*r[1][i];
      h1 = f*r[2][i-1] + g*r[1][i] + h*r[0][i+1];
      (*se)[i] = 1 - p*dy[i]*dy[i]*(f*f1 + g*g1 + h*h1);
   }
   (*se)[n-1] = 1 - p*dy[n-1]*dy[n-1]*h*h*r[0][n-2];
/*
 * Calculate standard error estimates
 */
    for (int i = 0; i < n; ++i) {
        double const tmp = (*se)[i]*var;
        (*se)[i] = (tmp >= 0) ? sqrt(tmp)*dy[i] : 0;
    }
}
}}}}
