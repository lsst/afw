// -*- LSST-C++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 
/**
 * \file
 * \brief Calculate a taut spline interpolant for a set of x,y vectors
 * \ingroup afw
 */
#include <limits>
#include <algorithm>
#include <map>
#include "boost/format.hpp"
#include "boost/shared_ptr.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/math/detail/InterpolateTaut.h"

namespace lsst {
namespace ex = pex::exceptions;
namespace afw {
namespace math {
namespace detail {
namespace {
    int search_array(double z, double const *x, int n, int i);
}

/*****************************************************************************/
/**
 * \brief Allocate the storage a Spline needs
 */
void
InterpolateSdssSpline::_allocateSpline(int const nknot ///< Number of knots
                                      )
{
    _knots.resize(nknot);
    _coeffs.resize(4);
    for (unsigned int i = 0; i != _coeffs.size(); ++i) {
        _coeffs[i].reserve(nknot);
    }
}

/*****************************************************************************/
/**
 * \brief Return the value at xInterp
 *
 * \note The form taking std::vector<double> is more efficient
 */
double
InterpolateSdssSpline::interpolate(double const xInterp) const
{
    std::vector<double> x(1), y(1);

    x[0] = xInterp;
    interpolate(x, y);
    return y[0];
}

/**
 * \brief Interpolate a Spline.
 */
void
InterpolateSdssSpline::interpolate(std::vector<double> const& x, ///< points to interpolate at
                                   std::vector<double> & y       ///< values of spline interpolation
                                  ) const
{
    int const nknot = _knots.size();
    int const n = x.size();

    y.resize(n);                         // may default-construct elements which is a little inefficient
    /*
     * For _knots[i] <= x <= _knots[i+1], the interpolant
     * has the form
     *    val = _coeff[0][i] +dx*(_coeff[1][i] + dx*(_coeff[2][i]/2 + dx*_coeff[3][i]/6))
     * with
     *    dx = x - knots[i]
     */
    int ind = -1;                        // no idea initially
    for (int i = 0; i != n; ++i) {
        ind = search_array(x[i], &_knots[0], nknot, ind);

        if(ind < 0) {			// off bottom
            ind = 0;
        } else if(ind >= nknot) {		// off top
            ind = nknot - 1;
        }

        double const dx = x[i] - _knots[ind];
        y[i] = _coeffs[0][ind] + dx*(_coeffs[1][ind] + dx*(_coeffs[2][ind]/2 + dx*_coeffs[3][ind]/6));
    }
}

/*****************************************************************************/
/**
 * \brief Return the derivative at xInterp
 *
 * \note The form taking std::vector<double> is more efficient
 */
double
InterpolateSdssSpline::derivative(double const xInterp) const
{
    std::vector<double> x(1), y(1);

    x[0] = xInterp;
    derivative(x, y);
    return y[0];
}

/**
 * \brief Find the derivative of a Spline.
 */
void
InterpolateSdssSpline::derivative(std::vector<double> const& x, ///< points to evaluate derivative at
                                  std::vector<double> &dydx     ///< derivatives at x
                                 ) const
{
    int const nknot = _knots.size();
    int const n = x.size();

    dydx.resize(n);                      // may default-construct elements which is a little inefficient
    /*
     * For _knots[i] <= x <= _knots[i+1], the * interpolant has the form
     *    val = _coeff[0][i] +dx*(_coeff[1][i] + dx*(_coeff[2][i]/2 + dx*_coeff[3][i]/6))
     * with
     *    dx = x - knots[i]
     * so the derivative is
     *    val = _coeff[1][i] + dx*(_coeff[2][i] + dx*_coeff[3][i]/2))
     */
   
    int ind = -1;                        // no idea initially
    for (int i = 0; i != n; ++i) {
        ind = search_array(x[i], &_knots[0], nknot, ind);
       
        if(ind < 0) {			// off bottom
            ind = 0;
        } else if(ind >= nknot) {		// off top
            ind = nknot - 1;
        }
       
        double const dx = x[i] - _knots[ind];
        dydx[i] = _coeffs[1][ind] + dx*(_coeffs[2][ind] + dx*_coeffs[3][ind]/2);
    }
}

/************************************************************************************************************/

InterpolateControlTautSpline::InterpolateControlTautSpline(
        float gamma,                             ///< How taut should I be?
        InterpolateTautSpline::Symmetry symmetry ///< Desired symmetry
                                                          ) :
    InterpolateControl(Interpolate::TAUT_SPLINE),
    _gamma(gamma), _symmetry(symmetry)
{
    ;
}

/************************************************************************************************************/
/**
 * \brief Construct cubic spline interpolant to given data.
 */
InterpolateTautSpline::InterpolateTautSpline(
        std::vector<double> const& x,   ///< points where function's specified
        std::vector<double> const& y,   ///< values of function at tau[]
        InterpolateControlTautSpline const& ictrl
                                            ) : InterpolateSdssSpline(x, y)
{
    if(x.size() != y.size()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          (boost::format("TautSpline: x and y must have the same size; saw %d %d\n")
                           % x.size() % y.size()).str());
    }

    int const ntau = x.size();		// size of tau and gtau, must be >= 2
    if(ntau < 2) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          (boost::format("TautSpline: ntau = %d, should be >= 2\n")
                           % ntau).str());
    }

    switch (ictrl._symmetry) {
      case UNKNOWN:
        _calculateTautSpline(x, y, ictrl._gamma);
        break;
      case EVEN:
        _calculateTautSplineEvenOdd(x, y, ictrl._gamma, true);
        break;
      case ODD:
        _calculateTautSplineEvenOdd(x, y, ictrl._gamma, false);
        break;
    }
}

/************************************************************************************************************/
/**
 * \brief the worker routine for the InterpolateTautSpline ctor
 */
void
InterpolateTautSpline::_calculateTautSpline(
        std::vector<double> const& x,   ///< points where function's specified
        std::vector<double> const& y,   ///< values of function at tau[]
        double const gamma0             ///< control extra knots
                                           )
{
    const double *tau = &x[0];
    const double *gtau = &y[0];
    int const ntau = x.size();		// size of tau and gtau, must be >= 2

    if(ntau < 4) {			// use a single quadratic
        int const nknot = ntau;

        _allocateSpline(nknot);
        
        _knots[0] = tau[0];
        for (int i = 1; i < nknot;i++) {
            _knots[i] = tau[i];
            if(tau[i - 1] >= tau[i]) {
                throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                                  (boost::format("point %d and the next, %f %f, are out of order")
                                   % (i - 1)  % tau[i - 1] % tau[i]).str());
            }
        }
        
        if(ntau == 2) {
            _coeffs[0][0] = gtau[0];
            _coeffs[1][0] = (gtau[1] - gtau[0])/(tau[1] - tau[0]);
            _coeffs[2][0] = _coeffs[3][0] = 0;

            _coeffs[0][1] = gtau[1];
            _coeffs[1][1] = (gtau[1] - gtau[0])/(tau[1] - tau[0]);
            _coeffs[2][1] = _coeffs[3][1] = 0;
        } else {				/* must be 3 */
            double tmp = (tau[2]-tau[0])*(tau[2]-tau[1])*(tau[1]-tau[0]);
            _coeffs[0][0] = gtau[0];
            _coeffs[1][0] = ((gtau[1] - gtau[0])*pow(tau[2] - tau[0],2) -
                             (gtau[2] - gtau[0])*pow(tau[1] - tau[0],2))/tmp;
            _coeffs[2][0] = -2*((gtau[1] - gtau[0])*(tau[2] - tau[0]) -
                                (gtau[2] - gtau[0])*(tau[1] - tau[0]))/tmp;
            _coeffs[3][0] = 0;

            _coeffs[0][1] = gtau[1];
            _coeffs[1][1] = _coeffs[1][0] + (tau[1] - tau[0])*_coeffs[2][0];
            _coeffs[2][1] = _coeffs[2][0];
            _coeffs[3][1] = 0;

            _coeffs[0][2] = gtau[2];
            _coeffs[1][2] = _coeffs[1][0] + (tau[2] - tau[0])*_coeffs[2][0];
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
 *               = fsecnd = solution of linear system, namely the second derivatives of interpolant at tau
 *     s[4][...] = z = indicator of additional knots
 *     s[5][...] = 1/hsecnd(1,x) with x = z or = 1-z. see below.
 */
    std::vector<std::vector<double> > s(6); // scratch space

    for (int i = 0; i != 6; i++) {
        s[i].resize(ntau);
    }
/*
 * Construct delta tau and first and second (divided) differences of data 
 */

    for (int i = 0; i < ntau - 1; i++) {
        s[0][i] = tau[i + 1] - tau[i];
        if(s[0][i] <= 0.) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              (boost::format("point %d and the next, %f %f, are out of order")
                               % i % tau[i] % tau[i+1]).str());
        }
        s[3][i + 1] = (gtau[i + 1] - gtau[i])/s[0][i];
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
    s[1][1] = s[0][0]/3;

    int method;
    double gamma = gamma0;               // control the smoothing
    if(gamma <= 0) {
        method = 1;
    } else if(gamma > 3) {
        gamma -= 3;
        if (gamma >= 3) {
            gamma = 3*(1 - std::numeric_limits<double>::epsilon());
        }
        method = 3;
    } else {
        method = 2;
    }
    double const onemg3 = 1 - gamma/3;

    int nknot = ntau;                   // count number of knots
/*
 * Some compilers don't realise that the flow of control always initialises
 * these variables; so initialise them to placate e.g. gcc
 */
    double zeta = 0.0;
    double alpha = 0.0;
    double ratio = 0.0;

    //double c, d;
    //double z, denom, factr2;
    //double onemzt, zt2, del;


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
        if((method == 2 && s[3][i]*s[3][i + 1] >= 0) || method == 3) {
            double const temp = fabs(s[3][i + 1]);
            double const denom = fabs(s[3][i]) + temp;
            if(denom != 0) {
                z = temp/denom;
                if (fabs(z - 0.5) <= 1.0/6.0) {
                    z = 0.5;
                }
            }
        }
       
        s[4][i] = z;
/*
  set up part of the i-th equation which depends on the i-th interval
*/
        z_half = z - 0.5;
        if(z_half < 0) {
            zeta = gamma*z;
            onemzt = 1 - zeta;
            zt2 = zeta*zeta;
            double temp = onemg3/onemzt;
            alpha = (temp < 1 ? temp : 1);
            factor = zeta/(alpha*(zt2 - 1) + 1);
            s[5][i] = zeta*factor/6;
            s[1][i] += s[0][i]*((1 - alpha*onemzt)*factor/2 - s[5][i]);
/*
 * if z = 0 and the previous z = 1, then d[i] = 0. Since then
 * also u[i-1] = l[i+1] = 0, its value does not matter. Reset
 * d[i] = 1 to insure nonzero pivot in elimination.
 */
            if (s[1][i] <= 0.) {
                s[1][i] = 1;
            }
            s[2][i] = s[0][i]/6;
           
            if(z != 0) {			/* we'll get a new knot */
                nknot++;
            }
        } else if(z_half == 0) {
            s[1][i] += s[0][i]/3;
            s[2][i] = s[0][i]/6;
        } else {
            onemzt = gamma*(1 - z);
            zeta = 1 - onemzt;
            double const temp = onemg3/zeta;
            alpha = (temp < 1 ? temp : 1);
            factor = onemzt/(1 - alpha*zeta*(onemzt + 1));
            s[5][i] = onemzt*factor/6;
            s[1][i] += s[0][i]/3;
            s[2][i] = s[5][i]*s[0][i];
           
            if(onemzt != 0) {			/* we'll get a new knot */
                nknot++;
            }
        }
        if (i == 1) {
            s[4][0] = 0.5;
/*
 * the first two equations enforce continuity of the first and of
 * the third derivative across tau[1].
 */
            s[1][0] = s[0][0]/6;
            s[2][0] = s[1][1];
            entry3 = s[2][1];
            if(z_half < 0) {
                const double factr2 = zeta*(alpha*(zt2 - 1.) + 1.)/(alpha*(zeta*zt2 - 1.) + 1.);
                ratio = factr2*s[0][1]/s[1][0];
                s[1][1] = factr2*s[0][1] + s[0][0];
                s[2][1] = -factr2*s[0][0];
            } else if (z_half == 0) {
                ratio = s[0][1]/s[1][0];
                s[1][1] = s[0][1] + s[0][0];
                s[2][1] = -s[0][0];
            } else {
                ratio = s[0][1]/s[1][0];
                s[1][1] = s[0][1] + s[0][0];
                s[2][1] = -s[0][0]*6*alpha*s[5][1];
            }
/*
 * at this point, the first two equations read
 *              diag[0]*x0 +    u[0]*x1 + entry3*x2 = r[1]
 *       -ratio*diag[0]*x0 + diag[1]*x1 +   u[1]*x2 = 0
 * set r[0] = r[1] and eliminate x1 from the second equation
 */
            s[3][0] = s[3][1];
           
            s[1][1] += ratio*s[2][0];
            s[2][1] += ratio*entry3;
            s[3][1] = ratio*s[3][1];
        } else {
/*
 * the i-th equation enforces continuity of the first derivative
 * across tau[i]; it reads
 *         -ratio*diag[i-1]*x_{i-1} + diag[i]*x_i + u[i]*x_{i+1} = r[i]
 * eliminate x_{i-1} from this equation
 */
            s[1][i] += ratio*s[2][i - 1];
            s[3][i] += ratio*s[3][i - 1];
        }
/*
 * Set up the part of the next equation which depends on the i-th interval.
 */
        if(z_half < 0) {
            ratio = -s[5][i]*s[0][i]/s[1][i];
            s[1][i + 1] = s[0][i]/3;
        } else if(z_half == 0) {
            ratio = -(s[0][i]/6)/s[1][i];
            s[1][i + 1] = s[0][i]/3;
        } else {
            ratio = -(s[0][i]/6)/s[1][i];
            s[1][i + 1] = s[0][i]*((1 - zeta*alpha)*factor/2 - s[5][i]);
        }
    }
    
    s[4][ntau - 2] = 0.5;
/*
 * last two equations, which enforce continuity of third derivative and
 * of first derivative across tau[ntau - 2]
 */
    double const entry_ = ratio*s[2][ntau - 3] + s[1][ntau - 2] + s[0][ntau - 2]/3;
    s[1][ntau - 1] = s[0][ntau - 2]/6;
    s[3][ntau - 1] = ratio*s[3][ntau - 3] + s[3][ntau - 2];
    if (z_half < 0) {
        ratio = s[0][ntau - 2]*6*s[5][ntau - 3]*alpha/s[1][ntau - 3];
        s[1][ntau - 2] = ratio*s[2][ntau - 3] + s[0][ntau - 2] + s[0][ntau - 3];
        s[2][ntau - 2] = -s[0][ntau - 3];
    } else if (z_half == 0) {
        ratio = s[0][ntau - 2]/s[1][ntau - 3];
        s[1][ntau - 2] = ratio*s[2][ntau - 3] + s[0][ntau - 2] + s[0][ntau - 3];
        s[2][ntau - 2] = -s[0][ntau - 3];
    } else {
        const double factr2 = onemzt*(alpha*(onemzt*onemzt - 1) + 1)/(alpha*(onemzt*onemzt*onemzt - 1) + 1);
        ratio = factr2*s[0][ntau - 2]/s[1][ntau - 3];
        s[1][ntau - 2] = ratio*s[2][ntau - 3] + factr2*s[0][ntau - 3] + s[0][ntau - 2];
        s[2][ntau - 2] = -factr2*s[0][ntau - 3];
    }
/*
 * at this point, the last two equations read
 *             diag[i]*x_i +      u[i]*x_{i+1} = r[i]
 *      -ratio*diag[i]*x_i + diag[i+1]*x_{i+1} = r[i+1]
 *     eliminate x_i from last equation
 */
    s[3][ntau - 2] = ratio*s[3][ntau - 3];
    ratio = -entry_/s[1][ntau - 2];
    s[1][ntau - 1] += ratio*s[2][ntau - 2];
    s[3][ntau - 1] += ratio*s[3][ntau - 2];

/*
 * back substitution
 */
    s[3][ntau - 1] /= s[1][ntau - 1];
    for (int i = ntau - 2; i > 0; --i) {
        s[3][i] = (s[3][i] - s[2][i]*s[3][i + 1])/s[1][i];
    }

    s[3][0] = (s[3][0] - s[2][0]*s[3][1] - entry3*s[3][2])/s[1][0];
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
            if((z < 0.5 && z != 0.0) || (z > 0.5 && (1 - z) != 0.0)) {
                nknot++;
            }
        }
        assert (nknot == nknot0);
    }
#endif
    _allocateSpline(nknot);

    _knots[0] = tau[0];
    int j = 0;
    for (int i = 0; i < ntau - 1; ++i) {
        _coeffs[0][j] = gtau[i];
        _coeffs[2][j] = s[3][i];
        double const divdif = (gtau[i + 1] - gtau[i])/s[0][i];
        double z = s[4][i];
        double const z_half = z - 0.5;
        if (z_half < 0) {
            if (z == 0) {
                _coeffs[1][j] = divdif;
                _coeffs[2][j] = 0;
                _coeffs[3][j] = 0;
            } else {
                zeta = gamma*z;
                onemzt = 1 - zeta;
                double const c = s[3][i + 1]/6;
                double const d = s[3][i]*s[5][i];
                j++;
	      
                double const del = zeta*s[0][i];
                _knots[j] = tau[i] + del;
                zt2 = zeta*zeta;
                double temp = onemg3/onemzt;
                alpha = (temp < 1 ? temp : 1);
                factor = onemzt*onemzt*alpha;
                temp = s[0][i];
                _coeffs[0][j] = gtau[i] + divdif*del + temp*temp*(d*onemzt*(factor - 1) + c*zeta*(zt2 - 1));
                _coeffs[1][j] = divdif + s[0][i]*(d*(1 - 3*factor) + c*(3*zt2 - 1));
                _coeffs[2][j] = (d*alpha*onemzt + c*zeta)*6;
                _coeffs[3][j] = (c - d*alpha)*6/s[0][i];
                if(del*zt2 == 0) {
                    _coeffs[1][j - 1] = 0;	/* would be NaN in an */
                    _coeffs[3][j - 1] = 0;	/*              0-length interval */
                } else {
                    _coeffs[3][j - 1] = _coeffs[3][j] - d*6*(1 - alpha)/(del*zt2);
                    _coeffs[1][j - 1] = _coeffs[1][j] -
                        del*(_coeffs[2][j] - del/2*_coeffs[3][j - 1]);
                }
            }
        } else if (z_half == 0) {
            _coeffs[1][j] = divdif - s[0][i]*(s[3][i]*2 + s[3][i + 1])/6;
            _coeffs[3][j] = (s[3][i + 1] - s[3][i])/s[0][i];
        } else {
            onemzt = gamma*(1 - z);
            if (onemzt == 0) {
                _coeffs[1][j] = divdif;
                _coeffs[2][j] = 0;
                _coeffs[3][j] = 0;
            } else {
                zeta = 1 - onemzt;
                double const temp = onemg3/zeta;
                alpha = (temp < 1 ? temp : 1);
                double const c = s[3][i + 1]*s[5][i];
                double const d = s[3][i]/6;
                double const del = zeta*s[0][i];
                _knots[j + 1] = tau[i] + del;
                _coeffs[1][j] = divdif - s[0][i]*(2*d + c);
                _coeffs[3][j] = (c*alpha - d)*6/s[0][i];
                j++;

                _coeffs[3][j] = _coeffs[3][j - 1] +
                    (1 - alpha)*6*c/(s[0][i]*(onemzt*onemzt*onemzt));
                _coeffs[2][j] = _coeffs[2][j - 1] + del*_coeffs[3][j - 1];
                _coeffs[1][j] = _coeffs[1][j - 1] +
                    del*(_coeffs[2][j - 1] + del/2*_coeffs[3][j - 1]);
                _coeffs[0][j] = _coeffs[0][j - 1] +
                    del*(_coeffs[1][j - 1] +
                         del/2*(_coeffs[2][j - 1] + del/3*_coeffs[3][j - 1]));
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
        del*(_coeffs[1][nknot - 2] + del*(_coeffs[2][nknot - 2]/2 +
                                          del*_coeffs[3][nknot - 2]/6));
    _coeffs[1][nknot - 1] = _coeffs[1][nknot - 2] +
        del*(_coeffs[2][nknot - 2] + del*_coeffs[3][nknot - 2]/2);
    _coeffs[2][nknot - 1] = _coeffs[2][nknot - 2] + del*_coeffs[3][nknot - 2];
    _coeffs[3][nknot - 1] = _coeffs[3][nknot - 2];

    assert (j + 1 == nknot);
}

/*****************************************************************************/
/**
 * \brief * Fit a taut spline to a set of data, forcing the resulting spline to
 * obey S(x) = +-S(-x). The input points must have tau[] >= 0.
 *
 * See InterpolateTautSpline::InterpolateTautSpline() for a discussion of the algorithm, and
 * the meaning of the parameter gamma
 *
 * This is done by duplicating the input data for -ve x, so consider
 * carefully before using this function on many-thousand-point datasets
 */
void
InterpolateTautSpline::_calculateTautSplineEvenOdd(std::vector<double> const& _tau,
                                                   std::vector<double> const& _gtau,
                                                   double const gamma,
                                                   bool const even // ensure Even symmetry
                                      )
{
    const double *tau = &_tau[0];
    const double *gtau = &_gtau[0];
    int const ntau = _tau.size();       // size of tau and gtau, must be >= 2
    std::vector<double> x, y;           // tau and gtau, extended to -ve tau

    if(tau[0] == 0.0f) {
        int const np = 2*ntau - 1;
        x.resize(np);
        y.resize(np);

        x[ntau - 1] =  tau[0]; y[ntau - 1] = gtau[0];
        for (int i = 1; i != ntau; ++i) {
            if (even) {
                x[ntau - 1 + i] =  tau[i]; y[ntau - 1 + i] =  gtau[i];
                x[ntau - 1 - i] = -tau[i]; y[ntau - 1 - i] =  gtau[i];
            } else {
                x[ntau - 1 + i] =  tau[i]; y[ntau - 1 + i] =  gtau[i];
                x[ntau - 1 - i] = -tau[i]; y[ntau - 1 - i] = -gtau[i];
            }
        }
    } else {
        int const np = 2*ntau;
        x.resize(np);
        y.resize(np);

        for (int i = 0; i != ntau; ++i) {
            if (even) {
                x[ntau + i] =      tau[i]; y[ntau + i] =      gtau[i];
                x[ntau - 1 - i] = -tau[i]; y[ntau - 1 - i] =  gtau[i];
            } else {
                x[ntau + i] =      tau[i]; y[ntau + i] =      gtau[i];
                x[ntau - 1 - i] = -tau[i]; y[ntau - 1 - i] = -gtau[i];
            }
        }
    }

    InterpolateTautSpline sp(x, y, InterpolateControlTautSpline(_ictrl->_gamma)); // fit a taut spline to x, y
/*
 * Now repackage that spline to reflect the original points
 */
    int ii;
    for (ii = sp._knots.size() - 1; ii >= 0; --ii) {
        if(sp._knots[ii] < 0.0f) {
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

/*****************************************************************************/
/*
 * returns index i of first element of x >= z; the input i is an initial guess
 *
 * N.b. we could use std::lower_bound except that we use i as an initial hint
 */
namespace {
int
search_array(double z, double const *x, int n, int i)
{
   register int lo, hi, mid;
   double xm;

   if(i < 0 || i >= n) {		/* initial guess is useless */
      lo = -1;
      hi = n;
   } else {
      unsigned int step = 1;		/* how much to step up/down */

      if(z > x[i]) {			/* expand search upwards */
	 if(i == n - 1) {		/* off top of array */
	    return(n - 1);
	 }

	 lo = i; hi = lo + 1;
	 while(z >= x[hi]) {
	    lo = hi;
	    step += step;		/* double step size */
	    hi = lo + step;
	    if(hi >= n) {		/* reached top of array */
	       hi = n - 1;
	       break;
	    }
	 }
      } else {				/* expand it downwards */
	 if(i == 0) {			/* off bottom of array */
	    return(-1);
	 }

	 hi = i; lo = i - 1;
	 while(z < x[lo]) {
	    hi = lo;
	    step += step;		/* double step size */
	    lo = hi - step;
	    if(lo < 0) {		/* off bottom of array */
	       lo = -1;
	       break;
	    }
	 }
      }
   }
/*
 * perform bisection
 */
   while(hi - lo > 1) {
      mid = (lo + hi)/2;
      xm = x[mid];
      if(z <= xm) {
	 hi = mid;
      } else {
	 lo = mid;
      }
   }

   if(lo == -1) {			/* off the bottom */
      return(lo);
   }
/*
 * If there's a discontinuity (many knots at same x-value), choose the
 * largest
 */
   xm = x[lo];
   while(lo < n - 1 && x[lo + 1] == xm) lo++;

   return(lo);
}


/*****************************************************************************/
/*
 * Move the roots that lie in the specified range [x0,x1) from newRoots to roots
 */
void
keep_valid_roots(std::vector<double>& roots,
                 std::vector<double>& newRoots, double x0, double x1)
{
    for (unsigned int i = 0; i != newRoots.size(); ++i) {
        if(newRoots[i] >= x0 && newRoots[i] < x1) { // keep this root
            roots.push_back(newRoots[i]);
        }
    }

    newRoots.clear();
}

/*****************************************************************************/
/*
 * find the real roots of a quadratic ax^2 + bx + c = 0
 */
void
do_quadratic(double a, double b, double c, std::vector<double> & roots)
{
    if(::fabs(a) < std::numeric_limits<double>::epsilon()) {
        if(::fabs(b) >= std::numeric_limits<double>::epsilon()) {
            roots.push_back(-c/b);
        }
   } else {
      double const tmp = b*b - 4*a*c;
      
      if(tmp >= 0) {
          if (b >= 0) {
              roots.push_back((-b - sqrt(tmp))/(2*a));
          } else {
              roots.push_back((-b + sqrt(tmp))/(2*a));
          }
          roots.push_back(c/(a*roots[0]));
/*
 * sort roots
 */
          if(roots[0] > roots[1]) {
              double const tmp2 = roots[0]; roots[0] = roots[1]; roots[1] = tmp2;
          }
      }
   }
}

/*****************************************************************************/
/*
 * find the real roots of a cubic ax^3 + bx^2 + cx + d = 0
 */
void
do_cubic(double a, double b, double c, double d, std::vector<double> & roots)
{
    if (::fabs(a) < std::numeric_limits<double>::epsilon()) {
        do_quadratic(b, c, d, roots);
        return;
    }
    b /= a; c /= a; d /= a;

    double const q = (b*b - 3*c)/9;
    double const r = (2*b*b*b - 9*b*c + 27*d)/54;
    /*
     * n.b. note that the test for the number of roots is carried out on the
     * same variables as are used in (e.g.) the acos, as it is possible for
     * r*r < q*q*q && r > sq*sq*sq due to rounding.
     */
    double const sq = (q >= 0) ? sqrt(q) : -sqrt(-q);
    double const sq3 = sq*sq*sq;
    if(::fabs(r) < sq3) {                   // three real roots
        double const theta = ::acos(r/sq3); // sq3 cannot be zero
        
        roots.push_back(-2*sq*cos(theta/3) - b/3);
        roots.push_back(-2*sq*cos((theta + geom::TWOPI)/3) - b/3);
        roots.push_back(-2*sq*cos((theta - geom::TWOPI)/3) - b/3);
        /*
         * sort roots
         */
        if(roots[0] > roots[1]) {
            std::swap(roots[0], roots[1]);
        }
        if(roots[1] > roots[2]) {
            std::swap(roots[1], roots[2]);
        }
        if(roots[0] > roots[1]) {
            std::swap(roots[0], roots[1]);
        }
        
        return;
    } else if(::fabs(r) == sq3) {		/* no more than two real roots */
        double const aa = -((r < 0) ? -::pow(-r,1.0/3.0) : ::pow(r,1.0/3.0));

        if(::fabs(aa) < std::numeric_limits<double>::epsilon()) { /* degenerate case; one real root */
            roots.push_back(-b/3);
            return;
        } else {
            roots.push_back(2*aa - b/3);
            roots.push_back(-aa - b/3);
            
            if(roots[0] > roots[1]) {
                std::swap(roots[0], roots[1]);
            }

            return;
        }
    } else {				/* only one real root */
        double tmp = ::sqrt(r*r - (q > 0 ? sq3*sq3 : -sq3*sq3));
        tmp = r + (r < 0 ? -tmp : tmp);
        double const  aa = -((tmp < 0) ? -::pow(-tmp,1.0/3.0) : ::pow(tmp,1.0/3.0));
        double const bb = (fabs(aa) < std::numeric_limits<double>::epsilon()) ? 0 : q/aa;

        roots.push_back((aa + bb) - b/3);
#if 0
        roots.push_back(-(aa + bb)/2 - b/3);	// the real
        roots.push_back(::sqrt(3)/2*(aa - bb)); //         and imaginary parts of the complex roots
#endif
        return;
    }
}

}

/*****************************************************************************/
/*
 * \brief Find the roots of Spline - val = 0 in the range [x0, x1).
 *
 * \returns a vector of all the roots found
 */
std::vector<double>
InterpolateSdssSpline::roots(double const value,       ///< desired value
              double a,                 ///< specify desired range is [a,b)
              double const b            ///< specify desired range is [a,b)
             ) const
{
   /*
    * Strategy: we know that the interpolant has the form
    *    val = coef[0][i] +dx*(coef[1][i] + dx*(coef[2][i]/2 + dx*coef[3][i]/6))
    * so we can use the usual analytic solution for a cubic. Note that the
    * cubic quoted above returns dx, the distance from the previous knot,
    * rather than x itself
    */
    std::vector<double> roots;		/* the roots found */
    double x0 = a;                     // lower end of current range
    double const x1 = b;
    int const nknot = _knots.size();

    int i0 = search_array(x0, &_knots[0], nknot, -1);
    int const i1 = search_array(x1, &_knots[0], nknot, i0);
    assert (i1 >= i0 && i1 <= nknot - 1);

    std::vector<double> newRoots;       // the roots we find in some interval
/*
 * Deal with special case that x0 may be off one end or the other of
 * the array of knots.
 */
    if(i0 < 0) {				/* off bottom */
        i0 = 0;
        do_cubic(_coeffs[3][i0]/6, _coeffs[2][i0]/2, _coeffs[1][i0], _coeffs[0][i0] - value, newRoots);
        //
        // Could use
        //    std::transform(newRoots.begin(), newRoots.end(), newRoots.begin(),
        //                   std::tr1::bind(std::plus<double>(), _1, _knots[i0]));
        // but let's not
        //
        for (unsigned int j = 0; j != newRoots.size(); ++j) {
            newRoots[j] += _knots[i0];
        }
        keep_valid_roots(roots, newRoots, x0, _knots[i0]);

        x0 = _knots[i0];
    } else if(i0 >= nknot) {		/* off top */
        i0 = nknot - 1;
        assert (i0 >= 0);
        do_cubic(_coeffs[3][i0]/6, _coeffs[2][i0]/2, _coeffs[1][i0], _coeffs[0][i0] - value, newRoots);

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
    for (int i = i0;i <= i1;i++) {
        do_cubic(_coeffs[3][i]/6, _coeffs[2][i]/2, _coeffs[1][i], _coeffs[0][i] - value, newRoots);

        for (unsigned int j = 0; j != newRoots.size(); ++j) {
            newRoots[j] += _knots[i];
        }
        keep_valid_roots(roots, newRoots, ((i == i0) ? x0 : _knots[i]), ((i == i1) ? x1 : _knots[i + 1]));
    }

    return roots;
}

/************************************************************************************************************/
}}}}
