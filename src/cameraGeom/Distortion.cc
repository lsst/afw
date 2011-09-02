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
 */
#include <cmath>
#include <algorithm>

#include "Eigen/Core.h"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "Eigen/Geometry"

#include "lsst/afw/cameraGeom/Distortion.h"

#include "lsst/pex/exceptions.h"
#include "boost/format.hpp"

namespace pexEx      = lsst::pex::exceptions;
namespace afwGeom    = lsst::afw::geom;
namespace cameraGeom = lsst::afw::cameraGeom;

/* Distortion ... have it be a null distortion*/

afwGeom::Point2D cameraGeom::Distortion::distort(afwGeom::Point2D const &p) {
    return afwGeom::Point2D(p.getX(), p.getY());
}
afwGeom::Point2D cameraGeom::Distortion::undistort(afwGeom::Point2D const &p)  {
    return afwGeom::Point2D(p.getX(), p.getY());
}
cameraGeom::Moment cameraGeom::Distortion::distort(afwGeom::Point2D const &p,
                                                   cameraGeom::Moment const &Iqq) {
    return cameraGeom::Moment(Iqq);
    
}
cameraGeom::Moment cameraGeom::Distortion::undistort(afwGeom::Point2D const &p,
                                                     cameraGeom::Moment const &Iqq) {
    return cameraGeom::Moment(Iqq);
}


/* NullDistortion  */

afwGeom::Point2D cameraGeom::NullDistortion::distort(afwGeom::Point2D const &p) {
    return afwGeom::Point2D(p.getX(), p.getY());
}
afwGeom::Point2D cameraGeom::NullDistortion::undistort(afwGeom::Point2D const &p)  {
    return afwGeom::Point2D(p.getX(), p.getY());
}
cameraGeom::Moment cameraGeom::NullDistortion::distort(afwGeom::Point2D const &p,
                                                       cameraGeom::Moment const &Iqq) {
    return cameraGeom::Moment(Iqq);
}
cameraGeom::Moment cameraGeom::NullDistortion::undistort(afwGeom::Point2D const &p,
                                                         cameraGeom::Moment const &Iqq) {
    return cameraGeom::Moment(Iqq);
}



/* RadialPolyDistortion  */

cameraGeom::RadialPolyDistortion::RadialPolyDistortion(std::vector<double> const &coeffs) :
    Distortion(), _maxN(7) {

    
    // initialize maxN-th order to zero
    _coeffs.resize(_maxN);
    //_icoeffs.resize(_maxN);
    std::fill(_coeffs.begin(), _coeffs.end(), 0.0);
    //std::fill(_icoeffs.begin(), _icoeffs.end(), 0.0);

    // fill in the actual coeffs
    std::copy(coeffs.begin(), coeffs.end(), _coeffs.begin());

    int n = coeffs.size();

    // if no terms given, default to no-op transform 
    if (n == 0) {
        _coeffs[0] = 0.0;
        _coeffs[1] = 1.0;
    }
    if (n > _maxN) {
        throw LSST_EXCEPT(pexEx::InvalidParameterException,
                          "Only 6th-order polynomials supported for radial distortion.");
    }

    _icoeffs  = _invert(_coeffs);
    _dcoeffs  = _deriv(_coeffs);
}
    
// We'll need the coeffs for the inverse of the input polynomial
// handle up to 6th order
// terms from Mathematical Handbook of Formula's and Tables, Spiegel & Liu.
// This is a taylor approx, so not perfect.  We'll use it to get close to the inverse
// and then use Newton Raphson to get to machine precision. (only needs 1 or 2 iterations)
std::vector<double> cameraGeom::RadialPolyDistortion::_invert(std::vector<double> const &c) {

    std::vector<double> ic(_maxN, 0.0);

    ic[0]  = 0.0; //-c[0]; // this inversion form only valid if c0 = 0.0

    if (c[1] != 0.0) {
        ic[1]  = 1.0;
        ic[1] /= c[1];
    }

    if (c[2] != 0.0) {
        ic[2]  = -c[2];
        ic[2] /= std::pow(c[1],3);
    }

    if (c[3] != 0.0) {
        ic[3]  = 2.0*c[2]*c[2] - c[1]*c[3];
        ic[3] /= std::pow(c[1],5);
    }

    if (c[4] != 0.0) {
        ic[4]  = 5.0*c[1]*c[2]*c[3] - 5.0*c[2]*c[2]*c[2] - c[1]*c[1]*c[4];
        ic[4] /= std::pow(c[1],7);
    }

    
    if (c[5] != 0.0) {
        ic[5]  = 6.0*c[1]*c[1]*c[2]*c[4] + 3.0*c[1]*c[1]*c[3]*c[3] - c[1]*c[1]*c[1]*c[5] + 
            14.0*std::pow(c[2], 4) - 21.0*c[1]*c[2]*c[2]*c[3];
        ic[5] /= std::pow(c[1],9);
    }
    

    if (c[6] != 0.0) {
        ic[6]  = 7.0*c[1]*c[1]*c[1]*c[2]*c[5] + 84.0*c[1]*c[2]*c[2]*c[2]*c[3] +
            7.0*c[1]*c[1]*c[1]*c[3]*c[4] - 28.0*c[1]*c[1]*c[2]*c[3]*c[3] - 
            std::pow(c[1], 4)*c[6] - 28.0*c[1]*c[1]*c[2]*c[2]*c[4] - 42.0*std::pow(c[2], 5);
        ic[6] /= std::pow(c[1],11);
    }
    
    return ic;
}

std::vector<double> cameraGeom::RadialPolyDistortion::_deriv(std::vector<double> const &coeffs) {
    std::vector<double> dcoeffs(_maxN, 0.0);
    for (int i=0; i<_maxN-1; i++) {
        dcoeffs[i] = (i + 1.0)*coeffs[i+1];
    }
    return dcoeffs;
}


double cameraGeom::RadialPolyDistortion::transformR(double r, std::vector<double> const &coeffs) {
    
    double rp = 0.0;
    for (std::vector<double>::size_type i = coeffs.size() - 1; i > 0; i--) {
        rp += coeffs[i];
        rp *= r;
    }
    rp += coeffs[0];
    return rp;
}


double cameraGeom::RadialPolyDistortion::iTransformR(double rp) {

    double tolerance = 1.0e-10;

    double r = transformR(rp, _icoeffs); // this gets us *very* close (with 0.01%)
    double err = 2.0*tolerance;
    int iter = 0, maxIter = 10;;

    // only 1 or 2 iterations needed to get to machine precision
    while ( (err > tolerance) && (iter < maxIter)) {
        double fr   = transformR(r, _coeffs) - rp;
        double dfr  = transformR(r, _dcoeffs);
        double rnew = r - fr/dfr;
        err = std::abs((rnew - r)/r);
        r = rnew;
        iter++;
    }
    return r;
}


double cameraGeom::RadialPolyDistortion::iTransformDr(double rp) {

    // The inverse function is a reflection in y=x
    // If r' = f(r), and r = g(r') is its inverse, then the derivative dg(r') = 1.0/df(g(r'))
    // sorry to use prime ' to denote a different coord 
    double r  = iTransformR(rp);
    double dr = 1.0/transformR(r, _dcoeffs);
    return dr;
}


afwGeom::Point2D cameraGeom::RadialPolyDistortion::_transform(afwGeom::Point2D const &p,
                                                              bool forward) {
    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);

    double rp = (forward) ? transformR(r, _coeffs) : iTransformR(r);

    return afwGeom::Point2D(rp*std::cos(t), rp*std::sin(t));
}



cameraGeom::Moment cameraGeom::RadialPolyDistortion::_transform(afwGeom::Point2D const &p,
                                                                cameraGeom::Moment const &iqq,
                                                                bool forward) {

    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);
    double cost = std::cos(t);
    double sint = std::sin(t);

    double dr = (forward) ? transformR(r, _dcoeffs) : iTransformDr(r); 

    Eigen::Matrix2d M, R, Rinv;
    M    <<   dr,  0.0,   0.0,  1.0;  // scaling matrix to stretch along x-axis
    R    << cost,  sint, -sint, cost;  // rotate from theta to along x-axis
    Rinv = R.inverse();
    Eigen::Matrix2d Mp = Rinv*M*R;

    double ixx = iqq.getIxx();
    double iyy = iqq.getIyy();
    double ixy = iqq.getIxy();

    Eigen::Matrix2d I;
    I << ixx, 0.5*ixy, 0.5*ixy, iyy;

    //Eigen::Matrix2d Inew = Mp.inverse()*I*(Mp.transpose()).inverse();
    Eigen::Matrix2d Inew = Mp*I*Mp.transpose();

    double iuu = Inew(0,0);
    double ivv = Inew(1,1);
    double iuv = (Inew(0,1) + Inew(1,0));
    
    return cameraGeom::Moment(iuu, ivv, iuv);
    
}

afwGeom::Point2D cameraGeom::RadialPolyDistortion::distort(afwGeom::Point2D const &p)  {
    return this->_transform(p, true);
}
afwGeom::Point2D cameraGeom::RadialPolyDistortion::undistort(afwGeom::Point2D const &p)  {
    return this->_transform(p, false);
}
cameraGeom::Moment cameraGeom::RadialPolyDistortion::distort(afwGeom::Point2D const &p,
                                                             cameraGeom::Moment const &Iqq) {
    return this->_transform(p, Iqq, true);
}
cameraGeom::Moment cameraGeom::RadialPolyDistortion::undistort(afwGeom::Point2D const &p,
                                                               cameraGeom::Moment const &Iqq) {
    return this->_transform(p, Iqq, false);
}
