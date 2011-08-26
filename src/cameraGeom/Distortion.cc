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

    _icoeffs = _invert(_coeffs);
    _dcoeffs = _deriv(_coeffs);

    // can't invert _dcoeffs as it has non-zero c0 term
    //_idcoeffs = _invert(_dcoeffs);
    // --> must take derivative of _icoeffs instead:
    _idcoeffs = _deriv(_icoeffs);
    
}
    
    // We'll need the coeffs for the inverse of the input polynomial

    // handle up to 6th order
    // terms from Mathematical Handbook of Formula's and Tables, Spiegel & Liu.

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
        dcoeffs[i] = (i+1)*coeffs[i+1];
    }
    return dcoeffs;
}


double cameraGeom::RadialPolyDistortion::_transformR(double r, std::vector<double> const &coeffs) {
    
    double rp = 0.0;
    for (std::vector<double>::size_type i = coeffs.size() - 1; i > 0; i--) {
        rp += coeffs[i];
        rp *= r;
    }
    rp += coeffs[0];
    return rp;
}


afwGeom::Point2D cameraGeom::RadialPolyDistortion::_transform(afwGeom::Point2D const &p, std::vector<double> const &coeffs) {
    
    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);
    
    double rp = _transformR(r, coeffs);
    return afwGeom::Point2D(rp*std::cos(t), rp*std::sin(t));
}



cameraGeom::Moment cameraGeom::RadialPolyDistortion::_transform(afwGeom::Point2D const &p, cameraGeom::Moment const &iqq, std::vector<double> const &coeffs) {

    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);
    double cost = std::cos(t);
    double sint = std::sin(t);
    double dr = _transformR(r, coeffs);

    Eigen::Matrix2d M, R, Rinv;
    M    <<   dr,  0.0,   0.0,  1.0;  // scaling matrix to stretch along x-axis
    R    << cost,  -sint, sint, cost;  // rotate from theta to along x-axis
    Rinv = R.inverse();
    Eigen::Matrix2d Mp = Rinv*M*R;

    double ixx = iqq.getIxx();
    double iyy = iqq.getIyy();
    double ixy = iqq.getIxy();
    double alpha = Mp(0,0);
    double beta  = Mp(0,1);
    double gamma = Mp(1,0);
    double delta = Mp(1,1);

    //std::cout << alpha << " " << beta <<" "<<gamma<<" "<<delta<<std::endl;
    //double iuu = alpha*alpha*ixx +   beta*beta*iyy + 2.0*alpha*beta*ixy;
    //double ivv = gamma*gamma*ixx + delta*delta*iyy + 2.0*gamma*delta*ixy;
    //double iuv = alpha*gamma*ixx +  delta*beta*iyy + (alpha*delta+beta*gamma)*ixy;
    //double denom = std::sqrt(std::pow((ixx - iyy)/2.0, 2) + ixy*ixy);
    //double cxx = iyy/denom;
    //double cyy = ixx/denom;
    //double cxy = -2.0*ixy/denom;
    
    Eigen::Matrix2d I;
    I << ixx, ixy, ixy, iyy;

    Eigen::Matrix2d Inew = Mp.inverse()*I*(Mp.transpose()).inverse();

    //cxx = Inew(0,0);
    //cyy = Inew(1,1);
    //cxy = Inew(0,1) + Inew(1,0);

    //denom = std::sqrt(std::pow((cxx - cyy)/2.0, 2) + cxy*cxy);
    //ixx = cyy/denom;
    //iyy = cxx/denom;
    //ixy = -2.0*cxy/denom;

    double iuu = Inew(0,0);
    double ivv = Inew(1,1);
    double iuv = 0.5*(Inew(0,1) + Inew(1,0));
    return cameraGeom::Moment(iuu, ivv, iuv);
    
}


afwGeom::Point2D cameraGeom::RadialPolyDistortion::distort(afwGeom::Point2D const &p)  {
    return this->_transform(p, _coeffs);
}
afwGeom::Point2D cameraGeom::RadialPolyDistortion::undistort(afwGeom::Point2D const &p)  {
    return this->_transform(p, _icoeffs);
}
cameraGeom::Moment cameraGeom::RadialPolyDistortion::distort(afwGeom::Point2D const &p,
                                                             cameraGeom::Moment const &Iqq) {
    return this->_transform(p, Iqq, _dcoeffs);
}
cameraGeom::Moment cameraGeom::RadialPolyDistortion::undistort(afwGeom::Point2D const &p,
                                                               cameraGeom::Moment const &Iqq) {
    return this->_transform(p, Iqq, _idcoeffs);
}
