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
 * @file Distortion.cc
 * @brief Provide Classes to handle coordinate/moment distortion due to camera optics
 * @ingroup afw
 * @author Steve Bickerton
 *
 */

#include <cmath>
#include <algorithm>

#include "Eigen/Core.h"
#include "Eigen/LU"
#include "Eigen/SVD"
#include "Eigen/Geometry"

#include "lsst/afw/cameraGeom/Distortion.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/image.h"

#include "lsst/pex/exceptions.h"
#include "boost/format.hpp"

namespace pexEx      = lsst::pex::exceptions;
namespace afwImage   = lsst::afw::image;
namespace afwGeom    = lsst::afw::geom;
namespace afwMath    = lsst::afw::math;
namespace geomEllip  = lsst::afw::geom::ellipses;
namespace cameraGeom = lsst::afw::cameraGeom;

/* ========================================================================================*/
/* Distortion ... have it be a null distortion*/
/* ========================================================================================*/

/*
 * @brief method to distort a Point
 * - call the virtual method computePointTransform() to get the LinearTransform
 */
afwGeom::Point2D cameraGeom::Distortion::distort(
                                                 afwGeom::Point2D const &p ///< point to distort
                                                ) {
    afwGeom::LinearTransform linTran = this->computePointTransform(p, true);
    return linTran(p);
}
/*
 * @brief method to undistort a Point
 * - call the virtual method computePointTransform() to get the LinearTransform
 */
afwGeom::Point2D cameraGeom::Distortion::undistort(
                                                   afwGeom::Point2D const &p ///< point to undistort
                                                  )  {
    afwGeom::LinearTransform linTran = this->computePointTransform(p, false);
    return linTran(p);
}

/*
 * @brief method to distort a Quadrupole
 * - call the virtual method computeQuadrupoleTransform() to get the LinearTransform
 */
geomEllip::Quadrupole cameraGeom::Distortion::distort(
    afwGeom::Point2D const &p,          ///< location of quadrupole
    geomEllip::Quadrupole const &Iqq    ///< quad to distort
                                                     ) {
    afwGeom::LinearTransform linTran = this->computeQuadrupoleTransform(p, true);
    return Iqq.transform(linTran);
    
}
/*
 * @brief method to undistort a Quadrupole
 * - call the virtual method computeQuadrupoleTransform() to get the LinearTransform
 */
geomEllip::Quadrupole cameraGeom::Distortion::undistort(
    afwGeom::Point2D const &p,         ///< location of distorted quad
    geomEllip::Quadrupole const &Iqq   ///< distorted quad to be undistorted
                                                       ) {
    afwGeom::LinearTransform linTran = this->computeQuadrupoleTransform(p, false);
    return Iqq.transform(linTran);
}


/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to transform a Point
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::Distortion::computePointTransform(
    afwGeom::Point2D const &p, ///< Location of transform                                
    bool forward               ///< is this forward (undistorted to distorted) or reverse
                                                                      ) {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object
 * to transform a Quadrupole
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::Distortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p, ///< Location of transform                                
    bool forward               ///< is this forward (undistorted to distorted) or reverse
                                                                       ) {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/*
 * @brief private method to warp an image *locally*
 * - the transform is computed at point 'p'
 * - the point 'p' corresponds to pixel 'pix' in the input image
 *
 */
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr cameraGeom::Distortion::_warp(
    afwGeom::Point2D const &p,          ///< Location of transform                       
    afwImage::Image<PixelT> const &img, ///< Image to be (un)distorted                   
    afwGeom::Point2D const &pix,        ///< Pixel corresponding to location of transform
    bool forward                        ///< is this forward (undistorted to distorted) or reverse
                                                                   ) { 
    typename afwImage::Image<PixelT>::Ptr warpImg(new afwImage::Image<PixelT>(img, true));
    afwMath::LanczosWarpingKernel kernel(_lanczosOrder);
    afwGeom::LinearTransform linTran = this->computeQuadrupoleTransform(p, forward);
    afwMath::warpCenteredImage(*warpImg, img, kernel, linTran, pix);
    return warpImg;
}

/*
 * @brief method to distort (warp) an image locally 
 * - the transform is computed at point 'p'
 * - the point 'p' corresponds to pixel 'pix' in the input image
 */
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr cameraGeom::Distortion::distort(
    afwGeom::Point2D const &p,          ///< Location of transform                       
    afwImage::Image<PixelT> const &img, ///< Image to be distorted                       
    afwGeom::Point2D const &pix         ///< Pixel corresponding to location of transform
                                                                     ) {
    return _warp(p, img, pix, true);
}

/*
 * @brief method to undistort (via warp) an image locally 
 * - the transform is computed at point 'p'
 * - the point 'p' corresponds to pixel 'pix' in the input image
 */
template<typename PixelT>
typename afwImage::Image<PixelT>::Ptr cameraGeom::Distortion::undistort(
    afwGeom::Point2D const &p,           ///< Location of transform                       
    afwImage::Image<PixelT> const &img,  ///< Image to be distorted                       
    afwGeom::Point2D const &pix          ///< Pixel corresponding to location of transform
                                                                       ) {
    return _warp(p, img, pix, false);
}


/* ========================================================================================*/
/* NullDistortion  */
/* ========================================================================================*/

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to transform a Point
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::NullDistortion::computePointTransform(
    afwGeom::Point2D const &p,  ///< Location of transform                                
    bool forward                ///< is this forward (undistorted to distorted) or reverse
                                                                           ) {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to
 * transform a Quadrupole
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::NullDistortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p,   ///< Location of transform                                
    bool forward                 ///< is this forward (undistorted to distorted) or reverse
                                                                           ) {
    return afwGeom::LinearTransform(); // no args means an identity transform
}


/* ========================================================================================*/
/* RadialPolyDistortion  */
/* ========================================================================================*/

/*
 * @brief Constructor for RadialPolyDistortion
 */
cameraGeom::RadialPolyDistortion::RadialPolyDistortion(
    std::vector<double> const &coeffs ///<  polynomial coefficients a_i corresponding to r^i terms
                                                      ) :
    Distortion(), _maxN(7) {

    
    // initialize maxN-th order to zero
    _coeffs.resize(_maxN);
    std::fill(_coeffs.begin(), _coeffs.end(), 0.0);

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

    _icoeffs  = _invert(_coeffs); // the coefficients for the inverse transform  r = f(r')
    _dcoeffs  = _deriv(_coeffs);  // the coefficients for derivative dr = f(r)
}

/*
 * @brief Invert the coefficients for the polynomial.
 *
 * We'll need the coeffs for the inverse of the input polynomial
 * handle up to 6th order
 * terms from Mathematical Handbook of Formula's and Tables, Spiegel & Liu.
 * This is a taylor approx, so not perfect.  We'll use it to get close to the inverse
 * and then use Newton-Raphson to get to machine precision. (only needs 1 or 2 iterations)
 */
std::vector<double> cameraGeom::RadialPolyDistortion::_invert(
    std::vector<double> const &c ///< polynomial coefficients a_i corresponding to r^i terms
                                                             ) {
    
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

/*
 * @brief (private method) Compute the coefficients, dcoeffs_i, for the derivative dr=sum_i (dcoeffs_i*r^i)
 */
std::vector<double> cameraGeom::RadialPolyDistortion::_deriv(
    std::vector<double> const &coeffs ///< polynomial coefficients a_i corresponding to r^i terms
                                                            ) {
    std::vector<double> dcoeffs(_maxN, 0.0);
    for (int i=0; i<_maxN-1; i++) {
        dcoeffs[i] = (i + 1.0)*coeffs[i+1];
    }
    return dcoeffs;
}

///<
///<

/*
 * @brief (private method) Transform R ... r' = sum_i (coeffs_i * r^i) for the coeffs provided
 * - if we call this with _coeffs, we get the forward transform r'(r)
 * - if we call with _icoeffs, we get the inverse transform r(r')
 *   NOTE: the inverse transform isn't perfect needs newton-raphson iteration to get to machine prec.
 */
double cameraGeom::RadialPolyDistortion::_transformR(
    double r,                          ///< radius to transform
    std::vector<double> const &coeffs  ///< coeffs to use for the polynomial
                                                    ) {
    
    double rp = 0.0;
    for (std::vector<double>::size_type i = coeffs.size() - 1; i > 0; i--) {
        rp += coeffs[i];
        rp *= r;
    }
    rp += coeffs[0];
    return rp;
}

/*
 * @brief (private method) Compute the inverse transform of R: r = sum_i (icoeffs_i * r'^i)
 * - use _transformR with _icoeffs to get close, then do a few iterations of newton-raphson
 */
double cameraGeom::RadialPolyDistortion::_iTransformR(
                                                      double rp ///< radius to un-transform
                                                     ) { 

    double tolerance = 1.0e-10;

    double r = _transformR(rp, _icoeffs); // this gets us *very* close (with 0.01%)
    double err = 2.0*tolerance;
    int iter = 0, maxIter = 10;;

    // only 1 or 2 iterations needed to get to machine precision
    while ( (err > tolerance) && (iter < maxIter)) {
        double fr   = _transformR(r, _coeffs) - rp;
        double dfr  = _transformR(r, _dcoeffs);
        double rnew = r - fr/dfr;
        err = std::abs((rnew - r)/r);
        r = rnew;
        iter++;
    }
    return r;
}

/*
 * @brief (private method) Compute the derivative in distorted coords: dr' = f(r')
 */
double cameraGeom::RadialPolyDistortion::_iTransformDr(
                                                       double rp ///< r' radius where you want dr'
                                                      ) {

    // The inverse function is a reflection in y=x
    // If r' = f(r), and r = g(r') is its inverse, then the derivative dg(r') = 1.0/df(g(r'))
    // sorry to use prime ' to denote a different coord 
    double r  = _iTransformR(rp);
    double dr = 1.0/_transformR(r, _dcoeffs);
    return dr;
}

/*
 * @brief Compute the LinearTransform object, L, to use for the matrix operation r' = L r on a point.
 */
afwGeom::LinearTransform cameraGeom::RadialPolyDistortion::computePointTransform(
    afwGeom::Point2D const &p,   ///< Location of transform                                
    bool forward                 ///< is this forward (undistorted to distorted) or reverse
                                                                                ) {
    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);

    double rp = (forward) ? _transformR(r, _coeffs) : _iTransformR(r);

    double scale = (r > 0.0) ? rp/r : 1.0;
    return afwGeom::LinearTransform().makeScaling(scale);
}


/*
 * @brief Compute the LinearTransform object, L, to use to map a quadrupole L to L'
 */
afwGeom::LinearTransform cameraGeom::RadialPolyDistortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p,   ///< Location of transform                                
    bool forward                 ///< is this forward (undistorted to distorted) or reverse
                                                                                     ) {

    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);
    double cost = std::cos(t);
    double sint = std::sin(t);

    double dr = (forward) ? _transformR(r, _dcoeffs) : _iTransformDr(r); 

    Eigen::Matrix2d M, R, Rinv;
    M    <<   dr,  0.0,   0.0,  1.0;  // scaling matrix to stretch along x-axis
    R    << cost,  sint, -sint, cost;  // rotate from theta to along x-axis
    Rinv = R.inverse();
    Eigen::Matrix2d Mp = Rinv*M*R;
    
    return afwGeom::LinearTransform(Mp);
}


// explicit instantiations
/*
 */
#define INSTANTIATE(TYPE)                                               \
    template afwImage::Image<TYPE>::Ptr cameraGeom::Distortion::_warp(afwGeom::Point2D const &p, \
                                                                      afwImage::Image<TYPE> const &img, \
                                                                      afwGeom::Point2D const &pix, \
                                                                      bool forward); \
    template afwImage::Image<TYPE>::Ptr cameraGeom::Distortion::distort(afwGeom::Point2D const &p, \
                                                                        afwImage::Image<TYPE> const &img, \
                                                                        afwGeom::Point2D const &pix); \
    template afwImage::Image<TYPE>::Ptr cameraGeom::Distortion::undistort(afwGeom::Point2D const &p, \
                                                                          afwImage::Image<TYPE> const &img, \
                                                                          afwGeom::Point2D const &pix);

INSTANTIATE(float);
INSTANTIATE(double);
