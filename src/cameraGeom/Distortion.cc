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
 * @file src/cameraGeom/Distortion.cc
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
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses.h" 
#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/image/MaskedImage.h"

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
 * @brief method to distort or undistort a Point in a detector
 *
 * Here we handle converting p to focal plane coordinate 'position'
 * We then compute the linear transform appropriate at position, and transform the coordinate
 * Finally, we convert position back to a local pixel in the detector, and return
 */
afwGeom::Point2D cameraGeom::Distortion::_distort(
                                                 afwGeom::Point2D const &p, ///< point to distort (pixels)
                                                 cameraGeom::Detector const &det, ///< detector containing p
                                                 bool forward  ///< is this a forward transform
                                                ) const {
    // convert p to focal plane pixels
    afwGeom::Point2D pixPos = det.getPositionFromPixel(p).getPixels(det.getPixelSize());
    // get the linear transform
    afwGeom::LinearTransform linTran = this->computePointTransform(pixPos, forward);
    // transform fpPixels and convert to mm
    afwGeom::Extent2D posTrans = afwGeom::Extent2D(linTran(pixPos))*det.getPixelSize();
    // compute the new pixel coord
    afwGeom::Point2D pix = det.getPixelFromPosition(cameraGeom::FpPoint(posTrans));
    
    return pix;
}


/*
 * @brief method to distort a point p in a detector by forwarding the call to private _distort()
 */
afwGeom::Point2D cameraGeom::Distortion::distort(
                                                 afwGeom::Point2D const &p, ///< point to distort (pixels)
                                                 cameraGeom::Detector const &det ///< detector containing p
                                                ) const {
    return _distort(p, det, true);
}
/*
 * @brief method to undistort a Point in a detector by forwarding the call to private _distort()
 */
afwGeom::Point2D cameraGeom::Distortion::undistort(
                                                   afwGeom::Point2D const &p, ///< point to undistort
                                                   cameraGeom::Detector const &det ///<detector containing p
                                                  ) const  {
    // convert 
    return _distort(p, det, false);
}

/*
 * @brief method to distort a Quadrupole at position p in a detector
 *
 * Convert the pixel coord to a position in the focal plane
 * Compute the linear Transform to transform a Quadrupole at that position
 * return the transformed value.
 */
geomEllip::Quadrupole cameraGeom::Distortion::_distort(
    afwGeom::Point2D const &p,          ///< location of quadrupole (pixels)
    geomEllip::Quadrupole const &Iqq,    ///< quad to distort
    cameraGeom::Detector const &det,     ///< detector wherein lives the point
    bool forward                        ///< forward transform?
                                                     ) const {
    // convert p to focal plane pixels
    afwGeom::Point2D pixPos = det.getPositionFromPixel(p).getPixels(det.getPixelSize());
    // get the linear transform
    afwGeom::LinearTransform linTran = this->computeQuadrupoleTransform(pixPos, forward);
    
    return Iqq.transform(linTran);
    
}

/*
 * @brief method to distort a Quadrupole (forward to _distort())
 */
geomEllip::Quadrupole cameraGeom::Distortion::distort(
    afwGeom::Point2D const &p,          ///< location of quadrupole (pixels)
    geomEllip::Quadrupole const &Iqq,    ///< quad to distort
    cameraGeom::Detector const &det     ///< detector wherein lives the point
                                                     ) const {
    return _distort(p, Iqq, det, true);
}
/*
 * @brief method to undistort a Quadrupole (forward to _distort())
 */
geomEllip::Quadrupole cameraGeom::Distortion::undistort(
    afwGeom::Point2D const &p,         ///< location of distorted quad
    geomEllip::Quadrupole const &Iqq,   ///< distorted quad to be undistorted
    cameraGeom::Detector const &det     ///< detector wherein lives the point
                                                       ) const {
    return _distort(p, Iqq, det, false);
}


/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to transform a Point
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::Distortion::computePointTransform(
    afwGeom::Point2D const &p, ///< Location of transform                                
    bool forward               ///< is this forward (undistorted to distorted) or reverse
                                                                      ) const {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/**
 * @brief compute the maximum fractional shear expected for a pixel anywhere in a bbox
 *
 * eg. a value of 1.05 suggests a pixel in this detector may be distorted by up to 5%
 * This is useful to determine the border/buffer to keep around a PSF image.
 * You want PSF images in one detector to be the same size so you can do a PCA on them,
 * and you need enough border so distorting won't leave you with undefined pixels.
 *
 */
double cameraGeom::Distortion::computeMaxShear(
                                               cameraGeom::Detector const &det
                                              ) const {

    
    // don't recompute it if we already have it (it was initialized to NaN)
    if ( ! _maxShear.count(det.getId()) ) {
        
        // go to each corner of the box
        afwGeom::Box2I box = det.getAllPixels(); // in pixels
    
        std::vector<afwGeom::Point2D> corners;
        corners.push_back(afwGeom::Point2D(box.getMinX(), box.getMinY()));
        corners.push_back(afwGeom::Point2D(box.getMinX(), box.getMaxY()));
        corners.push_back(afwGeom::Point2D(box.getMaxX(), box.getMinY()));
        corners.push_back(afwGeom::Point2D(box.getMaxX(), box.getMaxY()));

        double maxShear = 0.0;
        for (std::vector<afwGeom::Point2D>::iterator it=corners.begin(); it != corners.end(); ++it) {
        
            // put a unit circle Quadrupole there
            geomEllip::Quadrupole q;
        
            // shear it with distort() method
            geomEllip::Quadrupole qShear = this->distort(afwGeom::Point2D(*it), q, det);
            geomEllip::Axes axes(qShear);
            
            // compute A for the sheared Quad
            double a = axes.getA();
        
            // get the max A
            if (a > maxShear) {
                maxShear = a;
            }
        }
        // cache maxShear
        _maxShear[det.getId()] = maxShear;
    }
    
    // return
    return _maxShear[det.getId()];
}

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object
 * to transform a Quadrupole
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::Distortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p, ///< Location of transform                                
    bool forward               ///< is this forward (undistorted to distorted) or reverse
                                                                       ) const {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/*
 * @brief private method to warp an image *locally*
 * - the transform is computed at pixel 'p' in detector
 */
template<typename ImageT>
typename ImageT::Ptr cameraGeom::Distortion::_warp(
    afwGeom::Point2D const &p,             ///< Pixel Coordinate
    ImageT const &img,                     ///< Image to be (un)distorted                   
    Detector const &det,                   ///< Detector describing location of image in focal plane
    bool forward,                          ///< is this forward (distorting) or reverse (undistorting)
    typename ImageT::SinglePixel padValue  ///< Set unspecified pixels to this value
                                                  ) const {

    // location of p in the focal plane ... in units of pixels
    afwGeom::Point2D pos = det.getPositionFromPixel(p).getPixels(det.getPixelSize());
    int nx = img.getWidth();
    int ny = img.getHeight();
    
    // make an image
    typename ImageT::Ptr warpImg(new ImageT(nx, ny));
    warpImg->setXY0(img.getXY0());
    
    // call the warp code 
    //afwMath::LanczosWarpingKernel kernel(_lanczosOrder);
    afwGeom::LinearTransform linTran = this->computeQuadrupoleTransform(pos, forward);
    afwMath::warpCenteredImage(*warpImg, img, linTran, p, _warpingControl, padValue);

    return warpImg;
}

/*
 * @brief method to distort (warp) an image locally 
 * - the transform is computed at pixel 'p' in the image.
 */
template<typename ImageT>
typename ImageT::Ptr cameraGeom::Distortion::distort(
    afwGeom::Point2D const &p,              ///< Pixel Coordinate in image
    ImageT const &img,                      ///< Image to be distorted                       
    cameraGeom::Detector const &det,        ///< Detector describing location of image in focal plane
    typename ImageT::SinglePixel padValue   ///< Set unspecified pixels to this value
                                                                     ) const {
    return _warp(p, img, det, true, padValue);
}

/*
 * @brief method to undistort (via warp) an image locally 
 * - the transform is computed at pixel 'p' in the image
 */
template<typename ImageT>
typename ImageT::Ptr cameraGeom::Distortion::undistort(
    afwGeom::Point2D const &p,            ///< Pixel Coordinate in the image
    ImageT const &img,                    ///< Image to be distorted                       
    cameraGeom::Detector const &det,      ///< Detector describing location of image in focal plane
    typename ImageT::SinglePixel padValue                ///< Set unspecified pixels to this value
                                                                       ) const {
    return _warp(p, img, det, false, padValue);
}


/* ========================================================================================*/
/* NullDistortion  */
/* ========================================================================================*/

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to transform a Point
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::NullDistortion::computePointTransform(
    afwGeom::Point2D const &p,  ///< Focal plane location of transform in pixels
    bool forward                ///< is this forward (undistorted to distorted) or reverse
                                                                           ) const {
    return afwGeom::LinearTransform(); // no args means an identity transform
}

/*
 * @brief virtual method computePointTransform to compute the LinearTransform object to
 * transform a Quadrupole
 * In this case, we return an identity matrix.
 */
afwGeom::LinearTransform cameraGeom::NullDistortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p,   ///< Focal plane location of transform in pixels
    bool forward                 ///< is this forward (undistorted to distorted) or reverse
                                                                           ) const {
    return afwGeom::LinearTransform(); // no args means an identity transform
}


/* ========================================================================================*/
/* RadialPolyDistortion  */
/* ========================================================================================*/

/*
 * @brief Constructor for RadialPolyDistortion
 */
cameraGeom::RadialPolyDistortion::RadialPolyDistortion(
    std::vector<double> const &coeffs, ///<  polynomial coefficients a_i corresponding to r^i terms
    bool const coefficientsDistort,    ///< coefficients apply to \em forward transform
    int lanczosOrder                   ///< lanczos order to use for interpolation kernels
                                                      ) :
    Distortion(lanczosOrder), _maxN(7), _coefficientsDistort(coefficientsDistort)
{
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
                                                             ) const {
    
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
                                                            ) const {
    std::vector<double> dcoeffs(_maxN, 0.0);
    for (int i=0; i<_maxN-1; i++) {
        dcoeffs[i] = (i + 1.0)*coeffs[i+1];
    }
    return dcoeffs;
}

/*
 * @brief (private method) Transform R ... r' = sum_i (coeffs_i * r^i) for the coeffs provided
 * - if we call this with _coeffs, we get the forward transform r'(r)
 * - if we call with _icoeffs, we get the inverse transform r(r')
 *   NOTE: the inverse transform isn't perfect needs newton-raphson iteration to get to machine prec.
 */
double cameraGeom::RadialPolyDistortion::_transformR(
    double r,                          ///< radius to transform
    std::vector<double> const &coeffs  ///< coeffs to use for the polynomial
                                                    ) const {
    
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
                                                     ) const { 

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
                                                      ) const {

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
                                                                                ) const {
    double x = p.getX();
    double y = p.getY();
    double r = ::hypot(x, y);

    double rp = (_coefficientsDistort ? forward : !forward) ? _transformR(r, _coeffs) : _iTransformR(r);

    double scale = (r > 0.0) ? rp/r : 1.0;
    return afwGeom::LinearTransform().makeScaling(scale);
}


/*
 * @brief Compute the LinearTransform object, L, to use to map a quadrupole L to L'
 */
afwGeom::LinearTransform cameraGeom::RadialPolyDistortion::computeQuadrupoleTransform(
    afwGeom::Point2D const &p,   ///< Location of transform                                
    bool forward                 ///< is this forward (undistorted to distorted) or reverse
                                                                                     ) const {

    double x = p.getX();
    double y = p.getY();
    double r = std::sqrt(x*x + y*y);
    double t = std::atan2(y, x);
    double cost = std::cos(t);
    double sint = std::sin(t);

    double dr = (_coefficientsDistort ? forward : !forward) ? _transformR(r, _dcoeffs) : _iTransformDr(r); 

    afwGeom::LinearTransform::Matrix M, R, Rinv;
    M    <<   dr,  0.0,   0.0,  1.0;  // scaling matrix to stretch along x-axis
    R    << cost,  sint, -sint, cost;  // rotate from theta to along x-axis
    Rinv = R.inverse();
    afwGeom::LinearTransform::Matrix Mp = Rinv*M*R;

    afwGeom::LinearTransform lintran(Mp);
    
    return lintran;
}

#ifndef DOXYGEN

// explicit instantiations
/*
 */
#define INSTANTIATE(IMTYPE, TYPE)                                        \
    template afwImage::IMTYPE<TYPE>::Ptr cameraGeom::Distortion::_warp(afwGeom::Point2D const &p, \
                                                                       afwImage::IMTYPE<TYPE> const &img, \
                                                                       cameraGeom::Detector const &det, \
                                                                       bool foward, \
                                                                       afwImage::IMTYPE<TYPE>::SinglePixel padValue) const; \
    template afwImage::IMTYPE<TYPE>::Ptr cameraGeom::Distortion::distort(afwGeom::Point2D const &p, \
                                                                        afwImage::IMTYPE<TYPE> const &img, \
                                                                         cameraGeom::Detector const &det, \
                                                                         afwImage::IMTYPE<TYPE>::SinglePixel padValue) const; \
    template afwImage::IMTYPE<TYPE>::Ptr cameraGeom::Distortion::undistort(afwGeom::Point2D const &p, \
                                                                          afwImage::IMTYPE<TYPE> const &img, \
                                                                           cameraGeom::Detector const &det, \
                                                                           afwImage::IMTYPE<TYPE>::SinglePixel padValue) const;

INSTANTIATE(Image, float);
INSTANTIATE(Image, double);
INSTANTIATE(MaskedImage, float);
INSTANTIATE(MaskedImage, double);

#endif // !DOXYGEN
