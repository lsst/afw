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
 
#if !defined(LSST_AFW_CAMERAGEOM_DISTORTION_H)
#define LSST_AFW_CAMERAGEOM_DISTORTION_H

#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"
#include "boost/tuple/tuple.hpp"

#include "lsst/afw/image.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"


/**
 * @file Distortion.cc
 * @brief Provide Classes to handle coordinate/moment distortion due to camera optics
 * @ingroup afw
 * @author Steve Bickerton
 *
 */

namespace lsst {
namespace afw {
namespace cameraGeom {

    
/**
 * @class Distort/Undistort coordinates, moments, and images according to camera optical distortion
 */
class Distortion {
public:
    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<const Distortion> ConstPtr;

    Distortion(int lanczosOrder=5) : _lanczosOrder(lanczosOrder) {}
    virtual ~Distortion() {}

    //virtual Distortion::Ptr clone() const { return Distortion::Ptr(new Distortion(*this)); }

    // distort a point
    lsst::afw::geom::Point2D distort(lsst::afw::geom::Point2D const &p); // = 0;
    lsst::afw::geom::Point2D undistort(lsst::afw::geom::Point2D const &p); // = 0;

    // distort an adaptive moment ... ie. a Quadrupole object
    lsst::afw::geom::ellipses::Quadrupole distort(lsst::afw::geom::Point2D const &p,
                                                  lsst::afw::geom::ellipses::Quadrupole const &Iqq); 
    lsst::afw::geom::ellipses::Quadrupole undistort(lsst::afw::geom::Point2D const &p,
                                                    lsst::afw::geom::ellipses::Quadrupole const &Iqq);

    // distort an image locally (ie. using the Quadrupole Linear Transform)
    template<typename PixelT>
    typename lsst::afw::image::Image<PixelT>::Ptr distort(lsst::afw::geom::Point2D const &p,
                                                 lsst::afw::image::Image<PixelT> const &img,
                                                 lsst::afw::geom::Point2D const &pix);
    template<typename PixelT>
    typename lsst::afw::image::Image<PixelT>::Ptr undistort(lsst::afw::geom::Point2D const &p,
                                                   lsst::afw::image::Image<PixelT> const &img,
                                                   lsst::afw::geom::Point2D const &pix);


    // all derived classes must define these two public methods
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward);
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward);
    
private: 
    template<typename PixelT>
    typename lsst::afw::image::Image<PixelT>::Ptr _warp(lsst::afw::geom::Point2D const &p,
                                                        lsst::afw::image::Image<PixelT> const &img,
                                                        lsst::afw::geom::Point2D const &pix,
                                                        bool forward);
    int _lanczosOrder;
};


/**
 * @class Offer a derived 'no-op' class with identity operators for all transforms
 */
class NullDistortion : public Distortion {
public:
    NullDistortion() :  Distortion() {}
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward);
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward);
};

/**
 * @class Handle optical distortions described by a polynomial function of radius
 */
class RadialPolyDistortion : public Distortion {
public:
    RadialPolyDistortion(std::vector<double> const &coeffs);

#if 0
    // these may be useful for debugging
    std::vector<double> getCoeffs()   {return _coeffs;   }
    std::vector<double> getICoeffs()  {return _icoeffs;  }
    std::vector<double> getDCoeffs()  {return _dcoeffs;  }
#endif
    
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward);
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward);
private:
    int _maxN;
    std::vector<double> _coeffs;
    std::vector<double> _icoeffs;
    std::vector<double> _dcoeffs;
    
    std::vector<double> _invert(std::vector<double> const &coeffs);
    std::vector<double> _deriv(std::vector<double> const &coeffs);
    lsst::afw::geom::Point2D _transform(lsst::afw::geom::Point2D const &p, bool forward=true);

    double _transformR(double r, std::vector<double> const &coeffs);
    double _iTransformR(double rp); 
    double _iTransformDr(double rp);
};


}}}
    
#endif
