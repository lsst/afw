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
#include "boost/format.hpp"
#include <map>

#include "lsst/afw/math/warpExposure.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/cameraGeom/Id.h"

/**
 * @file lsst/afw/cameraGeom/Distortion.h
 * @brief Provide Classes to handle coordinate/moment distortion due to camera optics
 * @ingroup afw
 * @author Steve Bickerton
 *
 */

namespace lsst {
namespace afw {
namespace cameraGeom {

    
/**
 *  Distort/Undistort coordinates, moments, and images according to camera optical distortion
 */
class Distortion {
public:
    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<const Distortion> ConstPtr;

    Distortion(int lanczosOrder=3) :
        _lanczosOrder(lanczosOrder),
        _maxShear(std::map<Id,double>()),
        _warpingControl(boost::str(boost::format("lanczos%d") % lanczosOrder), "", 10000) {}
    
    virtual ~Distortion() {}

    //virtual Distortion::Ptr clone() const { return Distortion::Ptr(new Distortion(*this)); }

    // accessors
    int getLanczosOrder() const { return _lanczosOrder; }
    // Cannot do this.  We cache the kernel.  If you need a different lanczos order
    // you must create a new Distortion object. (note that Kernel cannot be assigned-to
    // as operator=() is private. So we can't assign to a new Lanczos warping kernel.)
    //void setLanczosOrder(lanczosOrder);
    
    /**
     * @brief Set kernel cache size; 0 for no cache
     *
     * Note that the new cache is only computed when the kernel is first used to distort an image,
     * and only if the cache size has changed.
     */
    void setCacheSize(int cacheSize) {
        _warpingControl.setCacheSize(cacheSize);
    }
    
    // distort a point
    lsst::afw::geom::Point2D distort(lsst::afw::geom::Point2D const &p, Detector const &det) const;
    lsst::afw::geom::Point2D undistort(lsst::afw::geom::Point2D const &p, Detector const &det) const;

    // distort an adaptive moment ... ie. a Quadrupole object
    lsst::afw::geom::ellipses::Quadrupole distort(lsst::afw::geom::Point2D const &p,
                                                  lsst::afw::geom::ellipses::Quadrupole const &Iqq,
                                                  Detector const &det) const;  
    lsst::afw::geom::ellipses::Quadrupole undistort(lsst::afw::geom::Point2D const &p,
                                                    lsst::afw::geom::ellipses::Quadrupole const &Iqq,
                                                    Detector const &det) const; 

    // distort an image locally (ie. using the Quadrupole Linear Transform)
    template<typename ImageT>
    typename ImageT::Ptr distort(lsst::afw::geom::Point2D const &p,
                                 ImageT const &img,
                                 Detector const &det,
                                 typename ImageT::SinglePixel padValue=afw::image::pixel::PixelTypeTraits<typename ImageT::SinglePixel>::padValue()
                                ) const;
    template<typename ImageT>
    typename ImageT::Ptr undistort(lsst::afw::geom::Point2D const &p,
                                   ImageT const &img,
                                   Detector const &det,
                                   typename ImageT::SinglePixel padValue=afw::image::pixel::PixelTypeTraits<typename ImageT::SinglePixel>::padValue()
                                  ) const;

    double computeMaxShear(Detector const &det) const;
    
    // all derived classes must define these two public methods
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward) const;
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward) const;

    virtual std::string prynt() const { return std::string("Distortion Base Class"); }

    virtual std::vector<double> getCoeffs() const { return std::vector<double>(0); }
    //std::vector<double> getICoeffs()  {return _icoeffs;  }
    //std::vector<double> getDCoeffs()  {return _dcoeffs;  }
    
private: 

    virtual lsst::afw::geom::Point2D _distort(lsst::afw::geom::Point2D const &p,
                                              Detector const &det, bool foward) const;
    virtual lsst::afw::geom::ellipses::Quadrupole _distort(lsst::afw::geom::Point2D const &p,
                                                           lsst::afw::geom::ellipses::Quadrupole const &Iqq,
                                                           Detector const &det,
                                                           bool forward) const;

    template<typename ImageT>
    typename ImageT::Ptr _warp(lsst::afw::geom::Point2D const &p,
                               ImageT const &img,
                               Detector const &det, 
                               bool forard,
                               typename ImageT::SinglePixel padValue=afw::image::pixel::PixelTypeTraits<typename ImageT::SinglePixel>::padValue()
                               ) const;
    int _lanczosOrder;
    mutable std::map<Id,double> _maxShear;
    lsst::afw::math::WarpingControl _warpingControl;
};


/**
 *  Offer a derived 'no-op' class with identity operators for all transforms
 */
class NullDistortion : public Distortion {
public:
    NullDistortion() :  Distortion() {}
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward) const;
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward) const;
    virtual std::string prynt() const { return std::string("NullDistortion Derived Class"); }
    virtual std::vector<double> getCoeffs() const {return std::vector<double>(0);  }
    //std::vector<double> getICoeffs()  {return _icoeffs;  }
    //std::vector<double> getDCoeffs()  {return _dcoeffs;  }
private:
    virtual lsst::afw::geom::Point2D _distort(lsst::afw::geom::Point2D const &p,
                                              Detector const &, bool) const { return p; }
    virtual lsst::afw::geom::ellipses::Quadrupole _distort(lsst::afw::geom::Point2D const &,
                                                           lsst::afw::geom::ellipses::Quadrupole const &Iqq,
                                                           Detector const &,
                                                           bool) const { return Iqq; }
};

/**
 *  Handle optical distortions described by a polynomial function of radius
 */
class RadialPolyDistortion : public Distortion {
public:
    RadialPolyDistortion(std::vector<double> const &coeffs,
                         bool const coefficientsDistort=true,
                         int const lanczosOrder=5);
#if 1
    // these may be useful for debugging
    std::vector<double> getCoeffs() const  {return _coeffs;   }
    std::vector<double> getICoeffs() const {return _icoeffs;  }
    std::vector<double> getDCoeffs() const {return _dcoeffs;  }
#endif
    
    virtual lsst::afw::geom::LinearTransform computePointTransform(lsst::afw::geom::Point2D const &p,
                                                                   bool forward) const;
    virtual lsst::afw::geom::LinearTransform computeQuadrupoleTransform(lsst::afw::geom::Point2D const &p,
                                                                        bool forward) const;
    virtual std::string prynt() const { return std::string("RadialPolyDistortion Derived Class"); }
private:
    int _maxN;
    bool const _coefficientsDistort;  // if true, the provided coefficients \em distort positions
    std::vector<double> _coeffs;
    std::vector<double> _icoeffs;
    std::vector<double> _dcoeffs;
    
    std::vector<double> _invert(std::vector<double> const &coeffs) const;
    std::vector<double> _deriv(std::vector<double> const &coeffs) const;
    lsst::afw::geom::Point2D _transform(lsst::afw::geom::Point2D const &p, bool forward=true) const;

    double _transformR(double r, std::vector<double> const &coeffs) const;
    double _iTransformR(double rp) const; 
    double _iTransformDr(double rp) const;
};


}}}
    
#endif
