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

#include "lsst/afw/geom/Point.h"


/**
 * @file
 *
 * Describe the Distortion for a Detector
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

class Moment { //: public boost::tuple<double, double, double> {
public:
    Moment(double ixx, double iyy, double ixy) : _ixx(ixx), _iyy(iyy), _ixy(ixy) {} //boost::tuple<double, double, double>(ixx, iyy, ixy) {}
    //Moment(Moment const &iqq) : boost::tuple<double, double, double>(iqq.get<0>(), iqq.get<1>(), iqq.get<2>()) {};
    //iqq.getIxx(), iqq.getIyy, iqq.getIxy()) {}
    Moment(Moment const &iqq) : _ixx(iqq.getIxx()), _iyy(iqq.getIyy()), _ixy(iqq.getIxy()) {} //boost::tuple<double, double, double>(iqq.getIxx(), iqq.getIyy(), iqq.getIxy()) {}; //iqq.getIxx(), iqq.getIyy, iqq.getIxy()) {}
    //double setIxx(double ixx) { this->set<0>(ixx); }
    //double setIyy(double iyy) { this->set<1>(iyy); }
    //double setIxy(double ixy) { this->set<2>(ixy); }
    double getIxx() const { return _ixx; } //this->get<0>(); }
    double getIyy() const { return _iyy; } //this->get<1>(); }
    double getIxy() const { return _ixy; } //this->get<2>(); }
private:
    double _ixx, _iyy, _ixy;
};

/**
 * Describe a set of Detectors that are physically closely related (e.g. on the same invar support)
 */
class Distortion {
public:
    typedef boost::shared_ptr<Distortion> Ptr;
    typedef boost::shared_ptr<const Distortion> ConstPtr;

    Distortion() {}
    virtual ~Distortion() {}

    //virtual Distortion::Ptr clone() const { return Distortion::Ptr(new Distortion(*this)); }
    
    virtual lsst::afw::geom::Point2D distort(lsst::afw::geom::Point2D const &p); // = 0;
    virtual lsst::afw::geom::Point2D undistort(lsst::afw::geom::Point2D const &p); // = 0;

    virtual Moment distort(lsst::afw::geom::Point2D const &p, Moment const &Iqq); 
    virtual Moment undistort(lsst::afw::geom::Point2D const &p, Moment const &Iqq);

};


class NullDistortion : public Distortion {
public:
    NullDistortion() :  Distortion() {}
    
    lsst::afw::geom::Point2D distort(lsst::afw::geom::Point2D const &p);
    lsst::afw::geom::Point2D undistort(lsst::afw::geom::Point2D const &p);

    Moment distort(lsst::afw::geom::Point2D const &p, Moment const &Iqq); 
    Moment undistort(lsst::afw::geom::Point2D const &p, Moment const &Iqq);
};


class RadialPolyDistortion : public Distortion {
public:
    RadialPolyDistortion(std::vector<double> const &coeffs);

    lsst::afw::geom::Point2D distort(lsst::afw::geom::Point2D const &p);
    lsst::afw::geom::Point2D undistort(lsst::afw::geom::Point2D const &p);
    
    Moment distort(lsst::afw::geom::Point2D const &p, Moment const &Iqq); 
    Moment undistort(lsst::afw::geom::Point2D const &p, Moment const &Iqq);

    std::vector<double> getCoeffs()   {return _coeffs;   }
    std::vector<double> getICoeffs()  {return _icoeffs;  }
    std::vector<double> getDCoeffs()  {return _dcoeffs;  }
    std::vector<double> getIdCoeffs() {return _idcoeffs; }
    
private:
    int _maxN;
    std::vector<double> _coeffs;
    std::vector<double> _icoeffs;
    std::vector<double> _dcoeffs;
    std::vector<double> _idcoeffs;
    std::vector<double> _invert(std::vector<double> const &coeffs);
    std::vector<double> _deriv(std::vector<double> const &coeffs);
    double _transformR(double r, std::vector<double> const &coeffs);
    lsst::afw::geom::Point2D _transform(lsst::afw::geom::Point2D const &p, std::vector<double> const &coeffs);
    Moment _transform(lsst::afw::geom::Point2D const &p, cameraGeom::Moment const &iqq,
                      std::vector<double> const &coeffs);
};


}}}
    
#endif
