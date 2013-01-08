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
 
#if !defined(LSST_AFW_DETECTION_PEAK_H)
#define LSST_AFW_DETECTION_PEAK_H
/*!
 * \file
 * \brief Support for peaks in images
 */

#include <algorithm>
#include <list>
#include <cmath>
#include <limits>
#include <boost/shared_ptr.hpp>
#include "lsst/daf/base/Citizen.h"
#include "lsst/afw/geom/Point.h"

#include <boost/serialization/nvp.hpp>

namespace boost {
namespace serialization {
    class access;
}}

#ifndef SWIG
using boost::serialization::make_nvp;
#endif

namespace lsst { namespace afw { namespace detection {
/// A peak in an %image
class Peak : public lsst::daf::base::Citizen {
public:
    typedef boost::shared_ptr<Peak> Ptr;
    /// A peak at the pixel <tt>(ix, iy)</tt>
    explicit Peak(int ix,               //!< column pixel
                  int iy,               //!< row pixel
                  float peakValue=std::numeric_limits<float>::quiet_NaN() //!< value of image at peak
                 ) : lsst::daf::base::Citizen(typeid(this)),
          _id(++id),
          _ix(ix), _iy(iy), _fx(ix), _fy(iy), _peakValue(peakValue) {};
    /// A peak at the floating-point position <tt>(fx, fy)</tt>
    explicit Peak(float fx=std::numeric_limits<float>::quiet_NaN(),       //!< column centre
                  float fy=std::numeric_limits<float>::quiet_NaN(),       //!< row centre
                  float peakValue=std::numeric_limits<float>::quiet_NaN() //!< value of image at peak
                 ) : lsst::daf::base::Citizen(typeid(this)),
          _id(++id),
          _ix(fx > 0 ? static_cast<int>(fx) : -static_cast<int>(-fx) - 1),
          _iy(fy > 0 ? static_cast<int>(fy) : -static_cast<int>(-fy) - 1),
          _fx(fx), _fy(fy), _peakValue(peakValue) {};
    Peak(Peak const & other) 
      : lsst::daf::base::Citizen(typeid(this)),
        _id(++id),
        _ix(other._ix), _iy(other._iy),
        _fx(other._fx), _fy(other._fy), _peakValue(other._peakValue) 
    { }
    Peak& operator=(Peak const & other)
    {
        if (this != &other) {
            _id = ++id;
            _ix = other._ix; _iy = other._iy;
            _fx = other._fx; _fy = other._fy;
            _peakValue = other._peakValue;
        }
        return *this;
    }

    ~Peak() {};
    
    bool operator==(Peak const& rhs) const {
        return _id == rhs._id;
    }
    bool operator!=(Peak const& rhs) const {
        return !(*this == rhs);
    }

    int getId() const { return _id; }   //!< Return the Peak's unique ID

    int getIx() const { return _ix; }         //!< Return the column pixel position
    int getIy() const { return _iy; }         //!< Return the row pixel position
    /// Return the coordinates of the highest pixel
    lsst::afw::geom::Point2I getI() const { return getCentroid(true); }
    /// Return the coordinates of the highest pixel
    lsst::afw::geom::Point2I getCentroid(bool) const { return lsst::afw::geom::Point2I(_ix, _iy); }
    float getFx() const { return _fx; }       //!< Return the column centroid
    float getFy() const { return _fy; }       //!< Return the row centroid
    /// Return the peak's centroid
    lsst::afw::geom::Point2D getF() const { return getCentroid(); }
    /// Return the peak's centroid
    lsst::afw::geom::Point2D getCentroid() const { return lsst::afw::geom::Point2D(_fx, _fy); }
    void setFx(float fx) { _fx = fx; }        //!< Set the column centroid
    void setFy(float fy) { _fy = fy; }        //!< Set the row centroid

    void setPeakValue(float peakValue) { _peakValue = peakValue; } //!< Set the value of the image at the peak
    float getPeakValue() const { return _peakValue; } //!< Return the value of the image at the peak

    std::string toString();    
private:
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive & ar, unsigned int const version) {
        ar & make_nvp("id", _id);
        ar & make_nvp("ix", _ix) & make_nvp("iy", _iy);
        ar & make_nvp("fx", _fx) & make_nvp("fy", _fy);
        ar & make_nvp("peakValue", _peakValue);
    }
    //Peak(const Peak &) {}             // XXX How do we manage Citizen's copy constructor?
    static int id;
    mutable int _id;                    //!< unique ID for this peak
    int _ix;                            //!< column-position of peak pixel
    int _iy;                            //!< row-position of peak pixel
    float _fx;                          //!< column-position of peak
    float _fy;                          //!< row-position of peak
    float _peakValue;                   //!< value of image at peak position
};

}}}

#endif
