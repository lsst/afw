// -*- lsst-c++ -*-

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
 *
 * Classes to support calibration (e.g. photometric zero points, exposure times)
 */
#include <cmath>
#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst { namespace afw { namespace image {
/**
 * ctor
 */
Calib::Calib() : _midTime(), _exptime(0.0), _fluxMag0(0.0), _fluxMag0Sigma(0.0) {}

/**
 * Set the time of the middle of an exposure
 *
 * \note In general this is a function of position of the position of the detector in the focal plane
 */
void Calib::setMidTime(lsst::daf::base::DateTime const& midTime ///< Time at middle of exposure
                      )
{
    _midTime = midTime;
}

/**
 * Return the time at the middle of an exposure
 */
lsst::daf::base::DateTime Calib::getMidTime () const
{
    return _midTime;
}

/**
 * Return the time at the middle of an exposure at the specified position in the focal plane (as
 * described by a cameraGeom::Detector)
 */
lsst::daf::base::DateTime Calib::getMidTime(
        lsst::afw::cameraGeom::Detector::ConstPtr, ///< description of focal plane
        lsst::afw::geom::Point2I const&            ///< position in focal plane
                                           ) const
{
    return _midTime;
}

/**
 * Set the length of an exposure
 */
void Calib::setExptime(double exptime      ///< the length of the exposure (s)
                       ) {
    _exptime = exptime;
}

/**
 * Return the length of an exposure in seconds
 */
double Calib::getExptime() const
{
    return _exptime;
}

/**
 * Set the flux of a zero-magnitude object
 */
void Calib::setFluxMag0(double fluxMag0,      ///< The flux in question
                        double fluxMag0Sigma  ///< The error in the flux
                       )
{
    _fluxMag0 = fluxMag0;
    _fluxMag0Sigma = fluxMag0Sigma;
}

/**
 * Return the flux, and error in flux, of a zero-magnitude object
 */
std::pair<double, double> Calib::getFluxMag0() const
{
    return std::make_pair(_fluxMag0, _fluxMag0Sigma);
}
            
/**
 * Return a magnitude given a flux
 */
double Calib::getMagnitude(double const flux ///< the measured flux of the object
                         ) const
{
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    if (flux <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux must be >= 0: saw %g") % flux).str());
    }
    
    using ::log10;
    
    return -2.5*log10(flux/_fluxMag0);
}

/**
 * Return a magnitude and magnitude error given a flux and flux error
 */
std::pair<double, double> Calib::getMagnitude(double const flux, ///< the measured flux of the object
                                            double const fluxErr ///< the error in the measured flux
                                           ) const
{
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    if (flux <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux must be >= 0: saw %g") % flux).str());
    }
    
    using ::pow; using ::sqrt;
    using ::log; using ::log10;
    
    double const rat = flux/_fluxMag0;
    double const ratErr = sqrt((pow(fluxErr, 2) + pow(flux*_fluxMag0Sigma/_fluxMag0, 2))/pow(_fluxMag0, 2));
    
    double const mag = -2.5*log10(rat);
    double const magErr = 2.5/log(10.0)*ratErr/rat;
    return std::make_pair(mag, magErr);
}

}}}  // lsst::afw::image
