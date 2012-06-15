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
#include "boost/algorithm/string/trim.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst { namespace afw { namespace image {
/**
 * ctor
 */
Calib::Calib() : _midTime(), _exptime(0.0), _fluxMag0(0.0), _fluxMag0Sigma(0.0) {}
/**
 * ctor from a vector of Calibs
 *
 * \note All the input calibs must have the same zeropoint; throw InvalidParameterException if this isn't true
 */
Calib::Calib(std::vector<CONST_PTR(Calib)> const& calibs ///< Set of calibs to be merged
            ) :
    _midTime(), _exptime(0.0), _fluxMag0(0.0), _fluxMag0Sigma(0.0)
{
    if (calibs.empty()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                          "You must provide at least one input Calib");
    }

    double const fluxMag00 = calibs[0]->_fluxMag0;
    double const fluxMag0Sigma0 = calibs[0]->_fluxMag0Sigma;

    double midTimeSum = 0.0;            // sum(time*expTime)
    for (std::vector<CONST_PTR(Calib)>::const_iterator ptr = calibs.begin(); ptr != calibs.end(); ++ptr) {
        Calib const& calib = **ptr;

        if (::fabs(fluxMag00 - calib._fluxMag0) > std::numeric_limits<double>::epsilon() ||
            ::fabs(fluxMag0Sigma0 - calib._fluxMag0Sigma) > std::numeric_limits<double>::epsilon()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterException,
                              (boost::format("You may only combine calibs with the same fluxMag0: "
                                             "%g +- %g v %g +- %g")
                               % calib.getFluxMag0().first % calib.getFluxMag0().second
                               % calibs[0]->getFluxMag0().first % calibs[0]->getFluxMag0().second
                              ).str());
        }

        double const exptime = calib._exptime;

        midTimeSum += calib._midTime.get()*exptime;
        _exptime += exptime;
    }

    daf::base::DateTime tmp(midTimeSum/_exptime); // there's no way to set the double value directly
    using std::swap;
    swap(_midTime, tmp);
}

/**
 * ctor
 */
Calib::Calib(CONST_PTR(lsst::daf::base::PropertySet) metadata) {
    lsst::daf::base::DateTime midTime;
    double exptime = 0.0;
    double fluxMag0 = 0.0, fluxMag0Sigma = 0.0;

    std::string key = "TIME-MID";
    if (metadata->exists(key)) {
        midTime  = lsst::daf::base::DateTime(boost::algorithm::trim_right_copy(metadata->getAsString(key)));
    }

    key = "EXPTIME";
    if (metadata->exists(key)) {
        exptime = metadata->getAsDouble(key);
    }

    key = "FLUXMAG0";
    if (metadata->exists(key)) {
        fluxMag0 = metadata->getAsDouble(key);
        
        key = "FLUXMAG0ERR";
        if (metadata->exists(key)) {
            fluxMag0Sigma = metadata->getAsDouble(key);
        }
    }
        
    _midTime = midTime;
    _exptime = exptime;
    _fluxMag0 = fluxMag0;
    _fluxMag0Sigma = fluxMag0Sigma;
}
/**
 * Control whether we throw an exception when faced with a negative flux
 */
bool Calib::_throwOnNegativeFlux = true;
/**
 * Set whether Calib should throw an exception when asked to convert a flux to a magnitude
 */
void
Calib::setThrowOnNegativeFlux(bool raiseException ///< Should the exception be raised?
                             )
{
    _throwOnNegativeFlux = raiseException;
}

/**
 * Tell me whether Calib will throw an exception if asked to convert a flux to a magnitude
 */
bool
Calib::getThrowOnNegativeFlux()
{
    return _throwOnNegativeFlux;
}

namespace detail {
/**
 * Remove Calib-related keywords from the metadata
 *
 * \return Number of keywords stripped
 */
int stripCalibKeywords(PTR(lsst::daf::base::PropertySet) metadata ///< Metadata to be stripped
                      )
{
    int nstripped = 0;

    std::string key = "TIME-MID";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    key = "EXPTIME";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    key = "FLUXMAG0";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }
        
    key = "FLUXMAG0ERR";
    if (metadata->exists(key)) {
        metadata->remove(key);
        nstripped++;
    }

    return nstripped;
}
}

/**
 * Are two Calibs identical?
 *
 * \note Maybe this should be an approximate comparison
 */
bool Calib::operator==(Calib const& rhs) const {
    return
        ::fabs(_midTime.get() - rhs._midTime.get()) < std::numeric_limits<double>::epsilon() &&
        _exptime == rhs._exptime &&
        _fluxMag0 == rhs._fluxMag0 &&
        _fluxMag0Sigma == rhs._fluxMag0Sigma;
}

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
 *
 * @warning This implementation ignores its arguments!
 */
lsst::daf::base::DateTime Calib::getMidTime(
        boost::shared_ptr<const lsst::afw::cameraGeom::Detector>, ///< description of focal plane (ignored)
        lsst::afw::geom::Point2I const&            ///< position in focal plane (ignored)
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
void Calib::setFluxMag0(double fluxMag0,      ///< The flux in question (ADUs)
                        double fluxMag0Sigma  ///< The error in the flux (ADUs)
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
 * Return a flux (in ADUs) given a magnitude
 */
double Calib::getFlux(double const mag ///< the magnitude of the object
                        ) const {
    
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    
    return _fluxMag0 * ::pow(10.0, -0.4 * mag);
}

/**
 * Return a flux and flux error (in ADUs) given a magnitude and magnitude error
 *
 * Assumes that the errors are small and uncorrelated.
 */
std::pair<double, double> Calib::getFlux(
        double const mag,       ///< the magnitude of the object
        double const magSigma   ///< the error in the magnitude
                
                        ) const {
    
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    
    double const flux = getFlux(mag);
    
//    double const fluxSigma = flux * hypot(_fluxMag0Sigma / _fluxMag0, 0.4 * std::log(10) * magSigma / mag);
    // hypot is not standard C++ so use <http://en.wikipedia.org/wiki/Hypot#Implementation>
    double a = _fluxMag0Sigma / _fluxMag0;
    double b = 0.4 * std::log(10.0) * magSigma;
    if (std::abs(a) < std::abs(b)) {
        double temp = a;
        a = b;
        b = temp;
    }
    double const fluxSigma = flux * std::abs(a) * std::sqrt(1 + std::pow(b / a, 2));
    return std::make_pair(flux, fluxSigma);
}

/**
 * Return a magnitude given a flux
 */
double Calib::getMagnitude(double const flux ///< the measured flux of the object (ADUs)
                         ) const
{
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    if (flux <= 0) {
        if (Calib::getThrowOnNegativeFlux()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                              (boost::format("Flux must be >= 0: saw %g") % flux).str());
        }

        return std::numeric_limits<double>::quiet_NaN();
    }
    
    using ::log10;
    
    return -2.5*log10(flux/_fluxMag0);
}

/**
 * Return a magnitude and magnitude error given a flux and flux error
 */
std::pair<double, double> Calib::getMagnitude(double const flux, ///< the measured flux of the object (ADUs)
                                              double const fluxErr ///< the error in the measured flux (ADUs)
                                              ) const
{
    if (_fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % _fluxMag0).str());
    }
    if (flux <= 0) {
        if (Calib::getThrowOnNegativeFlux()) {
            throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                              (boost::format("Flux must be >= 0: saw %g") % flux).str());
        }

        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return std::make_pair(NaN, NaN);
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
