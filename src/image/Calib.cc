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
#include "boost/lexical_cast.hpp"
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
        try {
            exptime = metadata->getAsDouble(key);
        } catch (daf::base::TypeMismatchException & err) {
            std::string exptimeStr = metadata->getAsString(key);
            exptime = boost::lexical_cast<double>(exptimeStr);
        }
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
void Calib::setFluxMag0(std::pair<double, double> fluxMag0AndSigma ///< The flux and error (ADUs)
                       )
{
    _fluxMag0 = fluxMag0AndSigma.first;
    _fluxMag0Sigma = fluxMag0AndSigma.second;
}

/**
 * Return the flux, and error in flux, of a zero-magnitude object
 */
std::pair<double, double> Calib::getFluxMag0() const
{
    return std::make_pair(_fluxMag0, _fluxMag0Sigma);
}

namespace {
inline void checkNegativeFlux0(double fluxMag0) {
    if (fluxMag0 <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux of 0-mag object must be >= 0: saw %g") % fluxMag0).str());
    }
}
inline bool isNegativeFlux(double flux, bool doThrow)
{
    if (flux <= 0) {
        if (doThrow) {
            throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                              (boost::format("Flux must be >= 0: saw %g") % flux).str());
        }
        return true;
    }
    return false;
}
inline double convertToFlux(double fluxMag0, double mag) {
    return fluxMag0 * ::pow(10.0, -0.4 * mag);
}
inline double convertToFluxErr(double fluxMag0InvSNR, double flux, double magErr) {
    // Want to:
    //     return flux * hypot(_fluxMag0Sigma/_fluxMag0, 0.4*std::log(10)*magSigma/mag);
    // But hypot is not standard C++ so use <http://en.wikipedia.org/wiki/Hypot#Implementation>
    double a = fluxMag0InvSNR;
    double b = 0.4 * std::log(10.0) * magErr;
    if (std::abs(a) < std::abs(b)) {
        std::swap(a, b);
    }
    return flux * std::abs(a) * std::sqrt(1 + std::pow(b / a, 2));
}
inline double convertToMag(double fluxMag0, double flux) {
    return -2.5*::log10(flux/fluxMag0);
}
inline void convertToMagWithErr(double *mag, double *magErr, double fluxMag0, double fluxMag0InvSNR,
                                double flux, double fluxErr)
{
    double const rat = flux/fluxMag0;
    double const ratErr = ::sqrt((::pow(fluxErr, 2) + ::pow(flux*fluxMag0InvSNR, 2))/::pow(fluxMag0, 2));
    
    *mag = -2.5*::log10(rat);
    *magErr = 2.5/::log(10.0)*ratErr/rat;
}

} // anonymous namespace

/**
 * Return a flux (in ADUs) given a magnitude
 */
double Calib::getFlux(double const mag ///< the magnitude of the object
                        ) const {
    checkNegativeFlux0(_fluxMag0);
    return convertToFlux(_fluxMag0, mag);
}
std::vector<double> Calib::getFlux(std::vector<double> const& mag) const {
    checkNegativeFlux0(_fluxMag0);
    std::vector<double> flux(mag.size());
    std::vector<double>::const_iterator inIter = mag.begin();
    std::vector<double>::iterator outIter = flux.begin();
    for (; inIter != mag.end(); ++inIter, ++outIter) {
        *outIter = convertToFlux(_fluxMag0, *inIter);
    }
    return flux;
}

/**
 * Return a flux and flux error (in ADUs) given a magnitude and magnitude error
 *
 * Assumes that the errors are small and uncorrelated.
 */
std::pair<double, double> Calib::getFlux(
        double const mag,       ///< the magnitude of the object
        double const magSigma   ///< the error in the magnitude
    ) const
{
    checkNegativeFlux0(_fluxMag0);
    double const flux = convertToFlux(_fluxMag0, mag);
    double const fluxErr = convertToFluxErr(_fluxMag0Sigma/_fluxMag0, flux, magSigma);
    return std::make_pair(flux, fluxErr);
}

std::pair<std::vector<double>, std::vector<double> > Calib::getFlux(std::vector<double> const& mag,
                                                                    std::vector<double> const& magErr
    ) const
{
    checkNegativeFlux0(_fluxMag0);
    if (mag.size() != magErr.size()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Size of mag (%d) and magErr (%d) don't match") % 
                           mag.size() % magErr.size()).str());
    }

    std::vector<double> flux(mag.size()), fluxErr(mag.size());
    std::vector<double>::const_iterator magIter = mag.begin();
    std::vector<double>::const_iterator magErrIter = magErr.begin();
    std::vector<double>::iterator fluxIter = flux.begin();
    std::vector<double>::iterator fluxErrIter = fluxErr.begin();

    double fluxMag0InvSNR = _fluxMag0Sigma/_fluxMag0;
    for (; magIter != mag.end(); ++magIter, ++magErrIter, ++fluxIter, ++fluxErrIter) {
        *fluxIter = convertToFlux(_fluxMag0, *magIter);
        *fluxErrIter = convertToFluxErr(fluxMag0InvSNR, *fluxIter, *magErrIter);
    }

    return std::make_pair(flux, fluxErr);
}

/**
 * Return a magnitude given a flux
 */
double Calib::getMagnitude(double const flux ///< the measured flux of the object (ADUs)
                         ) const
{
    checkNegativeFlux0(_fluxMag0);
    if (isNegativeFlux(flux, Calib::getThrowOnNegativeFlux())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return convertToMag(_fluxMag0, flux);
}

/**
 * Return a magnitude and magnitude error given a flux and flux error
 */
std::pair<double, double> Calib::getMagnitude(double const flux, ///< the measured flux of the object (ADUs)
                                              double const fluxErr ///< the error in the measured flux (ADUs)
                                              ) const
{
    checkNegativeFlux0(_fluxMag0);
    if (isNegativeFlux(flux, Calib::getThrowOnNegativeFlux())) {
        double const NaN = std::numeric_limits<double>::quiet_NaN();
        return std::make_pair(NaN, NaN);
    }

    double mag, magErr;
    convertToMagWithErr(&mag, &magErr, _fluxMag0, _fluxMag0Sigma/_fluxMag0, flux, fluxErr);
    return std::make_pair(mag, magErr);
}

std::vector<double> Calib::getMagnitude(std::vector<double> const& flux) const
{
    checkNegativeFlux0(_fluxMag0);
    std::vector<double> mag(flux.size());
    std::vector<double>::const_iterator fluxIter = flux.begin();
    std::vector<double>::iterator magIter = mag.begin();
    int nonPositive = 0;
    for (; fluxIter != flux.end(); ++fluxIter, ++magIter) {
        if (isNegativeFlux(*fluxIter, false)) {
            ++nonPositive;
            *magIter = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        *magIter = convertToMag(_fluxMag0, *fluxIter);
    }
    if (nonPositive && Calib::getThrowOnNegativeFlux()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux must be >= 0: %d non-positive seen") % nonPositive).str());
    }
    return mag;
}

std::pair<std::vector<double>, std::vector<double> > Calib::getMagnitude(std::vector<double> const& flux,
                                                                         std::vector<double> const& fluxErr
    ) const
{
    checkNegativeFlux0(_fluxMag0);
    if (flux.size() != fluxErr.size()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthErrorException,
                          (boost::format("Size of flux (%d) and fluxErr (%d) don't match") % 
                           flux.size() % fluxErr.size()).str());
    }

    std::vector<double> mag(flux.size()), magErr(flux.size());
    std::vector<double>::const_iterator fluxIter = flux.begin();
    std::vector<double>::const_iterator fluxErrIter = fluxErr.begin();
    std::vector<double>::iterator magIter = mag.begin();
    std::vector<double>::iterator magErrIter = magErr.begin();
    int nonPositive = 0;
    double fluxMag0InvSNR = _fluxMag0Sigma/_fluxMag0;
    for (; fluxIter != flux.end(); ++fluxIter, ++fluxErrIter, ++magIter, ++magErrIter) {
        if (isNegativeFlux(*fluxIter, false)) {
            ++nonPositive;
            double const NaN = std::numeric_limits<double>::quiet_NaN();
            *magIter = NaN;
            *magErrIter = NaN;
            continue;
        }
        double f, df;
        convertToMagWithErr(&f, &df, _fluxMag0, fluxMag0InvSNR, *fluxIter, *fluxErrIter);
        *magIter = f;
        *magErrIter = df;
    }
    if (nonPositive && Calib::getThrowOnNegativeFlux()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::DomainErrorException,
                          (boost::format("Flux must be >= 0: %d non-positive seen") % nonPositive).str());
    }
    return std::make_pair(mag, magErr);
}
   



}}}  // lsst::afw::image
