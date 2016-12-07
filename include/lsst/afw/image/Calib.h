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

//
/**
 * \file
 *
 * Classes to support calibration (e.g. photometric zero points, exposure times)
 */
#ifndef LSST_AFW_IMAGE_CALIB_H
#define LSST_AFW_IMAGE_CALIB_H

#include <cmath>
#include <utility>
#include <memory>
#include "ndarray_fwd.h"
#include "lsst/base.h"
#include "lsst/daf/base/DateTime.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/table/io/Persistable.h"

namespace lsst {
namespace daf {
    namespace base {
        class PropertySet;
    }
}

namespace afw {
namespace cameraGeom {
    class Detector;
}
namespace image {

static double const JanskysPerABFlux = 3631.0;

/// Compute AB magnitude from flux in Janskys
inline double abMagFromFlux(double flux) {
    return -2.5*std::log10(flux/JanskysPerABFlux);
}

/// Compute AB magnitude error from flux and flux error in Janskys
inline double abMagErrFromFluxErr(double fluxErr, double flux) {
    return std::abs(fluxErr/(-0.4*flux*std::log(10)));
}

/// Compute flux in Janskys from AB magnitude
inline double fluxFromABMag(double mag) {
    return std::pow(10.0, -0.4*mag)*JanskysPerABFlux;
}

/// Compute flux error in Janskys from AB magnitude error and AB magnitude
inline double fluxErrFromABMagErr(double magErr, double mag) {
    return std::abs(-0.4*magErr*fluxFromABMag(mag)*std::log(10.0));
}

/**
 * Describe an exposure's calibration
 */

class Calib : public table::io::PersistableFacade<Calib>, public table::io::Persistable {
public :
    typedef std::shared_ptr<Calib> Ptr;
    typedef std::shared_ptr<Calib const> ConstPtr;

    explicit Calib();
    explicit Calib(double fluxMag0);
    explicit Calib(std::vector<CONST_PTR(Calib)> const& calibs);
    explicit Calib(CONST_PTR(lsst::daf::base::PropertySet));

    void setFluxMag0(double fluxMag0, double fluxMag0Sigma=0.0);
    void setFluxMag0(std::pair<double, double> fluxMag0AndSigma);
    std::pair<double, double> getFluxMag0() const;

    double getFlux(double const mag) const;

    std::pair<double, double> getFlux(double const mag, double const magErr) const;

    ndarray::Array<double,1> getFlux(ndarray::Array<double const,1> const & mag) const;

    std::pair< ndarray::Array<double,1>, ndarray::Array<double,1> > getFlux(
        ndarray::Array<double const,1> const & mag,
        ndarray::Array<double const,1> const & magErr
    ) const;

    double getMagnitude(double const flux) const;

    std::pair<double, double> getMagnitude(double const flux, double const fluxErr) const;

    ndarray::Array<double,1> getMagnitude(ndarray::Array<double const,1> const & flux) const;

    std::pair< ndarray::Array<double,1>,ndarray::Array<double,1> > getMagnitude(
        ndarray::Array<double const,1> const & flux,
        ndarray::Array<double const,1> const & fluxErr
    ) const;

    static void setThrowOnNegativeFlux(bool raiseException);
    static bool getThrowOnNegativeFlux();
    /*
     * Compare two Calibs
     */
    bool operator==(Calib const& rhs) const;
    bool operator!=(Calib const& rhs) const { return !(*this == rhs); }

    Calib & operator*=(double const scale);
    Calib & operator/=(double const scale) { (*this) *= 1.0/scale; return *this; }

    bool isPersistable() const { return true; }

protected:

    virtual std::string getPersistenceName() const;

    virtual void write(OutputArchiveHandle & handle) const;

private:
    double _fluxMag0;
    double _fluxMag0Sigma;
    static bool _throwOnNegativeFlux;
};

namespace detail {
    int stripCalibKeywords(PTR(lsst::daf::base::PropertySet) metadata);
}

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_CALIB_H
