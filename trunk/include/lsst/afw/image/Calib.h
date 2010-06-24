// -*- lsst-c++ -*-
//
/**
 * \file
 *
 * Classes to support calibration (e.g. photometric zero points, exposure times)
 */
#ifndef LSST_AFW_IMAGE_CALIB_H
#define LSST_AFW_IMAGE_CALIB_H

#include <utility>
#include "boost/shared_ptr.hpp"
#include "lsst/daf/base/DateTime.h"
#include "lsst/afw/geom/Point.h"

namespace lsst {
namespace afw {
namespace cameraGeom {
    class Detector;
}
namespace image {
/**
 * Describe an exposure's calibration
 */

class Calib
{
public :
    typedef boost::shared_ptr<Calib> Ptr;
    typedef boost::shared_ptr<Calib const> ConstPtr;

    explicit Calib();

    void setMidTime(lsst::daf::base::DateTime const& midTime);
    lsst::daf::base::DateTime getMidTime () const;
    lsst::daf::base::DateTime getMidTime(boost::shared_ptr<const lsst::afw::cameraGeom::Detector>,
                                         lsst::afw::geom::Point2I const&) const;

    void setExptime(double exptime);
    double getExptime() const;

    void setFluxMag0(double fluxMag0, double fluxMag0Sigma=0.0);
    std::pair<double, double> getFluxMag0() const;

    double getMagnitude(double const flux);
    std::pair<double, double> getMagnitude(double const flux, double const fluxErr);
private :
    lsst::daf::base::DateTime _midTime;
    double _exptime;
    double _fluxMag0;
    double _fluxMag0Sigma;
};

}}}  // lsst::afw::image

#endif // LSST_AFW_IMAGE_CALIB_H
