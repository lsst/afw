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
 
#if !defined(LSST_AFW_CAMERAGEOM_CCD_H)
#define LSST_AFW_CAMERAGEOM_CCD_H

#include <string>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/afw/cameraGeom/Detector.h"
#include "lsst/afw/cameraGeom/Amp.h"
#include "lsst/afw/cameraGeom/FpPoint.h"

/**
 * @file
 *
 * Describe the physical layout of pixels in the focal plane
 */
namespace lsst {
namespace afw {
namespace cameraGeom {

/**
 * Describe a CCD, containing a number of Amp%s
 */
class Ccd : public Detector {
public:
    typedef boost::shared_ptr<Ccd> Ptr;
    typedef boost::shared_ptr<const Ccd> ConstPtr;

    typedef std::vector<Amp::Ptr> AmpSet;
#if 0                                   // N.b. don't say "AmpSet::iterator" for swig's sake
    typedef ampSet::iterator iterator;
#else
    typedef std::vector<boost::shared_ptr<Amp> >::iterator iterator;
#endif
    typedef std::vector<Amp::Ptr>::const_iterator const_iterator;

    Ccd(Id id, double pixelSize=0.0) : Detector(id, true, pixelSize) {}
    virtual ~Ccd() {}
    //
    // Provide iterators for all the Ccd's Amps
    //
    iterator begin() { return _amps.begin(); }
    const_iterator begin() const { return _amps.begin(); }
    iterator end() { return _amps.end(); }
    const_iterator end() const { return _amps.end(); }
    //
    // Add an Amp to the Ccd
    //
    // This assumes that the amp bbox has already been set (via
    // setDiskToChipLayout).
    void addAmp(Amp const& amp);
    // One can also set the amp coordinates when setting up the ccd
    void addAmp(lsst::afw::geom::Point2I const pos, Amp const& amp);
    void addAmp(int const iX, int const iY, Amp const& amp) {
        addAmp(lsst::afw::geom::Point2I(iX, iY), amp);
    }

    virtual void setTrimmed(bool isTrimmed);
    //
    // Find an Amp given an Id or pixel position
    //
    Amp::Ptr findAmp(Id const id) const;
    Amp::Ptr findAmp(lsst::afw::geom::Point2I const& pixel) const;
    Amp::Ptr findAmp(lsst::afw::geom::Point2I const& pixel, bool const isTrimmed) const;

    virtual void setOrientation(Orientation const& orientation);
    virtual void shift(int dx, int dy);

    virtual void setDefects(std::vector<PTR(lsst::afw::image::DefectBase)> const& defects);
private:
    AmpSet _amps;                       // the Amps that make up this Ccd
};
    
}}}

#endif
