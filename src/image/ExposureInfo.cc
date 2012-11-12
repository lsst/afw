// -*- LSST-C++ -*- // fixed format comment for emacs
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

#include "lsst/afw/image/ExposureInfo.h"
#include "lsst/afw/image/Calib.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace lsst { namespace afw { namespace image {

// Clone various components; defined here so that we don't have to expose their insides in Exposure.h

PTR(detection::Psf) ExposureInfo::_clonePsf(CONST_PTR(detection::Psf) psf) {
    if (psf)
        return psf->clone();
    return PTR(detection::Psf)();
}

PTR(Calib) ExposureInfo::_cloneCalib(CONST_PTR(Calib) calib) {
    if (calib)
        return PTR(Calib)(new Calib(*calib));
    return PTR(Calib)();
}

PTR(Wcs) ExposureInfo::_cloneWcs(CONST_PTR(Wcs) wcs) {
    if (wcs)
        return wcs->clone();
    return PTR(Wcs)();
}

ExposureInfo::ExposureInfo(
    CONST_PTR(Wcs) const & wcs,
    CONST_PTR(detection::Psf) const & psf,
    CONST_PTR(Calib) const & calib,
    CONST_PTR(cameraGeom::Detector) const & detector,
    Filter const & filter,
    PTR(daf::base::PropertySet) const & metadata
) : _wcs(_cloneWcs(wcs)),
    _psf(_clonePsf(psf)),
    _calib(calib ? _cloneCalib(calib) : PTR(Calib)(new Calib())),
    _detector(detector),
    _filter(filter),
    _metadata(metadata ? metadata : PTR(daf::base::PropertySet)(new daf::base::PropertyList()))
{}

ExposureInfo::ExposureInfo(ExposureInfo const & other) : 
    _wcs(_cloneWcs(other._wcs)),
    _psf(_clonePsf(other._psf)),
    _calib(_cloneCalib(other._calib)),
    _detector(other._detector),
    _filter(other._filter),
    _metadata(other._metadata)
{}

ExposureInfo::ExposureInfo(ExposureInfo const & other, bool copyMetadata) :
    _wcs(_cloneWcs(other._wcs)),
    _psf(_clonePsf(other._psf)),
    _calib(_cloneCalib(other._calib)),
    _detector(other._detector),
    _filter(other._filter),
    _metadata(other._metadata)
{
    if (copyMetadata) _metadata = _metadata->deepCopy();
}

ExposureInfo & ExposureInfo::operator=(ExposureInfo const & other) {
    if (&other != this) {
        _wcs = _cloneWcs(other._wcs);
        _psf = _clonePsf(other._psf);
        _calib = _cloneCalib(other._calib);
        _detector = other._detector;
        _filter = other._filter;
        _metadata = other._metadata;
    }
    return *this;
}

ExposureInfo::~ExposureInfo() {}

}}} // namespace lsst::afw::image
