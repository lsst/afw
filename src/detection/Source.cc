// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for Sources
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace det = lsst::afw::detection;

det::Source::Source()
	: _petroMag(0.0),
      _petroMagErr(0.0),
      _sky(0.0),
      _skyErr(0.0)
{}

det::Source::Source(Source const & other)
	: BaseSourceAttributes<NUM_SOURCE_NULLABLE_FIELDS>(other),
	  _petroMag(other._petroMag),
      _petroMagErr(other._petroMagErr),
      _sky(other._sky),
      _skyErr(other._skyErr)
{
	for(int i =0; i < NUM_SOURCE_NULLABLE_FIELDS; i++)
		setNull(i, other.isNull(i));
}

bool det::Source::operator==(Source const & d) const {
    return ( areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId,  AMP_EXPOSURE_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId,  OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId,  MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErr4wcs, d._raErr4wcs) &&
        areEqual(_decErr4wcs, d._decErr4wcs) &&
        areEqual(_raErr4detection, d._raErr4detection,  RA_ERR_4_DETECTION) &&
        areEqual(_decErr4detection, d._decErr4detection,  DEC_ERR_4_DETECTION) &&
        areEqual(_xFlux, d._xFlux,  X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr,  X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux,  Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr,  Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak,  X_PEAK) && 
        areEqual(_yPeak, d._yPeak,  Y_PEAK) && 
        areEqual(_raPeak, d._raPeak,  RA_PEAK) && 
        areEqual(_decPeak, d._decPeak,  DEC_PEAK) &&   
        areEqual(_xAstrom, d._xAstrom,  X_ASTROM) &&
        areEqual(_xAstromErr, d._xAstromErr,  X_ASTROM_ERR) &&                   
        areEqual(_yAstrom, d._yAstrom,  Y_ASTROM) &&
        areEqual(_yAstromErr, d._yAstromErr,  Y_ASTROM_ERR) &&                                   
        areEqual(_raAstrom, d._raAstrom,  RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr,  RA_ASTROM_ERR) &&                   
        areEqual(_decAstrom, d._decAstrom,  DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr,  DEC_ASTROM_ERR) &&
        areEqual(_taiMidPoint, d._taiMidPoint) &&
        areEqual(_taiRange, d._taiRange,  TAI_RANGE) &&
        areEqual(_fwhmA, d._fwhmA) &&
        areEqual(_fwhmB, d._fwhmB) &&
        areEqual(_fwhmTheta, d._fwhmTheta) &&
        areEqual(_psfMag, d._psfMag) &&
        areEqual(_psfMagErr, d._psfMagErr) &&
        areEqual(_apMag, d._apMag) &&
        areEqual(_apMagErr, d._apMagErr) &&
        areEqual(_modelMag, d._modelMag) &&
        areEqual(_modelMagErr, d._modelMagErr) &&
        areEqual(_petroMag, d._petroMag,  PETRO_MAG) &&
        areEqual(_petroMagErr, d._petroMagErr,  PETRO_MAG_ERR) &&             
        areEqual(_instMag, d._instMag) &&
        areEqual(_instMagErr, d._instMagErr) &&
        areEqual(_nonGrayCorrMag, d._nonGrayCorrMag,  NON_GRAY_CORR_MAG) &&
        areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr,  NON_GRAY_CORR_MAG_ERR) &&
        areEqual(_atmCorrMag, d._atmCorrMag,  ATM_CORR_MAG) &&
        areEqual(_atmCorrMagErr, d._atmCorrMagErr,  ATM_CORR_MAG_ERR) &&
        areEqual(_apDia, d._apDia,  AP_DIA) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_sky, d._sky,  SKY) &&
        areEqual(_skyErr, d._skyErr,  SKY_ERR) &&
        areEqual(_flag4association, d._flag4association,  FLAG_4_ASSOCIATION) &&
        areEqual(_flag4detection, d._flag4detection,  FLAG_4_DETECTION) &&
        areEqual(_flag4wcs, d._flag4wcs,  FLAG_4_WCS));
}
