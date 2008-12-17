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
{
    setNotNull(_id);
    setNotNull(_filterId);
    setNotNull(_procHistoryId);
    setNotNull(_ra);
    setNotNull(_dec);
    setNotNull(_raErr4wcs);
    setNotNull(_decErr4wcs);
    setNotNull(_taiMidPoint);   
    setNotNull(_fwhmA);
    setNotNull(_fwhmB);
    setNotNull(_fwhmTheta);
    setNotNull(_psfMag);
    setNotNull(_psfMagErr);
    setNotNull(_instMag);
    setNotNull(_instMagErr);    
    setNotNull(_apMag);
    setNotNull(_apMagErr);
    setNotNull(_modelMag);
    setNotNull(_modelMagErr);
    setNotNull(_snr);
    setNotNull(_chi2);
}

det::Source::~Source()
{
    setAllNull();
}

bool det::Source::isNull    (NullableField const f) const {
    switch(f) {
        case AMP_EXPOSURE_ID:
            return _ampExposureId == 0;
        case OBJECT_ID:
            return _objectId == 0;
        case MOVING_OBJECT_ID:
            return _movingObjectId == 0;
        case RA_ERR_4_DETECTION:
            return _raErr4detection == 0;
        case DEC_ERR_4_DETECTION:
            return _decErr4detection == 0;
        case X_FLUX:
            return _xFlux == 0;
        case X_FLUX_ERR:
            return _xFluxErr == 0;
        case Y_FLUX:
            return _yFlux == 0;
        case Y_FLUX_ERR:
            return _yFluxErr == 0;
        case RA_FLUX:
            return _raFlux == 0;
        case RA_FLUX_ERR:
            return _raFluxErr == 0;
        case DEC_FLUX:
            return _decFlux == 0;
        case DEC_FLUX_ERR:
            return _decFluxErr == 0;
        case X_PEAK:
            return _xPeak == 0;
        case Y_PEAK:
            return _yPeak == 0;
        case RA_PEAK:
            return _raPeak == 0;
        case DEC_PEAK:
            return _decPeak == 0;
        case X_ASTROM:
            return _xAstrom == 0;
        case X_ASTROM_ERR:
            return _xAstromErr == 0;
        case Y_ASTROM:
            return _yAstrom == 0;
        case Y_ASTROM_ERR:
            return _yAstromErr == 0;
        case RA_ASTROM:
            return _raAstrom == 0;
        case RA_ASTROM_ERR:
            return _raAstromErr == 0;
        case DEC_ASTROM:
            return _decAstrom == 0;
        case DEC_ASTROM_ERR:
            return _decAstromErr == 0;
        case TAI_RANGE:
            return _taiRange == 0;
        case PETRO_MAG:
            return _petroMag == 0;
        case PETRO_MAG_ERR:
            return _petroMagErr == 0;
        case NON_GRAY_CORR_MAG:
            return _nonGrayCorrMag == 0;
        case NON_GRAY_CORR_MAG_ERR:
            return _nonGrayCorrMagErr == 0;
        case ATM_CORR_MAG:
            return _atmCorrMag == 0;
        case ATM_CORR_MAG_ERR:
            return _atmCorrMagErr == 0;                                 
        case AP_DIA:
            return _apDia == 0;
        case SKY:
            return _sky == 0;
        case SKY_ERR:        
            return _skyErr == 0;
        case FLAG_4_ASSOCIATION:
            return _flag4association == 0;
        case FLAG_4_DETECTION:
            return _flag4detection == 0;
        case FLAG_4_WCS: 
            return _flag4wcs == 0;
        default:
            return false;
    }
}
void det::Source::setNull   (NullableField const f) {setNull(f, true);}
void det::Source::setNotNull(NullableField const f) {setNull(f, false);}
void det::Source::setNull   (NullableField const f, bool const null) {
    if(null)
        switch(f) {
            case AMP_EXPOSURE_ID:
                setNotNull(_ampExposureId);
                break;
            case OBJECT_ID:
                setNotNull(_objectId);
                break;
            case MOVING_OBJECT_ID:
                setNotNull(_movingObjectId);
                break;
            case RA_ERR_4_DETECTION:
                setNotNull(_raErr4detection);
                break;
            case DEC_ERR_4_DETECTION:
                setNotNull(_decErr4detection);
                break;
            case X_FLUX:
                setNotNull(_xFlux);
                break;
            case X_FLUX_ERR:
                setNotNull(_xFluxErr);
                break;
            case Y_FLUX:
                setNotNull(_yFlux);
                break;
            case Y_FLUX_ERR:
                setNotNull(_yFluxErr);
                break;
            case RA_FLUX:
                setNotNull(_raFlux);
                break;
            case RA_FLUX_ERR:
                setNotNull(_raFluxErr);
                break;
            case DEC_FLUX:
                setNotNull(_decFlux);
                break;
            case DEC_FLUX_ERR:
                setNotNull(_decFluxErr);
                break;
            case X_PEAK:
                setNotNull(_xPeak);
                break;
            case Y_PEAK:
                setNotNull(_yPeak);
                break;
            case RA_PEAK:
                setNotNull(_raPeak);
                break;
            case DEC_PEAK:
                setNotNull(_decPeak);
                break;
            case X_ASTROM:
                setNotNull(_xAstrom);
                break;
            case X_ASTROM_ERR:
                setNotNull(_xAstromErr);
                break;
            case Y_ASTROM:
                setNotNull(_yAstrom);
                break;
            case Y_ASTROM_ERR:
                setNotNull(_yAstromErr);
                break;
            case RA_ASTROM:
                setNotNull(_raAstrom);
                break;
            case RA_ASTROM_ERR:
                setNotNull(_raAstromErr);
                break;
            case DEC_ASTROM:
                setNotNull(_decAstrom);
                break;
            case DEC_ASTROM_ERR:
                setNotNull(_decAstromErr);
                break;
            case TAI_RANGE:
                setNotNull(_taiRange);
                break;
            case PETRO_MAG:
                setNotNull(_petroMag);
                break;
            case PETRO_MAG_ERR:
                setNotNull(_petroMagErr);
                break;
            case NON_GRAY_CORR_MAG:
                setNotNull( _nonGrayCorrMag);
                break;
            case NON_GRAY_CORR_MAG_ERR:
                setNotNull( _nonGrayCorrMagErr);
                break;
            case ATM_CORR_MAG:
                setNotNull( _atmCorrMag);
                break;
            case ATM_CORR_MAG_ERR:
                setNotNull( _atmCorrMagErr);                          
                break;
            case AP_DIA:            
                setNotNull(_apDia);
                break;
            case SKY:
                setNotNull(_sky);
                break;
            case SKY_ERR:        
                setNotNull(_skyErr);
                break;
            case FLAG_4_ASSOCIATION:
                setNotNull(_flag4association);
                break;
            case FLAG_4_DETECTION:
                setNotNull(_flag4detection);
                break;
            case FLAG_4_WCS: 
                setNotNull(_flag4wcs);
                break;
            default:
                return false;
        }
    } else {
        switch(f) {
            case AMP_EXPOSURE_ID:
                setNull(_ampExposureId);
                break;
            case OBJECT_ID:
                setNull(_objectId);
                break;
            case MOVING_OBJECT_ID:
                setNull(_movingObjectId);
                break;
            case RA_ERR_4_DETECTION:
                setNull(_raErr4detection);
                break;
            case DEC_ERR_4_DETECTION:
                setNull(_decErr4detection);
                break;
            case X_FLUX:
                setNull(_xFlux);
                break;
            case X_FLUX_ERR:
                setNull(_xFluxErr);
                break;
            case Y_FLUX:
                setNull(_yFlux);
                break;
            case Y_FLUX_ERR:
                setNull(_yFluxErr);
                break;
            case RA_FLUX:
                setNull(_raFlux);
                break;
            case RA_FLUX_ERR:
                setNull(_raFluxErr);
                break;
            case DEC_FLUX:
                setNull(_decFlux);
                break;
            case DEC_FLUX_ERR:
                setNull(_decFluxErr);
                break;
            case X_PEAK:
                setNull(_xPeak);
                break;
            case Y_PEAK:
                setNull(_yPeak);
                break;
            case RA_PEAK:
                setNull(_raPeak);
                break;
            case DEC_PEAK:
                setNull(_decPeak);
                break;
            case X_ASTROM:
                setNull(_xAstrom);
                break;
            case X_ASTROM_ERR:
                setNull(_xAstromErr);
                break;
            case Y_ASTROM:
                setNull(_yAstrom);
                break;
            case Y_ASTROM_ERR:
                setNull(_yAstromErr);
                break;
            case RA_ASTROM:
                setNull(_raAstrom);
                break;
            case RA_ASTROM_ERR:
                setNull(_raAstromErr);
                break;
            case DEC_ASTROM:
                setNull(_decAstrom);
                break;
            case DEC_ASTROM_ERR:
                setNull(_decAstromErr);
                break;
            case TAI_RANGE:
                setNull(_taiRange);
                break;
            case PETRO_MAG:
                setNull(_petroMag);
                break;
            case PETRO_MAG_ERR:
                setNull(_petroMagErr);
                break;
            case NON_GRAY_CORR_MAG:
                setNull( _nonGrayCorrMag);
                break;
            case NON_GRAY_CORR_MAG_ERR:
                setNull( _nonGrayCorrMagErr);
                break;
            case ATM_CORR_MAG:
                setNull( _atmCorrMag);
                break;
            case ATM_CORR_MAG_ERR:
                setNull( _atmCorrMagErr);                          
                break;                
            case AP_DIA:
                setNull(_apDia);
                break;
            case SKY:
                setNull(_sky);
                break;
            case SKY_ERR:        
                setNull(_skyErr);
                break;
            case FLAG_4_ASSOCIATION:
                setNull(_flag4association);
                break;
            case FLAG_4_DETECTION:
                setNull(_flag4detection);
                break;
            case FLAG_4_WCS: 
                setNull(_flag4wcs);
                break;
            default:
                return;            
        }
    }
}


void det::Source::setNull   () {
    setNull(_ampExposureId);
    setNull(_objectId);
    setNull(_movingObjectId);
    setNull(_raErr4detection);
    setNull(_decErr4detection);
    setNull(_xFlux);
    setNull(_xFluxErr);   
    setNull(_yFlux);
    setNull(_yFluxErr);   
    setNull(_raFlux);
    setNull(_raFluxErr);   
    setNull(_decFlux);
    setNull(_decFluxErr);   
    setNull(_xPeak);
    setNull(_yPeak);
    setNull(_raPeak);
    setNull(_decPeak);  
    setNull(_xAstrom);
    setNull(_xAstromErr);   
    setNull(_yAstrom);
    setNull(_yAstromErr);   
    setNull(_raAstrom);
    setNull(_raAstromErr);   
    setNull(_decAstrom);
    setNull(_decAstromErr);        
    setNull(_taiRange);
    setNull(_petroMag);
    setNull(_petroMagErr);
    setNull(_nonGrayCorrMag);
    setNull(_nonGrayCorrMagErr);
    setNull(_atmCorrMag);
    setNull(_atmCorrMagErr);
    setNull(_apDia);
    setNull(_sky);
    setNull(_skyErr);
    setNull(_flag4association);
    setNull(_flag4detection);
    setNull(_flag4wcs);
}
void det::Source::setNotNull() {
    setNotNull(_ampExposureId);
    setNotNull(_objectId);
    setNotNull(_movingObjectId);
    setNotNull(_raErr4detection);
    setNotNull(_decErr4detection);
    setNotNull(_xFlux);
    setNotNull(_xFluxErr);   
    setNotNull(_yFlux);
    setNotNull(_yFluxErr);   
    setNotNull(_raFlux);
    setNotNull(_raFluxErr);   
    setNotNull(_decFlux);
    setNotNull(_decFluxErr);   
    setNotNull(_xPeak);
    setNotNull(_yPeak);
    setNotNull(_raPeak);
    setNotNull(_decPeak);  
    setNotNull(_xAstrom);
    setNotNull(_xAstromErr);   
    setNotNull(_yAstrom);
    setNotNull(_yAstromErr);   
    setNotNull(_raAstrom);
    setNotNull(_raAstromErr);   
    setNotNull(_decAstrom);
    setNotNull(_decAstromErr);        
    setNotNull(_taiRange);
    setNotNull(_petroMag);
    setNotNull(_petroMagErr);
    setNotNull(_nonGrayCorrMag);
    setNotNull(_nonGrayCorrMagErr);
    setNotNull(_atmCorrMag);
    setNotNull(_atmCorrMagErr);
    setNotNull(_apDia);
    setNotNull(_sky);
    setNotNull(_skyErr);
    setNotNull(_flag4association);
    setNotNull(_flag4detection);
    setNotNull(_flag4wcs);
}

void det::Source::setAllNull() {
    BaseSourceAttributes::setAllNull();
    setNull(_petroMag);
    setNull(_petroMagErr);
    setNull(_sky);
    setNull(_skyErr);
}

void det::Source::setAllNotNull() {
    BaseSourceAttributes::setAllNotNull();
    setNotNull(_petroMag);
    setNotNull(_petroMagErr);
    setNotNull(_sky);
    setNotNull(_skyErr);
}

bool det::Source::operator==(Source const & d) const {
    if (this == &d)  {
        return true;
    }
    
    return (BaseSourceAttributes::operator==(*static_cast<BaseSourceAttributes const*>(&d)) &&
            areEqual(_petroMag, d._petroMag)        &&                   
            areEqual(_petroMagErr, d._petroMagErr)  &&
            areEqual(_sky, d._sky)                  &&                   
            areEqual(_skyErr, d._skyErr));
    
    return false;
}
