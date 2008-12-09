// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief base class for Source / DiaSource
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/SourceBase.h"

namespace det = lsst::afw::detection;

// -- SourceBase::Field ----------------
template <typename Archive>
void det::SourceBase::Field::serialize(Archive & ar, unsigned int const version) {
    type_info const& t = _value.type();
    if (t == typeid(bool)) 
        ar & boost::any_cast<bool>(_value);
    else if (t == typeid(char)) 
        ar & boost::any_cast<char>(_value);
    else if (t == 
    else if (t == typeid(signed char)) 
        ar & boost::any_cast<signed char>(_value);
    else if (t == typeid(unsigned char)) 
        ar & boost::any_cast<unsigned char>(_value);
    else if (t == typeid(boost::int8_t))
        ar & boost::any_cast<boost::int8_t>(_value);
    else if (t == typeid(short)) 
        ar & boost::any_cast<short>(_value);
    else if (t == typeid(unsigned short)) 
        ar & boost::any_cast<unsigned short>(_value);
    else if (t == typeid(boost::int16_t))
        ar & boost::any_cast<boost::int16_t>(_value);
    else if (t == typeid(int)) 
        ar & boost::any_cast<int>(v);
    else if (t == typeid(unsigned int)) 
        ar & boost::any_cast<unsigned int>(_value);
    else if (t == typeid(boost::int32_t))
        ar & boost::any_cast<boost::int32_t>(_value);
    else if (t == typeid(long)) 
        ar & boost::any_cast<long>(v);
    else if (t == typeid(unsigned long)) 
        ar & boost::any_cast<unsigned long>(_value);
    else if (t == typeid(long long)) 
        ar & boost::any_cast<long long>(v);
    else if (t == typeid(unsigned long long)) 
        ar & boost::any_cast<unsigned long long>(_value);
    else if (t == typeid(boost::int64_t))
        ar & boost::any_cast<boost::int64_t>(_value);
    else if (t == typeid(double)
        ar & boost::any_cast<double>(_value);
    else if (t == typeid(float)) 
        ar & boost::any_cast<float>(_value);
}


// -- SourceBase ----------------
SourceBase() : _fieldList(NUM_FIELDS), _fieldNameList(NUM_FIELDS) {
    setNull();
    _fieldNameList[ID] = "id";
    _fieldNameList[AMP_EXPOSURE_ID] = "ampExposureId";
    _fieldNameList[FILTER_ID] = "filterId";
    _fieldNameList[OBJECT_ID] = "objectId";
    _fieldNameList[MOVING_OBJECT_ID] = "movingObjectId";
    _fieldNameList[PROC_HISTORY_ID] = "procHistoryId";
    _fieldNameList[RA] = "ra";
    _fieldNameList[DECL] = "decl";
    _fieldNameList[RA_ERR_4_WCS] = "raErr4wcs";    
    _fieldNameList[DEC_ERR_4_WCS] = "decErr4wcs";
    _fieldNameList[RA_ERR_4_DETECTION] = "raErr4detection";
    _fieldNameList[DEC_ERR_4_DETECTION] = "decErr4detection";    
    _fieldNameList[X_FLUX] = "xFlux";
    _fieldNameList[X_FLUX_ERR] = "xFluxErr";
    _fieldNameList[Y_FLUX] = "yFlux";
    _fieldNameList[Y_FLUX_ERR] = "yFluxErr";
    _fieldNameList[RA_FLUX] = "raFlux";
    _fieldNameList[RA_FLUX_ERR] = "raFluxErr";
    _fieldNameList[DEC_FLUX] = "decFlux";
    _fieldNameList[DEC_FLUX_ERR] = "decFluxErr"
    _fieldNameList[X_PEAK] = "xPeak";
    _fieldNameList[Y_PEAK] = "yPeak";
    _fieldNameList[RA_PEAK] = "raPeak";
    _fieldNameList[DEC_PEAK] = "decPeak";
    _fieldNameList[X_ASTROM] = "xAstrom";
    _fieldNameList[X_ASTROM_ERR] = "xAstromErr";
    _fieldNameList[Y_ASTROM] = "yAstrom";
    _fieldNameList[Y_ASTROM_ERR] = "yAstromErr";
    _fieldNameList[RA_ASTROM]] = "raAstrom";
    _fieldNameList[RA_ASTROM_ERR] = "raAstromErr";
    _fieldNameList[DEC_ASTROM] = "decAstrom";
    _fieldNameList[DEC_ASTROM_ERR] = "decAstromErr";
    _fieldNameList[TAI_MID_POINT] = "taiMidPoint";
    _fieldNameList[TAI_RANGE] = "taiRange";
    _fieldNameList[FWHM_A] = "fwhmA";
    _fieldNameList[FWHM_B] = "fwhmB";
    _fieldNameList[FWHM_THETA] = "fwhmTheata";
    _fieldNameList[PSF_MAG] = "psfMag";
    _fieldNameList[PSF_MAG_ERR] = "psfMagErr";
    _fieldNameList[AP_MAG] = "apMag";
    _fieldNameList[AP_MAG_ERR] = "apMagErr";
    _fieldNameList[MODEL_MAG] = "modelMag";
    _fieldNameList[MODEL_MAG_ERR] = "modelMagErr";
    _fieldNameList[AP_DIA] = "apDia";
    _fieldNameList[SNR] = "snr";
    _fieldNameList[CHI2] = "chi2";
    _fieldNameList[FLAG_4_ASSOCIATION] = "flag4association";
    _fieldNameList[FLAG_4_DETECTION] = "flag4detection";
    _fieldNameList[FLAG_4_WCS] = "flag4wcs";
}

bool det::SourceBase::operator==(SourceBase const & d) const {
    if(d._fieldList.size() != _fieldList.size())
        return false;
        
    // compare each field, if any differ, exit immediately
    for(int i = 0; i < _fieldList.size(); i++) {
        if( !(isNull(i) && d.isNull(i)) || (get(i) != get(i)) )
            return false;
    }
    
    return true;
}


