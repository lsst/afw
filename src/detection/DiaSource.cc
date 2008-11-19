// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for DiaSource%s
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"

namespace det = lsst::afw::detection;

// -- DiaSource ----------------
det::DiaSource::DiaSource() : 
    _diaSourceId(0),
    _ccdExposureId(0),
    _filterId(0),
    _objectId(0),
    _movingObjectId(0),
    _procHistoryId(0),
    _scId(0);
    _ssmId(0);
    _ra(0.0),
    _decl(0.0),
    _raErr4detection(0.0),
    _decErr4detection(0.0),
    _raErr4wcs(0.0),
    _decErr4wcs(0.0),
    _col(0.0),
    _row(0.0), 
    _colErr(0.0),
    _rowErr(0.0),
    _cx(0.0),
    _cy(0.0),
    _cz(0.0),       
    _taiMidPoint(0.0),
    _taiRange(0.0),    
    _fwhmA(0.0),
    _fwhmB(0.0),
    _fwhmTheta(0.0),
    _lengthDeg(0.0),
    _flux(0.0),
    _fluxErr(0.0),
    _psfMag(0.0),
    _psfMagErr(0.0),
    _apMag(0.0),
    _apMagErr(0.0),
    _modelMag(0.0),
    _modelMagErr(0.0),
    _apDia(0.0),
    _refMag(0.0),
    _ixx(0.0),
    _ixxErr(0.0),
    _iyy(0.0),
    _iyyErr(0.0),
    _ixy(0.0),
    _ixyErr(0.0),
    _snr(0.0),
    _chi2(0.0),             
    _obsCode(0),
    _isSynthetic(0),
    _mopsStatus(0),
    _flag4association(0),
    _flag4detection(0),
    _flag4wcs(0),
{
    _nulls.set();
}

bool det::DiaSource::operator==(DiaSource const & d) const {
    if (this == &d)  {
        return true;
    }
    if (_diaSourceId      == d._diaSourceId      &&
        _ccdExposureId    == d._ccdExposureId    &&
        _filterId         == d._filterId         &&
        _procHistoryId    == d._procHistoryId    &&
        _scId             == d._scId             &&                
        _ra               == d._ra               &&
        _dec              == d._dec              &&
        _raErr4detection  == d._raErr4detection  &&
        _decErr4detection == d._decErr4detection &&
        _row              == d._row              &&
        _col              == d._col              &&
        _rowErr           == d._rowErr           &&
        _colErr           == d._colErr           &&
        _cx               == d._cx               &&
        _cy               == d._cy               &&
        _cz               == d._cz               &&        
        _taiMidPoint      == d._taiMidPoint      &&
        _taiRange         == d._taiRange         &&
        _fwhmA            == d._fwhmA            &&
        _fwhmB            == d._fwhmB            &&
        _fwhmTheta        == d._fwhmTheta        &&
        _lengthDeg        == d._lengthDeg        &&        
        _flux             == d._flux             &&
        _fluxErr          == d._fluxErr          &&
        _psfMag           == d._psfMag           &&
        _psfMagErr        == d._psfMagErr        &&
        _apMag            == d._apMag            &&
        _apMagErr         == d._apMagErr         &&
        _modelMag         == d._modelMag         &&
        _snr              == d._snr              &&
        _chi2             == d._chi2)     
    {
        if (_nulls == d._nulls) {
            return (isNull(OBJECT_ID)          || _objectId         == d._objectId        ) &&
                   (isNull(MOVING_OBJECT_ID)   || _movingObjectId   == d._movingObjectId  ) &&
                   (isNull(SSM_ID)             || _ssmId            == d._ssmId           ) &&                  
                   (isNull(RA_ERR_4_WCS)       || _raErr4wcs        == d._raErr4wcs       ) &&
                   (isNull(DEC_ERR_4_WCS)      || _decErr4wcs       == d._decErr4wcs      ) &&                      
                   (isNull(MODEL_MAG_ERR)      || _modelMagErr      == d._modelMagErr     ) &&
                   (isNull(AP_DIA)             || _apDia            == d._apDia           ) &&                   
                   (isNull(REF_MAG)            || _refMag           == d._refMag          ) &&                   
                   (isNull(IXX)                || _ixx              == d._ixx             ) &&
                   (isNull(IXX_ERR)            || _ixxErr           == d._ixxErr          ) &&
                   (isNull(IYY)                || _iyy              == d._iyy             ) &&
                   (isNull(IYY_ERR)            || _iyyErr           == d._iyyErr          ) &&
                   (isNull(IXY)                || _ixy              == d._ixy             ) &&
                   (isNull(IXY_ERR)            || _ixyErr           == d._ixyErr          ) &&
                   (isNull(OBS_CODE)           || _obsCode          == d._obsCode         ) &&
                   (isNull(IS_SYNTHETIC)       || _isSynthetic      == d._isSynthetic     ) &&
                   (isNull(STATUS)             || _status           == d._status          ) &&
                   (isNull(FLAG_4_ASSOCIATION) || _flag4association == d._flag4association) &&
                   (isNull(FLAG_4_DETECTION)   || _flag4detection   == d._flag4detection  ) &&
                   (isNull(FLAG_4_WCS)         || _flag4wcs         == d._flag4wcs        );
        }
    }
    return false;
}


// -- DiaSourceVector ----------------
det::DiaSourceVector::DiaSourceVector()            
    : lsst::daf::base::Citizen(typeid(*this)), _vec()  {}
det::DiaSourceVector::DiaSourceVector(size_type n) 
    : lsst::daf::base::Citizen(typeid(*this)), _vec(n) {}
det::DiaSourceVector::DiaSourceVector(size_type n, value_type const & val) 
    : lsst::daf::base::Citizen(typeid(*this)), _vec(n, val) {}
det::DiaSourceVector::DiaSourceVector(DiaSourceVector const & v)
    : lsst::daf::base::Citizen(typeid(*this)), _vec(v._vec) {}
det::DiaSourceVector::DiaSourceVector(Vector const & v)
    : daf::base::Citizen(typeid(*this)), _vec(v) {}


det::DiaSourceVector::~DiaSourceVector() {}

det::DiaSourceVector & det::DiaSourceVector::operator=(DiaSourceVector const & v) {
    if (this != &v) {
        _vec = v._vec;
    }
    return *this;
}


det::DiaSourceVector & det::DiaSourceVector::operator=(Vector const & v) {
    _vec = v;
    return *this;
}
