// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for Source%s
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"

namespace lsst{
namespace afw{
namespace detection{

// -- DiaSource ----------------
DiaSource::DiaSource() : 
    _diaSourceId(0),
    _ampExposureId(0),
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
    _xFlux(0.0),
    _xFluxErr(0.0),
    _yFlux(0.0),
    _yFluxErr(0.0),
    _raFlux(0.0),
    _raFluxErr(0.0),
    _decFlux(0.0),
    _decFluxErr(0.0),
    _xPeak(0.0),
    _yPeak(0.0),
    _raPeak(0.0),
    _decPeak(0.0),
    _xAstrom(0.0),
    _xAstromErr(0.0),
    _yAstrom(0.0),
    _yAstromErr(0.0),
    _raAstrom(0.0),
    _raAstromErr(0.0),
    _decAstrom(0.0),
    _decAstromErr(0.0),
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
    _valX1(0.0),
    _valX2(0.0),
    _valY1(0.0),
    _valY2(0.0),
    _valXY(0.0),                
    _obsCode({0,0,0});
    _isSynthetic(0);
    _mopsStatus(0);
    _flag4association(0),
    _flag4detection(0),
    _flag4wcs(0),
{
    _nulls.set();
}

bool DiaSource::operator==(DiaSource const & d) const {
    if (this == &d)  {
        return true;
    }
    if (_diaSourceId      == d._diaSourceId      &&
        _ampExposureId    == d._ampExposureId    &&
        _filterId         == d._filterId         &&
        _procHistoryId    == d._procHistoryId    &&
        _scId             == d._scId             &&                
        _ra               == d._ra               &&
        _dec              == d._dec              &&
        _raErr4detection  == d._raErr4detection  &&
        _decErr4detection == d._decErr4detection &&
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
        _chi2             == d._chi2             &&
        _valX1            == d._valX1            &&
        _valX2            == d._valX2            &&
        _valY1            == d._valY1            &&
        _valY2            == d._valY2            &&
        _valXY            == d._valXY)     
    {
        if (_nulls == d._nulls) {
            return (isNull(OBJECT_ID)          || _objectId         == d._objectId        ) &&
                   (isNull(MOVING_OBJECT_ID)   || _movingObjectId   == d._movingObjectId  ) &&
                   (isNull(SSM_ID)             || _ssmId            == d._ssmId           ) &&                  
                   (isNull(RA_ERR_4_WCS)       || _raErr4wcs        == d._raErr4wcs       ) &&
                   (isNull(DEC_ERR_4_WCS)      || _decErr4wcs       == d._decErr4wcs      ) &&
                   (isNull(X_FLUX)             || _xFlux            == d._xFlux           ) &&
                   (isNull(X_FLUX_ERR)         || _xFluxErr         == d._xFluxErr        ) &&                   
                   (isNull(Y_FLUX)             || _yFlux            == d._yFlux           ) &&
                   (isNull(Y_FLUX_ERR)         || _yFluxErr         == d._yFluxErr        ) &&
                   (isNull(X_PEAK)             || _xPeak            == d._xPeak           ) && 
                   (isNull(Y_PEAK)             || _yPeak            == d._yPeak           ) && 
                   (isNull(RA_PEAK)            || _raPeak           == d._raPeak          ) && 
                   (isNull(DEC_PEAK)           || _decPeak          == d._decPeak         ) &&   
                   (isNull(X_ASTROM)           || _xAstrom          == d._xAstrom         ) &&
                   (isNull(X_ASTROM_ERR)       || _xAstromErr       == d._xAstromErr      ) &&                   
                   (isNull(Y_ASTROM)           || _yAstrom          == d._yAstrom         ) &&
                   (isNull(Y_ASTROM_ERR)       || _yAstromErr       == d._yAstromErr      ) &&                                   
                   (isNull(RA_ASTROM)          || _raAstrom         == d._raAstrom        ) &&
                   (isNull(RA_ASTROM_ERR)      || _raAstromErr      == d._raAstromErr     ) &&                   
                   (isNull(DEC_ASTROM)         || _decAstrom        == d._decAstrom       ) &&
                   (isNull(DEC_ASTROM_ERR)     || _decAstromErr     == d._decAstromErr    ) &&                                   
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
                   (isNull(MOPS_STATUS)        || _mopsStatus       == d._mopsStatus      ) &&
                   (isNull(FLAG_4_ASSOCIATION) || _flag4association == d._flag4association) &&
                   (isNull(FLAG_4_DETECTION)   || _flag4detection   == d._flag4detection  ) &&
                   (isNull(FLAG_4_WCS)         || _flag4wcs         == d._flag4wcs        );
        }
    }
    return false;
}


// -- DiaSourceVector ----------------
DiaSourceVector::DiaSourceVector()            : daf::base::Citizen(typeid(*this)), _vec()  {}
DiaSourceVector::DiaSourceVector(size_type n) : daf::base::Citizen(typeid(*this)), _vec(n) {}

DiaSourceVector::DiaSourceVector(size_type n, value_type const & val) :
    daf::base::Citizen(typeid(*this)),
    _vec(n, val)
{}


DiaSourceVector::~DiaSourceVector() {}


DiaSourceVector::DiaSourceVector(DiaSourceVector const & v) :
    daf::base::Citizen(typeid(*this)),
    _vec(v._vec)
{}


DiaSourceVector::DiaSourceVector(Vector const & v) :
    daf::base::Citizen(typeid(*this)),
    _vec(v)
{}


DiaSourceVector & DiaSourceVector::operator=(DiaSourceVector const & v) {
    if (this != &v) {
        _vec = v._vec;
    }
    return *this;
}


DiaSourceVector & DiaSourceVector::operator=(Vector const & v) {
    _vec = v;
    return *this;
}

} // namespace detection
} // namespace afw
} // namespace lsst
