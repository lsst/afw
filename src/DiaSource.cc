// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file   DiaSource.cc
//!
//##====----------------                                ----------------====##/

#include "lsst/fw/DiaSource.h"


namespace lsst {
namespace fw {

// -- DiaSource ----------------

DiaSource::DiaSource() : 
    _diaSourceId(0),
    _ccdExposureId(0),
    _objectId(0),
    _movingObjectId(0),
    _colc(0.0),
    _rowc(0.0),
    _dcol(0.0),
    _drow(0.0),
    _ra(0.0),
    _decl(0.0),
    _raErr4detection(0.0),
    _decErr4detection(0.0),
    _raErr4wcs(0.0),
    _decErr4wcs(0.0),
    _cx(0.0),
    _cy(0.0),
    _cz(0.0),
    _taiMidPoint(0.0),
    _taiRange(0.0),
    _flux(0.0),
    _fluxErr(0.0),
    _psfMag(0.0),
    _psfMagErr(0.0),
    _apMag(0.0),
    _apMagErr(0.0),
    _modelMag(0.0),
    _modelMagErr(0.0),
    _colcErr(0.0),
    _rowcErr(0.0),
    _fwhmA(0.0),
    _fwhmB(0.0),
    _fwhmTheta(0.0),
    _apDia(0.0),
    _ixx(0.0),
    _ixxErr(0.0),
    _iyy(0.0),
    _iyyErr(0.0),
    _ixy(0.0),
    _ixyErr(0.0),
    _snr(0.0),
    _chi2(0.0),
    _scId(0),
    _flag4association(0),
    _flag4detection(0),
    _flag4wcs(0),
    _filterId(0),
    _dataSource(0)
{
    _nulls.set();
}


DiaSource::DiaSource(
    int64_t id,
    double  colc,
    double  rowc,
    double  dcol,
    double  drow
) : 
    _diaSourceId(id),
    _ccdExposureId(0),
    _objectId(0),
    _movingObjectId(0),
    _colc(colc),
    _rowc(rowc),
    _dcol(dcol),
    _drow(drow),
    _ra(0.0),
    _decl(0.0),
    _raErr4detection(0.0),
    _decErr4detection(0.0),
    _raErr4wcs(0.0),
    _decErr4wcs(0.0),
    _cx(0.0),
    _cy(0.0),
    _cz(0.0),
    _taiMidPoint(0.0),
    _taiRange(0.0),
    _flux(0.0),
    _fluxErr(0.0),
    _psfMag(0.0),
    _psfMagErr(0.0),
    _apMag(0.0),
    _apMagErr(0.0),
    _modelMag(0.0),
    _modelMagErr(0.0),
    _colcErr(0.0),
    _rowcErr(0.0),
    _fwhmA(0.0),
    _fwhmB(0.0),
    _fwhmTheta(0.0),
    _apDia(0.0),
    _ixx(0.0),
    _ixxErr(0.0),
    _iyy(0.0),
    _iyyErr(0.0),
    _ixy(0.0),
    _ixyErr(0.0),
    _snr(0.0),
    _chi2(0.0),
    _scId(0),
    _flag4association(0),
    _flag4detection(0),
    _flag4wcs(0),
    _filterId(0),
    _dataSource(0)
{
    _nulls.set();
}


bool DiaSource::operator==(DiaSource const & d) const {
    if (this == &d)  {
        return true;
    }
    if (_diaSourceId      == d._diaSourceId      &&
        _ccdExposureId    == d._ccdExposureId    &&
        _colc             == d._colc             &&
        _rowc             == d._rowc             &&
        _dcol             == d._dcol             &&
        _drow             == d._drow             &&
        _ra               == d._ra               &&
        _decl             == d._decl             &&
        _raErr4detection  == d._raErr4detection  &&
        _decErr4detection == d._decErr4detection &&
        _cx               == d._cx               &&
        _cy               == d._cy               &&
        _cz               == d._cz               &&
        _taiMidPoint      == d._taiMidPoint      &&
        _taiRange         == d._taiRange         &&
        _flux             == d._flux             &&
        _fluxErr          == d._fluxErr          &&
        _psfMag           == d._psfMag           &&
        _psfMagErr        == d._psfMagErr        &&
        _apMag            == d._apMag            &&
        _apMagErr         == d._apMagErr         &&
        _modelMag         == d._modelMag         &&
        _modelMagErr      == d._modelMagErr      &&
        _colcErr          == d._colcErr          &&
        _rowcErr          == d._rowcErr          &&
        _fwhmA            == d._fwhmA            &&
        _fwhmB            == d._fwhmB            &&
        _fwhmTheta        == d._fwhmTheta        &&
        _snr              == d._snr              &&
        _chi2             == d._chi2             &&
        _scId             == d._scId             &&
        _filterId         == d._filterId         &&
        _dataSource       == d._dataSource)
    {
        if (_nulls == d._nulls) {
            return (isNull(OBJECT_ID)          || _objectId         == d._objectId        ) &&
                   (isNull(MOVING_OBJECT_ID)   || _movingObjectId   == d._movingObjectId  ) &&
                   (isNull(RA_ERR_4_WCS)       || _raErr4wcs        == d._raErr4wcs       ) &&
                   (isNull(DEC_ERR_4_WCS)      || _decErr4wcs       == d._decErr4wcs      ) &&
                   (isNull(AP_DIA)             || _apDia            == d._apDia           ) &&
                   (isNull(IXX)                || _ixx              == d._ixx             ) &&
                   (isNull(IXX_ERR)            || _ixxErr           == d._ixxErr          ) &&
                   (isNull(IYY)                || _iyy              == d._iyy             ) &&
                   (isNull(IYY_ERR)            || _iyyErr           == d._iyyErr          ) &&
                   (isNull(IXY)                || _ixy              == d._ixy             ) &&
                   (isNull(IXY_ERR)            || _ixyErr           == d._ixyErr          ) &&
                   (isNull(FLAG_4_ASSOCIATION) || _flag4association == d._flag4association) &&
                   (isNull(FLAG_4_DETECTION)   || _flag4detection   == d._flag4detection  ) &&
                   (isNull(FLAG_4_WCS)         || _flag4wcs         == d._flag4wcs        );
        }
    }
    return false;
}


// -- DiaSourceVector ----------------

DiaSourceVector::DiaSourceVector()            : lsst::mwi::data::Citizen(typeid(*this)), _vec()  {}
DiaSourceVector::DiaSourceVector(size_type n) : lsst::mwi::data::Citizen(typeid(*this)), _vec(n) {}

DiaSourceVector::DiaSourceVector(size_type n, value_type const & val) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(n, val)
{}


DiaSourceVector::~DiaSourceVector() {}


DiaSourceVector::DiaSourceVector(DiaSourceVector const & v) :
    lsst::mwi::data::Citizen(typeid(*this)),
    _vec(v._vec)
{}


DiaSourceVector::DiaSourceVector(Vector const & v) :
    lsst::mwi::data::Citizen(typeid(*this)),
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


}} // end of namespace lsst::fw

