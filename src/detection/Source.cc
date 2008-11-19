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

namespace det = lsst::afw::detection;

det::Source::Source() : 
    _sourceId(0),
    _ampExposureId(0),
    _filterId(0),
    _objectId(0),
    _movingObjectId(0),
    _procHistoryId(0),
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
    _psfMag(0.0),
    _psfMagErr(0.0),
    _apMag(0.0),
    _apMagErr(0.0),
    _modelMag(0.0),
    _modelMagErr(0.0),
    _petroMag(0.0),
    _petroMagErr(0.0),
    _apDia(0.0),        
    _snr(0.0),
    _chi2(0.0),
    _sky(0.0),
    _skyErr(0.0),
    _flag4association(0),
    _flag4detection(0),
    _flag4wcs(0),
{
    _nulls.set();
}


bool det::Source::operator==(Source const & d) const {
    if (this == &d)  {
        return true;
    }
    if (_sourceId         == d._sourceId         &&
        _filterId         == d._filterId         &&
        _procHistoryId    == d._procHistoryId    &&
        _ra               == d._ra               &&
        _dec              == d._dec              &&
        _raErr4Wcs        == d._raErr4Wcs        &&
        _decErr4Wcs       == d._decErr4Wcs       &&
        _row              == d._row              &&
        _col              == d._col              &&
        _rowErr           == d._rowErr           &&
        _colErr           == d._colErr           &&
        _cx               == d._cx               &&
        _cy               == d._cy               &&
        _cz               == d._cz               &&          
        _taiMidPoint      == d._taiMidPoint      &&
        _fwhmA            == d._fwhmA            &&
        _fwhmB            == d._fwhmB            &&
        _fwhmTheta        == d._fwhmTheta        &&
        _psfMag           == d._psfMag           &&
        _psfMagErr        == d._psfMagErr        &&
        _apMag            == d._apMag            &&
        _apMagErr         == d._apMagErr         &&
        _modelMag         == d._modelMag         &&
        _modelMagErr      == d._modelMagErr      &&
        _snr              == d._snr              &&
        _chi2             == d._chi2)
    {
        if (_nulls == d._nulls) {
           
            return (isNull(AMP_EXPOSURE_ID)    || _ampExposureId    == d._ampExposureId   ) &&
                   (isNull(OBJECT_ID)          || _objectId         == d._objectId        ) &&
                   (isNull(MOVING_OBJECT_ID)   || _movingObjectId   == d._movingObjectId  ) &&
                   (isNull(RA_ERR_4_DETECTION) || _raErr4Detection  == d._raErr4Detection ) &&
                   (isNull(DEC_ERR_4_DETECTION)|| _decErr4Detection == d._decErr4Detection) &&                                  
                   (isNull(TAI_RANGE)          || _taiRange         == d._taiRange        ) &&
                   (isNull(PETRO_MAG)          || _petroMag         == d._petroMag        ) &&                   
                   (isNull(PETRO_MAG_ERR)      || _petroMagErr      == d._petroMagErr     ) &&
                   (isNull(AP_DIA)             || _apDia            == d._apDia           ) &&
                   (isNull(SKY)                || _sky              == d._sky             ) &&                   
                   (isNull(SKY_ERR)            || _skyErr           == d._skyErr          ) &&
                   (isNull(FLAG_4_ASSOCIATION) || _flag4association == d._flag4association) &&
                   (isNull(FLAG_4_DETECTION)   || _flag4detection   == d._flag4detection  ) &&
                   (isNull(FLAG_4_WCS)         || _flag4wcs         == d._flag4wcs        );
        }
    }
    return false;
}


// -- SourceVector ----------------
det::SourceVector::SourceVector()            
    : lsst::daf::base::Citizen(typeid(*this)), _vec()  {}
det::SourceVector::SourceVector(size_type n) 
    : lsst::daf::base::Citizen(typeid(*this)), _vec(n) {}
det::SourceVector::SourceVector(size_type n, value_type const & val) 
    : lsst::daf::base::Citizen(typeid(*this)), _vec(n, val) {}
det::SourceVector::SourceVector(SourceVector const & v) 
    : lsst::daf::base::Citizen(typeid(*this)), _vec(v._vec) {}
det::SourceVector::SourceVector(Vector const & v)
    : lsst::daf::base::Citizen(typeid(*this)), _vec(v) {}

det::SourceVector::~SourceVector() {}

det::SourceVector & det::SourceVector::operator=(SourceVector const & v) {
    if (this != &v) {
        _vec = v._vec;
    }
    return *this;
}

det::SourceVector & det::SourceVector::operator=(Vector const & v) {
    _vec = v;
    return *this;
}
