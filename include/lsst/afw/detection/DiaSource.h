// -*- lsst-c++ -*-

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
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  The C++ representation of a Difference-Image-Analysis Source.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_DETECTION_DIASOURCE_H
#define LSST_AFW_DETECTION_DIASOURCE_H

#include <bitset>
#include <string>
#include <vector>


#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/serialization/shared_ptr.hpp"

#include "lsst/daf/base/Citizen.h"
#include "lsst/daf/base/Persistable.h"

#include "lsst/afw/detection/BaseSourceAttributes.h"


namespace boost {
namespace serialization {
    class access;
}}

namespace lsst {
namespace afw {
    namespace formatters {
        class DiaSourceVectorFormatter;
    }
namespace detection {

/*! An integer id for each nullable field in DiaSource. */
enum DiaSourceNullableField {
    DIA_SOURCE_TO_ID = NUM_SHARED_NULLABLE_FIELDS,
    SSM_ID,
    RA_ERR_FOR_WCS,
    DEC_ERR_FOR_WCS,
    PSF_FLUX_ERR,
    AP_FLUX_ERR,
    MODEL_FLUX_ERR,
    INST_FLUX_ERR,
    REF_FLUX,
    OBS_CODE,
    IS_SYNTHETIC,
    MOPS_STATUS,
    FLAG_CLASSIFICATION,
    NUM_DIASOURCE_NULLABLE_FIELDS
};

/**
 * In-code representation of an entry in the DIASource catalog for
 *   persisting/retrieving DiaSources
 */
class DiaSource 
    : public BaseSourceAttributes<NUM_DIASOURCE_NULLABLE_FIELDS> {
public :
    typedef boost::shared_ptr<DiaSource> Ptr;

    DiaSource();
    DiaSource(DiaSource const & other);
    ~DiaSource() {};

    // getters    
    boost::int64_t getDiaSourceId() const { return getId(); }
    boost::int64_t getDiaSourceToId() const { return _diaSourceToId; }
    boost::int32_t getScId() const { return _scId; }
    boost::int64_t getSsmId() const { return _ssmId; }
    double  getLengthDeg() const { return _lengthDeg; }
    float   getRefFlux() const { return _refFlux; }
    double  getValX1() const { return _valX1; }
    double  getValX2() const { return _valX2; }
    double  getValY1() const { return _valY1; }
    double  getValY2() const { return _valY2; }
    double  getValXY() const { return _valXY; }
    std::string getObsCode() const { return _obsCode; }
    char    isSynthetic() const { return _isSynthetic; }
    char    getMopsStatus() const { return _mopsStatus; }
    boost::int64_t  getFlagClassification() const { return _flagClassification; }

    // setters
    void setDiaSourceId(boost::int64_t const diaSourceId) {setId(diaSourceId);}
    void setDiaSourceToId(boost::int64_t const diaSourceToId) {
        set(_diaSourceToId, diaSourceToId,  DIA_SOURCE_TO_ID);
    }
    void setScId(boost::int32_t const scId) {
        set(_scId, scId);        
    }
    void setSsmId(boost::int64_t const ssmId) {
        set(_ssmId, ssmId, SSM_ID);
    } 
    void setLengthDeg(double  const lengthDeg) {
        set(_lengthDeg, lengthDeg);
    }        
    void setRefFlux(float const refFlux) {
        set(_refFlux, refFlux, REF_FLUX);
    }

    void setValX1(double  const valX1) {
        set(_valX1, valX1);
    }
    void setValX2(double  const valX2) {
        set(_valX2, valX2);
    }
    void setValY1(double  const valY1) {
        set(_valY1, valY1);
    }
    void setValY2(double  const valY2) {
        set(_valY2, valY2);
    }
    void setValXY(double  const valXY) {
        set(_valXY, valXY);
    }         
    void setObsCode(std::string const& obsCode) {
        set(_obsCode, obsCode, OBS_CODE);
    }   
    void setIsSynthetic(char const isSynthetic) {
        set(_isSynthetic, isSynthetic, IS_SYNTHETIC);
    } 
    void setMopsStatus(char const mopsStatus) {
        set(_mopsStatus, mopsStatus, MOPS_STATUS);        
    }
    void setFlagClassification(boost::int64_t const flagClassification) {
        set(_flagClassification, flagClassification, FLAG_CLASSIFICATION);
    }
        
    //overloaded setters
    //Because these fields are not NULLABLE in all sources, 
    //  special behavior must be defined in the derived class
    // in radians
    void setRaErrForWcs(float const raErrForWcs) { 
        set(_raErrForWcs, raErrForWcs, RA_ERR_FOR_WCS);  
    }
    // in radians
    void setDecErrForWcs(float const decErrForWcs) { 
        set(_decErrForWcs, decErrForWcs, DEC_ERR_FOR_WCS); 
    }
    void setPsfFluxErr(float const psfFluxErr) {
        set(_psfFluxErr, psfFluxErr, PSF_FLUX_ERR);
    }
    void setApFluxErr(float const apFluxErr) {
        set(_apFluxErr, apFluxErr, AP_FLUX_ERR);
    }
    void setModelFluxErr(float const modelFluxErr) {
        set(_modelFluxErr, modelFluxErr, MODEL_FLUX_ERR);
    }
    void setInstFluxErr(float const instFluxErr) {
        set(_instFluxErr, instFluxErr, INST_FLUX_ERR);
    }


    
    bool operator==(DiaSource const & d) const;
private :
    boost::int64_t _ssmId;
    boost::int64_t _diaSourceToId;
    boost::int64_t _flagClassification;
    double _lengthDeg;
    double _valX1;
    double _valX2;
    double _valY1;
    double _valY2;
    double _valXY;
    float _refFlux;
    boost::int32_t _scId;
    std::string _obsCode;
    char _isSynthetic;
    char _mopsStatus;


    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version) {    
        ar & _diaSourceToId;
        ar & _scId;
        ar & _ssmId;
        fpSerialize(ar, _lengthDeg);
        fpSerialize(ar, _refFlux);
        fpSerialize(ar, _valX1);
        fpSerialize(ar, _valX2);
        fpSerialize(ar, _valY1);
        fpSerialize(ar, _valY2);
        fpSerialize(ar, _valXY);
        ar & _obsCode;
        ar & _isSynthetic;
        ar & _mopsStatus;
        ar & _flagClassification;

        BaseSourceAttributes< NUM_DIASOURCE_NULLABLE_FIELDS>::serialize<Archive>(ar, version);
    }

    friend class boost::serialization::access;
    friend class formatters::DiaSourceVectorFormatter;
};

inline bool operator!=(DiaSource const & d1, DiaSource const & d2) {
    return !(d1 == d2);
}

typedef std::vector<DiaSource::Ptr> DiaSourceSet;

class PersistableDiaSourceVector : public lsst::daf::base::Persistable {
public:
    typedef boost::shared_ptr<PersistableDiaSourceVector> Ptr;
    PersistableDiaSourceVector() {}
    PersistableDiaSourceVector(DiaSourceSet const & sources)
        : _sources(sources) {}
    ~PersistableDiaSourceVector(){_sources.clear();}
    DiaSourceSet getSources() const {return _sources; }
    void setSources(DiaSourceSet const & sources) {_sources = sources; }
        
    bool operator==(DiaSourceSet const & other) const {
        if (_sources.size() != other.size())
            return false;
                    
        DiaSourceSet::size_type i;
        for (i = 0; i < _sources.size(); ++i) {
            if (*_sources[i] != *other[i])
                return false;            
        }
        
        return true;        
    }

    bool operator==(PersistableDiaSourceVector const & other) const {
        return other == _sources;
    }

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::DiaSourceVectorFormatter)
    DiaSourceSet _sources;
}; 



}}}  // namespace lsst::afw::detection

#ifndef SWIG
BOOST_CLASS_VERSION(lsst::afw::detection::DiaSource, 2)
#endif

#endif // LSST_AFW_DETECTION_DIASOURCE_H


