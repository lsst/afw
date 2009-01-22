// -*- lsst-c++ -*-
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

#ifndef SWIG
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;
#endif

/*! An integer id for each nullable field in DiaSource. */
enum DiaSourceNullableField {
    DIA_SOURCE_2_ID = NUM_SHARED_NULLABLE_FIELDS,
    SSM_ID,
    RA_ERR_4_WCS,
    DEC_ERR_4_WCS,
    MODEL_MAG_ERR,
    REF_MAG,
    IXX,
    IXX_ERR,
    IYY,
    IYY_ERR,
    IXY,
    IXY_ERR,
    OBS_CODE,
    IS_SYNTHETIC,
    MOPS_STATUS,
    FLAG_CLASSIFICATION,
    NUM_DIASOURCE_NULLABLE_FIELDS
};

class DiaSource 
	: public BaseSourceAttributes< NUM_DIASOURCE_NULLABLE_FIELDS> {
public :
    typedef boost::shared_ptr<DiaSource> Ptr;

    DiaSource();
    DiaSource(DiaSource const & other);
    
    virtual ~DiaSource() {};

    // getters    
    int64_t getDiaSourceId()	  const { return getId();   	      }
    int64_t getDiaSource2Id()     const { return _diaSource2Id;       }
    int32_t getScId()             const { return _scId;               }
    int64_t getSsmId()            const { return _ssmId;              }
    double  getLengthDeg()        const { return _lengthDeg;          }         
    float   getFlux()             const { return _flux;               }
    float   getFluxErr()          const { return _fluxErr;            }
    float   getRefMag()           const { return _refMag;             }
    float   getIxx()              const { return _ixx;                }
    float   getIxxErr()           const { return _ixxErr;             }
    float   getIyy()              const { return _iyy;                }
    float   getIyyErr()           const { return _iyyErr;             }
    float   getIxy()              const { return _ixy;                }
    float   getIxyErr()           const { return _ixyErr;             }
    double  getValX1()            const { return _valX1;              }
    double  getValX2()            const { return _valX2;              }
    double  getValY1()            const { return _valY1;              }
    double  getValY2()            const { return _valY2;              }
    double  getValXY()            const { return _valXY;              }  
    char    getObsCode()          const { return _obsCode;            }
    char    isSynthetic()         const { return _isSynthetic;        }
    char    getMopsStatus()       const { return _mopsStatus;         }
    int64_t getFlagClassification() const { return _flagClassification; }

    // setters
    void setDiaSourceId (int64_t const diaSourceId) {setId(diaSourceId);}
    void setDiaSource2Id  (int64_t const diaSource2Id) {
        set(_diaSource2Id, diaSource2Id,  DIA_SOURCE_2_ID);
    }
    void setScId (int32_t const scId) {
        set(_scId, scId);        
    }
    void setSsmId (int64_t const ssmId) {
        set(_ssmId, ssmId,  SSM_ID);
    } 
    void setLengthDeg (double  const lengthDeg) {
        set(_lengthDeg, lengthDeg);
    }        
    void setFlux (float  const flux) { 
        set(_flux, flux);             
    }
    void setFluxErr (float  const fluxErr ) { 
        set(_fluxErr, fluxErr);          
    }
    void setRefMag (float const refMag) {
        set(_refMag, refMag,  REF_MAG);
    }
    void setIxx (float const ixx) { 
        set(_ixx, ixx,  IXX);    
    }
    void setIxxErr (float const ixxErr) {
        set(_ixxErr, ixxErr,  IXX_ERR); 
    }         
    void setIyy (float const iyy) { 
        set(_iyy, iyy,  IYY);    
    }     
    void setIyyErr (float const iyyErr) { 
        set(_iyyErr, iyyErr,  IYY_ERR); 
    }         
    void setIxy (float const ixy) { 
        set(_ixy, ixy,  IXY);    
    }      
    void setIxyErr (float const ixyErr) { 
        set(_ixyErr, ixyErr,  IXY_ERR); 
    }         
    void setValX1 (double  const valX1) {
        set(_valX1, valX1);
    }
    void setValX2 (double  const valX2) {
        set(_valX2, valX2);
    }
    void setValY1 (double  const valY1) {
        set(_valY1, valY1);
    }
    void setValY2 (double  const valY2) {
        set(_valY2, valY2);
    }
    void setValXY           (double  const valXY           ) {
        set(_valXY, valXY);
    }         
    void setObsCode (char const obsCode) {
        set(_obsCode, obsCode,  OBS_CODE);
    }   
    void setIsSynthetic (char const isSynthetic) {
        set(_isSynthetic, isSynthetic,  IS_SYNTHETIC);
    } 
    void setMopsStatus (char const mopsStatus) {
        set(_mopsStatus, mopsStatus,  MOPS_STATUS);        
    }
    void setFlagClassification(int64_t const flagClassification) {
        set(_flagClassification, flagClassification,  FLAG_CLASSIFICATION);
    }
        
   //overloaded setters
    void setRaErr4wcs (float const raErr4wcs) { 
        set(_raErr4wcs, raErr4wcs,  RA_ERR_4_WCS);  
    }
    void setDecErr4wcs(float const decErr4wcs) { 
        set(_decErr4wcs, decErr4wcs,  DEC_ERR_4_WCS); 
    }
    void setModelMagErr(float const modelMagErr) {
    	set(_modelMagErr, modelMagErr,  MODEL_MAG_ERR);
    }

    
    bool operator==(DiaSource const & d) const;

private :

    int64_t _ssmId;            // BIGINT        NULL
    int64_t _diaSource2Id;     // BIGINT        NULL
    int64_t _flagClassification;// BIGINT       NULL
    double  _lengthDeg;        // DOUBLE        NOT NULL 
    double  _valX1;            // DOUBLE        NOT NULL         
    double  _valX2;            // DOUBLE        NOT NULL
    double  _valY1;            // DOUBLE        NOT NULL
    double  _valY2;            // DOUBLE        NOT NULL
    double  _valXY;            // DOUBLE        NOT NULL        
    float  _flux;             // DECIMAL(12,2) NOT NULL
    float   _fluxErr;          // DECIMAL(10,2) NOT NULL    
    float   _refMag;           // FLOAT(0)      NULL
    float   _ixx;              // FLOAT(0)      NULL
    float   _ixxErr;           // FLOAT(0)      NULL
    float   _iyy;              // FLOAT(0)      NULL
    float   _iyyErr;           // FLOAT(0)      NULL
    float   _ixy;              // FLOAT(0)      NULL
    float   _ixyErr;           // FLOAT(0)      NULL
    int32_t _scId;             // INTEGER       NOT NULL 
    char    _obsCode;          // CHAR(3)       NULL       
    char    _isSynthetic;      // CHAR          NULL
    char    _mopsStatus;       // CHAR          NULL

    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version) {    
        ar & _diaSource2Id;
        ar & _scId;
        ar & _ssmId;        
        ar & _lengthDeg;        
        ar & _flux;
        ar & _fluxErr;
        ar & _refMag;
        ar & _ixx;
        ar & _ixxErr;
        ar & _iyy;
        ar & _iyyErr;
        ar & _ixy;
        ar & _ixyErr;
        ar & _valX1;
        ar & _valX2;
        ar & _valY1;
        ar & _valY2;
        ar & _valXY;
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

typedef std::vector<DiaSource> DiaSourceVector;

class PersistableDiaSourceVector : public lsst::daf::base::Persistable {
public:
	typedef boost::shared_ptr<PersistableDiaSourceVector> Ptr;
    PersistableDiaSourceVector() {}
    PersistableDiaSourceVector(DiaSourceVector const & sources)
        : _sources(sources) {}
        
    DiaSourceVector getSources() const {return _sources; }
    void setSources(DiaSourceVector const & sources) {_sources = sources; }
    	
    bool operator==(DiaSourceVector const & other) const {
        if(_sources.size() != other.size())
        	return false;
        	        
    	DiaSourceVector::size_type i;
        for(i =0; i<_sources.size();i++) {
    		if(_sources[i] != other[i])
    			return false;			
    	}
    	
    	return true;    	
    }

    bool operator==(PersistableDiaSourceVector const & other) const {
		return other==_sources;
	}

private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::DiaSourceVectorFormatter);
    DiaSourceVector _sources;
}; 



}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_DIASOURCE_H


