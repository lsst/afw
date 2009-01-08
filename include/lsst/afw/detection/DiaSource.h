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
    DIA_SOURCE_2_ID = 0
    OBJECT_ID,
    MOVING_OBJECT_ID,
    SSM_ID,
    RA_ERR_4_WCS,
    DEC_ERR_4_WCS,
    X_FLUX,
    X_FLUX_ERR,
    Y_FLUX,
    Y_FLUX_ERR,
    RA_FLUX,
    RA_FLUX_ERR,
    DEC_FLUX,
    DEC_FLUX_ERR,
    X_PEAK,
    Y_PEAK,
    RA_PEAK,
    DEC_PEAK,
    X_ASTROM,
    X_ASTROM_ERR,
    Y_ASTROM,
    Y_ASTROM_ERR,
    RA_ASTROM,
    RA_ASTROM_ERR,
    DEC_ASTROM,
    DEC_ASTROM_ERR,
    MODEL_MAG_ERR,
    NON_GRAY_CORR_MAG,
    NON_GRAY_CORR_MAG_ERR,
    ATM_CORR_MAG,        
    ATM_CORR_MAG_ERR,
    AP_DIA,
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
    FLAG_4_ASSOCIATION,
    FLAG_4_DETECTION,
    FLAG_4_WCS,
    FLAG_CLASSIFICATION,
    NUM_NULLABLE_FIELDS
};
    
class DiaSource 
	: public BaseSourceAttributes<DiaSourceNullableField::NUM_NULLABLE_FIELDS>{
	typedef DiaSourceNullableField Field;
public :
    typedef boost::shared_ptr<DiaSource> Ptr;

    DiaSource() {setNull();}
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
    int64_t getFlagClassification const { return _flagClassification; }

    // setters
    void setDiaSourceId (int64_t const diaSourceId) {setId(diaSourceId);}
    void setDiaSource2Id  (int64_t const diaSource2Id) {
        set(_diaSource2Id, diaSource2Id, Field::DIA_SOURCE_2_ID);
    }
    void setScId (int32_t const scId) {
        set(_scId, scId);        
    }
    void setSsmId (int64_t const ssmId) {
        set(_ssmId, ssmId, Field::SSM_ID);
    } 
    void setLengthDeg (double  const lengthDeg) {
        set(_lengthDeg, lengthDeg);
    }        
    void setFlux (double  const flux) { 
        set(_flux, flux);             
    }
    void setFluxErr (double  const fluxErr ) { 
        set(_fluxErr, fluxErr);          
    }
    void setRefMag (float const refMag) {
        set(_refMag, refMag, Field::REF_MAG);
    }
    void setIxx (float const ixx) { 
        set(_ixx, ixx, Field::IXX);    
    }
    void setIxxErr (float const ixxErr) {
        set(_ixxErr, ixxErr, Field::IXX_ERR); 
    }         
    void setIyy (float const iyy) { 
        set(_iyy, iyy, Field::IYY);    
    }     
    void setIyyErr (float const iyyErr) { 
        set(_iyyErr, iyyErr, Field::IYY_ERR); 
    }         
    void setIxy (float const ixy) { 
        set(_ixy, ixy, IXY);    
    }      
    void setIxyErr (float const ixyErr) { 
        set(_ixyErr, ixyErr, Field::IXY_ERR); 
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
        set(_obsCode, obsCode, Field::OBS_CODE);
    }   
    void setIsSynthetic (char const isSynthetic) {
        set(_isSynthetic, isSynthetic, Field::IS_SYNTHETIC);
    } 
    void setMopsStatus (char const mopsStatus) {
        set(_mopsStatu, mopsStatus, Field::MOPS_STATUS);        
    }
    void setFlagClassification(int64_t const flagClassification) {
        set(_flagClassification, flagClassification, Field::FLAG_CLASSIFICATION);
    }
        
   //overloaded setters
    void setObjectId (int64_t const objectId) {
        set(_objectId, objectId, Field::OBJECT_ID);
    }
    void setMovingObjectId (int64_t const movingObjectId) {
    	set(_movingObjectId, movingObjectId, Field::MOVING_OBJECT_ID);
    }
    void setRaErr4wcs (float const raErr4wcs) { 
        set(_raErr4wcs, raErr4wcs, Field::RA_ERR_4_WCS);  
    }
    void setDecErr4wcs(float const decErr4wcs) { 
        set(_decErr4wcs, decErr4wcs, Field::DEC_ERR_4_WCS); 
    }
    void setXFlux (double const xFlux) { 
        set(_xFlux, xFlux, X_FLUX);            
    }
    void setXFluxErr (double const xFluxErr) { 
        set(_xFluxErr, xFluxErr, Field::X_FLUX_ERR);            
    }    
    void setYFlux (double const yFlux) { 
        set(_yFlux, yFlux, Field::Y_FLUX);            
    }    
    void setYFluxErr (double const yFluxErr) { 
        set(_yFluxErr, yFluxErr, Field::Y_FLUX_ERR);            
    }    
    void setRaFlux (double const raFlux) { 
        set(_raFlux, raFlux, Field::RA_FLUX);            
    }
    void setRaFluxErr (double const raFluxErr) { 
        set(_raFluxErr, raFluxErr, Field::RA_FLUX_ERR);            
    }    
    void setDecFlux (double const decFlux) { 
        set(_decFlux, decFlux, Field::DEC_FLUX);
    }    
    void setDecFluxErr (double const decFluxErr) { 
        set(_decFluxErr, decFluxErr, Field::DEC_FLUX_ERR);            
    }    
    void setXPeak (double const xPeak) { 
        set(_xPeak, xPeak, Field::X_PEAK);            
    }
    void setYPeak (double const yPeak) { 
        set(_yPeak, yPeak, Field::Y_PEAK);            
    }    
    void setRaPeak (double const raPeak) { 
        set(_raPeak, raPeak, Field::RA_PEAK);            
    }    
    void setDecPeak (double const decPeak) { 
        set(_decPeak, decPeak, Field::DEC_PEAK);            
    }    
    void setXAstrom (double const xAstrom) { 
        set(_xAstrom, xAstrom, Field::X_ASTROM);            
    }
    void setXastromErr (double const xAstromErr) { 
        set(_xAstromErr, xAstromErr, Field::X_ASTROM_ERR);            
    }    
    void setYAstrom (double const yAstrom) { 
        set(_yAstrom, yAstrom, Field::Y_ASTROM);            
    }    
    void setYAstromErr (double const yAstromErr) { 
        set(_yAstromErr, yAstromErr, Field::Y_ASTROM_ERR);            
    }    
    void setRaAstrom (double const raAstrom) { 
        set(_raAstrom, raAstrom, Field::RA_ASTROM);            
    }
    void setRaAstromErr (double const raAstromErr) { 
        set(_raAstromErr, raAstromErr, Field::RA_ASTROM_ERR);            
    }    
    void setDecAstrom (double const decAstrom) { 
        set(_decAstrom, decAstrom, Field::DEC_ASTROM);            
    }    
    void setDecAstromErr (double const decAstromErr) { 
        set(_decAstromErr, decAstromErr, Field::DEC_ASTROM_ERR);            
    }         
    void setModelMagErr(float const modelMagErr) {
    	set(_modelMagErr, modelMagErr, Field::MODEL_MAG_ERR);
    }
    void setNonGrayCorrMag (double const nonGrayCorrMag) { 
        set(_instMag, instMag, Field::NON_GRAY_CORR_MAG);         
    }
    void setNonGrayCorrMagErr(double const nonGrayCorrMagErr) { 
        set(_nonGrayCorrMagErr, nonGrayCorrMagErr, Field::NON_GRAY_CORR_MAG_ERR);      
    }
    void setAtmCorrMag (double const atmCorrMag) { 
        set(_instMag, instMag, Field::ATM_CORR_MAG);         
    }
    void setAtmCorrErr (double const atmCorrErr) { 
        set(_atmCorrErr, atmCorrErr, Field::ATM_CORR_MAG_ERR);      
    }        
    void setApDia (float const apDia) {
        set(_apDia, apDia, Field::AP_DIA);
    }
    void setFlag4association(int16_t const flag4association) {
        set(_flag4association, flag4association, Field::FLAG_4_ASSOCIATION);
    }
    void setFlag4detection (int16_t const flag4detection) {
        set(_flag4detection, flag4detection, Field::FLAG_4_DETECTION);
    }
    void setFlag4wcs (int16_t const flag4wcs) {
        set(_flag4wcs, flag4wcs, Field::FLAG_4_WCS);
    }
    
    bool operator==(DiaSource const & d) const;

private :

    int64_t _ssmId;            // BIGINT        NULL
    int64_t _diaSource2Id;     // BIGINT        NULL
    int64_t _flagClassification;// BIGINT       NULL
    double  _lengthDeg;        // DOUBLE        NOT NULL 
    double  _valX1;            // DOUBLE        NOT NULL         
    double  _valX1;            // DOUBLE        NOT NULL
    double  _valY1;            // DOUBLE        NOT NULL
    double  _valY2;            // DOUBLE        NOT NULL
    double  _valXY;            // DOUBLE        NOT NULL        
    float   _flux;             // DECIMAL(12,2) NOT NULL
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

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {    
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

        BaseSourceAttributes::serialze(ar, version);
    }

    friend class boost::serialization::access;
    friend class formatters::DiaSourceVectorFormatter;
};

inline bool operator!=(DiaSource const & d1, DiaSource const & d2) {
    return !(d1 == d2);
}

class PersistableDiaSourceVector : public Persistable {
    typedef std::vector<DiaSource> DiaSourceVector;
public:
    PersistableDiaSourceVector() {}
    PersistableDiaSourceVector(DiaSourceVector const & sources)
        : _sources(sources) {}
        
    DiaSourceVector & getSources() {return _sources; }
    DiaSourceVector getSources() const {return _sources; } 
    
    void setSources(DiaSourceVector const & sources) {_sources = sources; }
private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::DiaSourceVectorFormatter);
    DiaSourceVector _sources;
}; 

}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_DIASOURCE_H


