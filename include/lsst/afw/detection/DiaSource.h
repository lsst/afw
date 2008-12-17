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

// forward declarations for formatters


/**
   \brief Contains attributes for Difference-Image-Analysis Source records.

   This class is useful
   when an unadorned data structure is required (e.g. for placement into shared memory) or
   is all that is necessary.

    The C++ fields are derived from the LSST DC2 MySQL schema, which is reproduced below:

    \code
    CREATE TABLE DIASource
    (
        diaSourceId      BIGINT         NOT NULL,
        ampExposureId    BIGINT         NOT NULL,
        filterId         TINYINT        NOT NULL,
        objectId         BIGINT         NULL,
        movingObjectId   BIGINT         NULL,
        procHistoryId    INTEGER        NOT NULL,
        scId             INTEGER        NOT NULL,
        ssmId            BIGINT         NULL
        ra               DOUBLE(12,9)   NOT NULL,
        decl             DOUBLE(11,9)   NOT NULL,
        raErr4detection  FLOAT(0)       NOT NULL,
        decErr4detection FLOAT(0)       NOT NULL,
        raErr4wcs        FLOAT(0)       NULL,
        decErr4wcs       FLOAT(0)       NULL,
        xFlux            DOUBLE(10)     NULL,
        xFluxErr         DOUBLE(10)     NULL,
        yFlux            DOUBLE(10)     NULL,
        yFluxErr         DOUBLE(10)     NULL,
        raFlux           DOUBLE(10)     NULL,
        raFluxErr        DOUBLE(10)     NULL,
        declFlux         DOUBLE(10)     NULL,
        declFluxErr      DOUBLE(10)     NULL,
        xPeak            DOUBLE(10)     NULL,
        yPeak            DOUBLE(10)     NULL,
        raPeak           DOUBLE(10)     NULL,
        declPeak         DOUBLE(10)     NULL,
        xAstrom          DOUBLE(10)     NULL,
        xAstromErr       DOUBLE(10)     NULL,
        yAstrom          DOUBLE(10)     NULL,
        yAstromErr       DOUBLE(10)     NULL,
        raAstrom         DOUBLE(10)     NULL,
        raAstromErr      DOUBLE(10)     NULL,
        declAstrom       DOUBLE(10)     NULL,
        declAstromErr    DOUBLE(10)     NULL,        
        taiMidPoint      DOUBLE(12,7)   NOT NULL,
        taiRange         FLOAT(0)       NOT NULL,
        fwhmA            FLOAT(0)       NOT NULL,
        fwhmB            FLOAT(0)       NOT NULL,
        fwhmTheta        FLOAT(0)       NOT NULL,
        lengthDeg        DOUBLE(0)      NOT NULL,
        flux             FLOAT(0)       NOT NULL,
        fluxErr          FLOAT(0)       NOT NULL,
        psfMag           DOUBLE(7,3)    NOT NULL,
        psfMagErr        FLOAT(0)       NOT NULL,
        apMag            DOUBLE(7,3)    NOT NULL,
        apMagErr         FLOAT(0)       NOT NULL,
        modelMag         DOUBLE(6,3)    NOT NULL,
        modelMagErr      FLOAT(0)       NULL,        
        apDia            FLOAT(0)       NULL,
        refMag           FLOAT(0)       NULL
        Ixx              FLOAT(0)       NULL,
        IxxErr           FLOAT(0)       NULL,
        Iyy              FLOAT(0)       NULL,
        IyyErr           FLOAT(0)       NULL,
        Ixy              FLOAT(0)       NULL,
        IxyErr           FLOAT(0)       NULL,
        snr              FLOAT(0)       NOT NULL,
        chi2             FLOAT(0)       NOT NULL,
        valx1            DOUBLE(10)     NOT NULL,
        valx2            DOUBLE(10)     NOT NULL,
        valy1            DOUBLE(10)     NOT NULL,
        valy2            DOUBLE(10)     NOT NULL,
        valxy            DOUBLE(10)     NOT NULL,
        obsCode          CHAR(3)        NULL,
        isSynthetic      CHAR(1)        NULL,
        mopsStatus       CHAS(1)        NULL,
        flag4association SMALLINT      NULL,
        flag4detection   SMALLINT      NULL,
        flag4wcs         SMALLINT      NULL,
        PRIMARY KEY (diaSourceId),
        KEY (ampExposureId),
        KEY (filterId),
        KEY (movingObjectId),
        KEY (objectId),
        KEY (procHistoryId),
        INDEX idx_DIASOURCE_ssmId (ssmId ASC),        
        KEY (scId)
        INDEX idx_DIA_SOURCE_psfMag (psfMag ASC),
        INDEX idx_DIASOURCE_taiMidPoint (taiMidPoint ASC)
    ) ENGINE=MyISAM;
    \endcode

    Note that the C++ fields are listed in a different order than their corresponding database
    columns: fields are sorted by type size to minimize the number of padding bytes that the
    compiler must insert to meet field alignment requirements.
 */
class DiaSource : public BaseSourceAttributes{

public :

    typedef boost::shared_ptr<DiaSource> Ptr;

    /*! An integer id for each nullable field. */
    enum NullableField {
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

    DiaSource();
    virtual ~DiaSource();

    // getters    
    int64_t * getDiaSource2Id()     const { return _diaSource2Id);      }
    int32_t * getScId()             const { return _scId;               }
    int64_t * getSsmId()            const { return _ssmId;              }
    double  * getLengthDeg()        const { return _lengthDeg;          }         
    float   * getFlux()             const { return _flux;               }
    float   * getFluxErr()          const { return _fluxErr;            }
    float   * getRefMag()           const { return _refMag;             }
    float   * getIxx()              const { return _ixx;                }
    float   * getIxxErr()           const { return _ixxErr;             }
    float   * getIyy()              const { return _iyy;                }
    float   * getIyyErr()           const { return _iyyErr;             }
    float   * getIxy()              const { return _ixy;                }
    float   * getIxyErr()           const { return _ixyErr;             }
    double  * getValX1()            const { return _valX1;              }
    double  * getValX2()            const { return _valX2;              }
    double  * getValY1()            const { return _valY1;              }
    double  * getValY2()            const { return _valY2;              }
    double  * getValXY()            const { return _valXY;              }  
    char    * getObsCode()          const { return _obsCode;            }
    char    * isSynthetic()         const { return _isSynthetic;        }
    char    * getMopsStatus()       const { return _mopsStatus;         }
    int64_t * getFlagClassification const { return _flagClassification; }

    // setters
    void setDiaSource2Id  (int64_t const diaSource2Id      ) {
        set(_diaSource2Id, diaSource2Id);
    }
    void setScId          (int32_t const scId              ) {
        set(_scId, scId);        
    }
    void setSsmId         (int64_t const ssmId             ) {
        set(_ssmId, ssmId);
    } 
    void setLengthDeg       (double  const lengthDeg       ) {
        set(_lengthDeg, lengthDeg);
    }        
    void setFlux            (double  const flux            ) { 
        set(_flux, flux);             
    }
    void setFluxErr         (double  const fluxErr         ) { 
        set(_fluxErr, fluxErr);          
    }
    void setRefMag          (float const refMag            ) {
        set(_refMag, refMag);
    }
    void setIxx             (float const ixx               ) { 
        set(_ixx, ixx);    
    }
    void setIxxErr          (float const ixxErr            ) { 
        set(_ixxErr, ixxErr); 
    }         
    void setIyy             (float const iyy               ) { 
        set(_iyy, iyy);    
    }     
    void setIyyErr          (float const iyyErr            ) { 
        set(_iyyErr, iyyErr); 
    }         
    void setIxy             (float const ixy               ) { 
        set(_ixy, ixy);    
    }      
    void setIxyErr          (float const ixyErr            ) { 
        set(_ixyErr, ixyErr); 
    }         
    void setValX1           (double  const valX1           ) {
        set(_valX1, valX1);
    }
    void setValX2           (double  const valX2           ) {
        set(_valX2, valX2);
    }
    void setValY1           (double  const valY1           ) {
        set(_valY1, valY1);
    }
    void setValY2           (double  const valY2           ) {
        set(_valY2, valY2);
    }
    void setValXY           (double  const valXY           ) {
        set(_valXY, valXY);
    }         
    void setObsCode         (char    const obsCode         ) {
        set(_obsCode, obsCode);
    }   
    void setIsSynthetic     (char    const isSynthetic     ) {
        set(_isSynthetic, isSynthetic);
    } 
    void setMopsStatus      (char    const mopsStatus      ) {
        set(_mopsStatu, mopsStatus);        
    }
    void setFlagClassification(int64_t const flagClassification) {
        set(_flagClassification, flagClassification);
    }
        


    // Get/set whether or not fields are null
    bool isNull    (NullableField const f) const;            
    void setNull   (NullableField const f);                  
    void setNotNull(NullableField const f);                  
    void setNull   (NullableField const f, bool const null); 
    void setNull   ();                                       
    void setNotNull();                                       

    virtual void setAllNotNull();
    virtual void setAllNull();
    
    bool operator==(DiaSource const & d) const;

private :

    int64_t * _ssmId;            // BIGINT        NULL
    int64_t * _diaSource2Id;     // BIGINT        NULL
    int64_t * _flagClassification;// BIGINT       NULL
    double  * _lengthDeg;        // DOUBLE        NOT NULL 
    double  * _valX1;            // DOUBLE        NOT NULL         
    double  * _valX1;            // DOUBLE        NOT NULL
    double  * _valY1;            // DOUBLE        NOT NULL
    double  * _valY2;            // DOUBLE        NOT NULL
    double  * _valXY;            // DOUBLE        NOT NULL        
    float   * _flux;             // DECIMAL(12,2) NOT NULL
    float   * _fluxErr;          // DECIMAL(10,2) NOT NULL    
    float   * _refMag;           // FLOAT(0)      NULL
    float   * _ixx;              // FLOAT(0)      NULL
    float   * _ixxErr;           // FLOAT(0)      NULL
    float   * _iyy;              // FLOAT(0)      NULL
    float   * _iyyErr;           // FLOAT(0)      NULL
    float   * _ixy;              // FLOAT(0)      NULL
    float   * _ixyErr;           // FLOAT(0)      NULL
    int32_t * _scId;             // INTEGER       NOT NULL 
    char    * _obsCode;          // CHAR(3)       NULL       
    char    * _isSynthetic;      // CHAR          NULL
    char    * _mopsStatus;       // CHAR          NULL

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        BaseSourceAttributes::serialze(ar, version);
        
        serializeData(ar, version, _diaSource2Id);
        serializeData(ar, version, _scId);
        serializeData(ar, version, _ssmId);        
        serializeData(ar, version, _lengthDeg);        
        serializeData(ar, version, _flux);
        serializeData(ar, version, _fluxErr);
        serializeData(ar, version, _refMag);
        serializeData(ar, version, _ixx);
        serializeData(ar, version, _ixxErr);
        serializeData(ar, version, _iyy);
        serializeData(ar, version, _iyyErr);
        serializeData(ar, version, _ixy);
        serializeData(ar, version, _ixyErr);
        serializeData(ar, version, _valX1);
        serializeData(ar, version, _valX2);
        serializeData(ar, version, _valY1);
        serializeData(ar, version, _valY2);
        serializeData(ar, version, _valXY);
        serializeData(ar, version, _obsCode);
        serializeData(ar, version, _isSynthetic);
        serializeData(ar, version, _mopsStatus);
        serializeData(ar, version, _flagClassification);

        bool b;
        //go through list of nullable fields,
        //store true if field is NULL
        //false if NOT NULL        
        if (Archive::is_loading::value) {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                ar & b;
                setNull(i, b);
            }
        } else {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                b = isNull(i);
                ar & b;
            }
        }
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


