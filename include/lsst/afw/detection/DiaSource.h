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
        ra               DOUBLE         NOT NULL,
        decl             DOUBLE         NOT NULL,
        raErr4detection  FLOAT(0)       NOT NULL,
        decErr4detection FLOAT(0)       NOT NULL,
        raErr4wcs        FLOAT(0)       NULL,
        decErr4wcs       FLOAT(0)       NULL,
        xFlux            DOUBLE         NULL,
        xFluxErr         DOUBLE         NULL,
        yFlux            DOUBLE         NULL,
        yFluxErr         DOUBLE         NULL,
        raFlux           DOUBLE         NULL,
        raFluxErr        DOUBLE         NULL,
        declFlux         DOUBLE         NULL,
        declFluxErr      DOUBLE         NULL,
        xPeak            DOUBLE         NULL,
        yPeak            DOUBLE         NULL,
        raPeak           DOUBLE         NULL,
        declPeak         DOUBLE         NULL,
        xAstrom          DOUBLE         NULL,
        xAstromErr       DOUBLE         NULL,
        yAstrom          DOUBLE         NULL,
        yAstromErr       DOUBLE         NULL,
        raAstrom         DOUBLE         NULL,
        raAstromErr      DOUBLE         NULL,
        declAstrom       DOUBLE         NULL,
        declAstromErr    DOUBLE         NULL,        
        taiMidPoint      DOUBLE         NOT NULL,
        taiRange         FLOAT(0)       NOT NULL,
        fwhmA            FLOAT(0)       NOT NULL,
        fwhmB            FLOAT(0)       NOT NULL,
        fwhmTheta        FLOAT(0)       NOT NULL,
        lengthDeg        DOUBLE         NOT NULL,
        flux             FLOAT(0)       NOT NULL,
        fluxErr          FLOAT(0)       NOT NULL,
        psfMag           DOUBLE         NOT NULL,
        psfMagErr        FLOAT(0)       NOT NULL,
        apMag            DOUBLE         NOT NULL,
        apMagErr         FLOAT(0)       NOT NULL,
        modelMag         DOUBLE         NOT NULL,
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
        valx1            DOUBLE         NOT NULL,
        valx2            DOUBLE         NOT NULL,
        valy1            DOUBLE         NOT NULL,
        valy2            DOUBLE         NOT NULL,
        valxy            DOUBLE         NOT NULL,
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
class DiaSource {

public :

    typedef boost::shared_ptr<DiaSource> Ptr;

    /*! An integer id for each nullable field. */
    enum NullableField {

        OBJECT_ID = 0,
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
        NUM_NULLABLE_FIELDS
    };

    DiaSource();

    // getters
    int64_t getId()               const { return _diaSourceId;      }
    int64_t getAmpExposureId()    const { return _ampExposureId;    }
    int8_t  getFilterId()         const { return _filterId;         }
    int64_t getObjectId()         const { return _objectId;         }
    int64_t getMovingObjectId()   const { return _movingObjectId;   }
    int32_t getProcHistoryId()    const { return _procHistoryId;    }
    int32_t getScId()             const { return _scId;             }
    int64_t getSsmId()            const { return _ssmId;            }
    double  getRa()               const { return _ra;               }
    double  getDec()              const { return _dec;              }
    float   getRaErr4detection()  const { return _raErr4detection;  }
    float   getDecErr4detection() const { return _decErr4detection; }
    float   getRaErr4wcs()        const { return _raErr4wcs;        }
    float   getDecErr4wcs()       const { return _decErr4Wcs;       }
    double  getXFlux()            const { return _xFlux;            }
    double  getXFluxErr()         const { return _xFluxErr;         }
    double  getYFlux()            const { return _yFlux;            }
    double  getYFluxErr()         const { return _yFluxErr;         }
    double  getRaFlux()           const { return _raFlux;           }
    double  getRaFluxErr()        const { return _raFluxErr;        }    
    double  getDecFlux()          const { return _decFlux;          }
    double  getDecFluxErr()       const { return _decFluxErr;       }        
    double  getXPeak()            const { return _xPeak;            }
    double  getYPeak()            const { return _yPeak;            }
    double  getRaPeak()           const { return _raPeak;           }    
    double  getDecPeak()          const { return _decPeak;          }       
    double  getXAstrom()          const { return _xAstrom;          }
    double  getXAstromErr()       const { return _xAstromErr;       }
    double  getYAstrom()          const { return _yAstrom;          }
    double  getYAstromErr()       const { return _yAstromErr;       }
    double  getRaAstrom()         const { return _raAstrom;         }
    double  getRaAstromErr()      const { return _raAstromErr;      }    
    double  getDecAstrom()        const { return _decAstrom;        }
    double  getDecAstromErr()     const { return _decAstromErr;     }
    double  getTaiMidPoint()      const { return _taiMidPoint;      }
    float   getTaiRange()         const { return _taiRange;         }
    float   getFwhmA()            const { return _fwhmA;            }
    float   getFwhmB()            const { return _fwhmB;            }
    float   getFwhmTheta()        const { return _fwhmTheta;        }
    double  getLengthDeg()        const { return _lengthDeg;        }        
    float   getFlux()             const { return _flux;             }
    float   getFluxErr()          const { return _fluxErr;          }
    double  getPsfMag()           const { return _psfMag;           }
    float   getPsfMagErr()        const { return _psfMagErr;        }
    double  getApMag()            const { return _apMag;            }
    float   getApMagErr()         const { return _apMagErr;         }
    double  getModelMag()         const { return _modelMag;         }
    float   getModelMagErr()      const { return _modelMagErr;      }
    float   getApDia()            const { return _apDia;            }
    float   getRefMag()           const { return _refMag;           }
    float   getIxx()              const { return _ixx;              }
    float   getIxxErr()           const { return _ixxErr;           }
    float   getIyy()              const { return _iyy;              }
    float   getIyyErr()           const { return _iyyErr;           }
    float   getIxy()              const { return _ixy;              }
    float   getIxyErr()           const { return _ixyErr;           }
    float   getSnr()              const { return _snr;              }
    float   getChi2()             const { return _chi2;             }
    double  getValX1()            const { return _valX1;            }
    double  getValX2()            const { return _valX2;            }
    double  getValY1()            const { return _valY1;            }
    double  getValY2()            const { return _valY2;            }
    double  getValXY()            const { return _valXY;            }  
    char[3] getObsCode()          const { return _obsCode;          }
    char    isSynthetic()         const { return _isSynthetic;      }
    char    getMopsStatus()       const { return _mopsStatus;       }
    int16_t getFlag4association() const { return _flag4association; }
    int16_t getFlag4detection()   const { return _flag4detection;   }
    int16_t getFlag4wcs()         const { return _flag4wcs;         }

    // setters
    void setId              (int64_t const sourceId        ) { 
        _sourceId = sourceId;         
    }
    void setAmpExposureId   (int64_t const ampExposureId   ) { 
        _ampExposureId = ampExposureId;
    }
    void setFilterId        (int8_t  const filterId        ) { 
        _filterId = filterId;         
    }
    void setObjectId        (int64_t const objectId) {
        _objectId = objectId;
        setNotNull(OBJECT_ID);
    }
    void setMovingObjectId(int64_t const movingObjectId) {
        _movingObjectId = movingObjectId;
        setNotNull(MOVING_OBJECT_ID);
    }
    void setProcHistoryId (int32_t const procHistoryId     ) { 
        _procHistoryId = procHistoryId;    
    }    
    void setScId          (int32_t const scId              ) {
        _scId = scId;        
    }
    void setSsmId         (int64_t const ssmId             ) {
        _ssmId = ssmId;
        setNotNull(SSM_ID);
    }
    void setRa              (double  const ra              ) { 
        _ra = ra;               
    }
    void setDec             (double  const dec             ) { 
        _dec = dec;             
    }
    void setRaErr4detection (float   const raErr4detection ) { 
        _raErr4detection  = raErr4detection;  
    }
    void setDecErr4detection(float   const decErr4detection) { 
        _decErr4detection = decErr4detection; 
    }
    void setRaErr4wcs       (float const raErr4wcs         ) {
        _raErr4wcs = raErr4wcs;
        setNotNull(RA_ERR_4_WCS);
    }
    void setDecErr4wcs      (float const decErr4wcs        ) {
        _decErr4wcs = decErr4wcs;
        setNotNull(DEC_ERR_4_WCS);
    }
    void setXFlux           (double  const xFlux           ) { 
        _xFlux = xFlux;            
        setNotNull(X_FLUX);
    }
    void setXFluxErr        (double  const xFluxErr        ) { 
        _xFluxErr = xFluxErr;            
        setNotNull(X_FLUX_ERR);
    }    
    void setYFlux           (double  const yFlux           ) { 
        _yFlux = yFlux;            
        setNotNull(Y_FLUX);
    }    
    void setYFluxErr        (double  const yFluxErr        ) { 
        _yFluxErr = yFluxErr;            
        setNotNull(Y_FLUX_ERR);
    }    
    void setRaFlux          (double  const raFlux          ) { 
        _raFlux = raFlux;            
        setNotNull(RA_FLUX);
    }
    void setRaFluxErr       (double  const raFluxErr       ) { 
        _raFluxErr = raFluxErr;            
        setNotNull(RA_FLUX_ERR);
    }    
    void setDecFlux         (double  const decFlux         ) { 
        _decFlux = decFlux;            
        setNotNull(DEC_FLUX);
    }    
    void setDecFluxErr      (double  const decFluxErr      ) { 
        _decFluxErr = decFluxErr;            
        setNotNull(DEC_FLUX_ERR);
    }    
    void setXPeak           (double  const xPeak           ) { 
        _xPeak = xPeak;            
        setNotNull(X_PEAK);
    }
    void setYPeak           (double  const yPeak           ) { 
        _yPeak = yPeak;            
        setNotNull(Y_PEAK);
    }    
    void setRaPeak          (double  const raPeak          ) { 
        _raPeak = raPeak;            
        setNotNull(RA_PEAK);
    }    
    void setDecPeak         (double  const decPeak         ) { 
        _decPeak = decPeak;            
        setNotNull(DEC_PEAK);
    }    
    void setXAstrom         (double  const xAstrom         ) { 
        _xAstrom = xAstrom;            
        setNotNull(X_ASTROM);
    }
    void setXastromErr      (double  const xAstromErr      ) { 
        _xAstromErr = xAstromErr;            
        setNotNull(X_ASTROM_ERR);
    }    
    void setYAstrom         (double  const yAstrom         ) { 
        _yAstrom = yAstrom;            
        setNotNull(Y_ASTROM);
    }    
    void setYAstromErr      (double  const yAstromErr      ) { 
        _yAstromErr = yAstromErr;            
        setNotNull(Y_ASTROM_ERR);
    }    
    void setRaAstrom        (double  const raAstrom        ) { 
        _raAstrom = raAstrom;            
        setNotNull(RA_ASTROM);
    }
    void setRaAstromErr     (double  const raAstromErr     ) { 
        _raAstromErr = raAstromErr;            
        setNotNull(RA_ASTROM_ERR);
    }    
    void setDecAstrom       (double  const decAstrom       ) { 
        _decAstrom = decAstrom;            
        setNotNull(DEC_ASTROM);
    }    
    void setDecAstromErr    (double  const decAstromErr    ) { 
        _decAstromErr = decAstromErr;            
        setNotNull(DEC_ASTROM_ERR);
    }           
    void setTaiMidPoint     (double  const taiMidPoint     ) { 
        _taiMidPoint = taiMidPoint;      
    }
    void setTaiRange        (float   const taiRange        ) { 
        _taiRange = taiRange;            
    }
    void setFwhmA           (float   const fwhmA           ) {
        _fwhmA = fwhmA;
    }
    void setFwhmB           (float   const fwhmB           ) {
        _fwhmB = fwhmB;
    }
    void setFwhmTheta       (float   const fwhmTheta       ) {
        _fwhmTheta = fwhmTheta;
    }    
    void setLengthDeg       (double  const lengthDeg       ) {
        _lengthDeg = lengthDeg;
    }        
    void setFlux            (double  const flux            ) { 
        _flux = flux;             
    }
    void setFluxErr         (double  const fluxErr         ) { 
        _fluxErr = fluxErr;          
    }
    void setPsfMag          (double  const psfMag          ) { 
        _psfMag = psfMag;           
    }
    void setPsfMagErr       (float   const psfMagErr       ) { 
        _psfMagErr = psfMagErr;        
    }
    void setApMag           (double  const apMag           ) { 
        _apMag = apMag;            
    }
    void setApMagErr        (float   const apMagErr        ) { 
        _apMagErr = apMagErr;         
    }
    void setModelMag        (double  const modelMag        ) { 
        _modelMag = modelMag;         
    }
    void setModelMagErr     (float   const modelMagErr     ) { 
        _modelMagErr = modelMagErr;      
    }    
    void setApDia           (float const apDia             ) {
        _apDia = apDia;
        setNotNull(AP_DIA);
    }    
    void setRefMag          (float const refMag            ) {
        _refMag = refMag;
    }
    void setIxx             (float const ixx               ) { 
        _ixx = ixx;    
        setNotNull(IXX);
    }
    void setIxxErr          (float const ixxErr            ) { 
        _ixxErr = ixxErr; 
        setNotNull(IXX_ERR); 
    }         
    void setIyy             (float const iyy               ) { 
        _iyy = iyy;    
        setNotNull(IYY);
    }     
    void setIyyErr          (float const iyyErr            ) { 
        _iyyErr = iyyErr; 
        setNotNull(IYY_ERR);
    }         
    void setIxy             (float const ixy               ) { 
        _ixy = ixy;    
        setNotNull(IXY);
    }      
    void setIxyErr          (float const ixyErr            ) { 
        _ixyErr = ixyErr; 
        setNotNull(IXY_ERR); 
    }         
    void setSnr             (float   const snr             ) { 
        _snr = snr;
    }   
    void setChi2            (float   const chi2            ) { 
        _chi2 = chi2;
    }
    void setValX1           (double  const valX1           ) {
        _valX1 = valX1;
    }
    void setValX2           (double  const valX2           ) {
        _valX2 = valX2;
    }
    void setValY1           (double  const valY1           ) {
        _valY1 = valY1;
    }
    void setValY2           (double  const valY2           ) {
        _valY2 = valY2;
    }
    void setValXY           (double  const valXY           ) {
        _valXY = valXY;
    }         
    void setObsCode         (char[3] const obsCode         ) {
        _obsCode[0] = obsCode[0];
        _obsCode[1] = obsCode[1];
        _obsCode[3] = obsCode[3];
        setNotNull(OBS_CODE);
    }   
    void setIsSynthetic     (char    const isSynthetic     ) {
        _isSynthetic = isSynthetic;
        setNotNull(IS_SYNTHETIC);
    } 
    void setMopsStatus      (char    const mopsStatus      ) {
        _mopsStatus = mopsStatus;
        setNotNull(MOPS_STATUS);
    }
    void setFlag4association(int16_t const flag4association) {
        _flag4association = flag4association;
        setNotNull(FLAG_4_ASSOCIATION);
    }
    void setFlag4detection(int16_t const flag4detection) {
        _flag4detection = flag4detection;
        setNotNull(FLAG_4_DETECTION);
    }
    void setFlag4wcs(int16_t const flag4wcs) {
        _flag4wcs = flag4wcs;
        setNotNull(FLAG_4_WCS);
    }


    // Get/set whether or not fields are null
    bool isNull    (NullableField const f) const            { return _nulls.test(f); }
    void setNull   (NullableField const f)                  { _nulls.set(f);         }
    void setNotNull(NullableField const f)                  { _nulls.reset(f);       }
    void setNull   (NullableField const f, bool const null) { _nulls.set(f, null);   }
    void setNull   ()                                       { _nulls.set();          }
    void setNotNull()                                       { _nulls.reset();        }

    bool operator==(DiaSource const & d) const;

private :

    int64_t _diaSourceId;      // BIGINT        NOT NULL
    int64_t _ampExposureId;    // BIGINT        NOT NULL
    int64_t _objectId;         // BIGINT        NULL
    int64_t _movingObjectId;   // BIGINT        NULL
    int64_t _ssmId;            // BIGINT        NULL
    double  _ra;               // DOUBLE        NOT NULL
    double  _decl;             // DOUBLE        NOT NULL
    double  _xFlux;            // DOUBLE        NULL
    double  _xFluxErr;         // DOUBLE        NULL
    double  _yFlux;            // DOUBLE        NULL
    double  _yFluxErr;         // DOUBLE        NULL
    double  _raFlux;           // DOUBLE        NULL
    double  _raFluxErr;        // DOUBLE        NULL
    double  _declFlux;         // DOUBLE        NULL
    double  _declFluxErr;      // DOUBLE        NULL
    double  _xPeak;            // DOUBLE        NULL
    double  _yPeak;            // DOUBLE        NULL
    double  _raPeak;           // DOUBLE        NULL
    double  _decPeak;          // DOUBLE        NULL
    double  _xAstrom;          // DOUBLE        NULL
    double  _xAstromErr;       // DOUBLE        NULL
    double  _yAstrom;          // DOUBLE        NULL
    double  _yAstromErr;       // DOUBLE        NULL
    double  _raAstrom;         // DOUBLE        NULL
    double  _raAstromErr;      // DOUBLE        NULL
    double  _declAstrom;       // DOUBLE        NULL
    double  _declAstromErr;    // DOUBLE        NULL
    double  _taiMidPoint;      // DOUBLE        NOT NULL
    double  _lengthDeg;        // DOUBLE        NOT NULL
    double  _psfMag;           // DOUBLE        NOT NULL
    double  _apMag;            // DOUBLE        NOT NULL
    double  _modelMag;         // DOUBLE        NOT NULL
    double  _valX1;            // DOUBLE        NOT NULL         
    double  _valX1;            // DOUBLE        NOT NULL
    double  _valY1;            // DOUBLE        NOT NULL
    double  _valY2;            // DOUBLE        NOT NULL
    double  _valXY;            // DOUBLE        NOT NULL    
    float   _raErr4detection;  // FLOAT(0)      NOT NULL
    float   _decErr4detection; // FLOAT(0)      NOT NULL
    float   _raErr4wcs;        // FLOAT(0)      NULL
    float   _decErr4wcs;       // FLOAT(0)      NULL
    float   _taiRange;         // FLOAT(0)      NOT NULL
    float   _fwhmA;            // DECIMAL(4,2)  NOT NULL
    float   _fwhmB;            // DECIMAL(4,2)  NOT NULL
    float   _fwhmTheta;        // DECIMAL(4,1)  NOT NULL
    float   _flux;             // DECIMAL(12,2) NOT NULL
    float   _fluxErr;          // DECIMAL(10,2) NOT NULL
    float   _psfMagErr;        // DECIMAL(6,3)  NOT NULL
    float   _apMagErr;         // DECIMAL(6,3)  NOT NULL
    float   _modelMagErr;      // DECIMAL(6,3)  NULL           
    float   _apDia;            // FLOAT(0)      NULL
    float   _refMag;           // FLOAT(0)      NULL
    float   _ixx;              // FLOAT(0)      NULL
    float   _ixxErr;           // FLOAT(0)      NULL
    float   _iyy;              // FLOAT(0)      NULL
    float   _iyyErr;           // FLOAT(0)      NULL
    float   _ixy;              // FLOAT(0)      NULL
    float   _ixyErr;           // FLOAT(0)      NULL
    float   _snr;              // FLOAT(0)      NOT NULL
    float   _chi2;             // FLOAT(0)      NOT NULL
    int32_t _procHistoryId;    // INTEGER       NOT NULL
    int32_t _scId;             // INTEGER       NOT NULL    
    char[3] _obsCode;          // CHAR(3)       NULL        
    int16_t _flag4association; // SMALLINT      NULL
    int16_t _flag4detection;   // SMALLINT      NULL
    int16_t _flag4wcs;         // SMALLINT      NULL    
    int8_t  _filterId;         // TINYINT       NOT NULL
    char    _isSynthetic;      // CHAR(1)       NULL
    char    _mopsStatus;       // CHAR(1)       NULL

    std::bitset<NUM_NULLABLE_FIELDS> _nulls;

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        ar & _diaSourceId;
        ar & _ampExposureId;
        ar & _filterId;
        ar & _objectId;
        ar & _movingObjectId;
        ar & _procHistoryId;
        ar & _scId;
        ar & _ssmId;        
        ar & _ra;
        ar & _dec;
        ar & _raErr4detection;
        ar & _decErr4detection;
        ar & _raErr4wcs;
        ar & _decErr4wcs;
        ar & _xFlux;
        ar & _xFluxErr;
        ar & _yFlux;
        ar & _yFluxErr;
        ar & _raFlux;
        ar & _raFluxErr;
        ar & _decFlux;
        ar & _decFluxErr;        
        ar & _xPeak;
        ar & _yPeak;
        ar & _raPeak;
        ar & _decPeak;
        ar & _xAstrom;
        ar & _xAstromErr;
        ar & _yAstrom;
        ar & _yAstromErr;
        ar & _raAstrom;
        ar & _raAstromErr;
        ar & _decAstrom;
        ar & _decAstromErr;        
        ar & _taiMidPoint;
        ar & _taiRange;
        ar & _fwhmA;
        ar & _fwhmB;
        ar & _fwhmTheta;
        ar & _lengthDeg;        
        ar & _flux;
        ar & _fluxErr;
        ar & _psfMag;
        ar & _psfMagErr;
        ar & _apMag;
        ar & _apMagErr;
        ar & _modelMag;
        ar & _modelMagErr;
        ar & _apDia;
        ar & _refMag;
        ar & _ixx;
        ar & _ixxErr;
        ar & _iyy;
        ar & _iyyErr;
        ar & _ixy;
        ar & _ixyErr;
        ar & _snr;
        ar & _chi2;
        ar & _valX1;
        ar & _valX2;
        ar & _valY1;
        ar & _valY2;
        ar & _valXY;
        ar & _obsCode;
        ar & _isSynthetic;
        ar & _mopsStatus;
        ar & _flag4association;
        ar & _flag4detection;
        ar & _flag4wcs;

        bool b;
        if (Archive::is_loading::value) {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                ar & b;
                _nulls.set(i, b);
            }
        } else {
            for (int i = 0; i < NUM_NULLABLE_FIELDS; ++i) {
                b = _nulls.test(i);
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


// Classes that require special handling in the SWIG interface file
#ifndef SWIG

/**
 * \brief A persistable container of Source instances (implemented using std::vector).
 */
class DiaSourceVector :
    public lsst::daf::base::Persistable,
    public lsst::daf::base::Citizen
{
public :
    typedef boost::shared_ptr<DiaSourceVector> Ptr;
    typedef std::vector<DiaSource>             Vector;

    typedef Vector::allocator_type         allocator_type;
    typedef Vector::iterator               iterator;
    typedef Vector::const_iterator         const_iterator;
    typedef Vector::reverse_iterator       reverse_iterator;
    typedef Vector::const_reverse_iterator const_reverse_iterator;
    typedef Vector::size_type              size_type;
    typedef Vector::difference_type        difference_type;
    typedef Vector::reference              reference;
    typedef Vector::const_reference        const_reference;
    typedef Vector::value_type             value_type;

    DiaSourceVector();
    explicit DiaSourceVector(size_type sz);
    DiaSourceVector(size_type sz, value_type const & val);

    template <typename InputIterator>
    DiaSourceVector(InputIterator beg, InputIterator end) :
        lsst::daf::base::Citizen(typeid(*this)),
        _vec(beg, end)
    {}

    virtual ~DiaSourceVector();

    DiaSourceVector(diaSourceVector const & vec);
    explicit DiaSourceVector(Vector const & vec);
    DiaSourceVector & operator=(DiaSourceVector const & vec);
    DiaSourceVector & operator=(Vector const & vec);

    void swap(DiaSourceVector & v) { using std::swap; swap(_vec, v._vec); }
    void swap(Vector & v)          { using std::swap; swap(_vec, v);      }

    size_type size()     const { return _vec.size();     }
    size_type max_size() const { return _vec.max_size(); }
    bool      empty()    const { return _vec.empty();    }
    size_type capacity() const { return _vec.capacity(); }

    void reserve(size_type const n) { _vec.reserve(n); }

    template <typename InputIterator>
    void assign(InputIterator beg, InputIterator end)      { _vec.assign(beg, end); }
    void assign(size_type const n, value_type const & val) { _vec.assign(n, val);   }

    reference       at        (size_type const i)       { return _vec.at(i); }
    const_reference at        (size_type const i) const { return _vec.at(i); }
    reference       operator[](size_type const i)       { return _vec[i];    }
    const_reference operator[](size_type const i) const { return _vec[i];    }

    reference       front()       { return _vec.front(); }
    const_reference front() const { return _vec.front(); }
    reference       back ()       { return _vec.back();  }
    const_reference back () const { return _vec.back();  }

    iterator               begin ()       { return _vec.begin();  }
    const_iterator         begin () const { return _vec.begin();  }
    reverse_iterator       rbegin()       { return _vec.rbegin(); }
    const_reverse_iterator rbegin() const { return _vec.rbegin(); }
    iterator               end   ()       { return _vec.end();    }
    const_iterator         end   () const { return _vec.end();    }
    reverse_iterator       rend  ()       { return _vec.rend();   }
    const_reverse_iterator rend  () const { return _vec.rend();   }

    void push_back(value_type const & value) { _vec.push_back(value);  }

    void pop_back() { _vec.pop_back();  }
    void clear()    { _vec.clear();     }

    iterator insert(iterator pos, value_type const & val)              { return _vec.insert(pos, val);    }
    void     insert(iterator pos, size_type n, value_type const & val) { return _vec.insert(pos, n, val); }

    template <typename InputIterator>
    void insert(iterator pos, InputIterator beg, InputIterator end) { _vec.insert(pos, beg, end); }

    iterator erase(iterator pos)               { return _vec.erase(pos);      }
    iterator erase(iterator beg, iterator end) { return _vec.erase(beg, end); }

    void resize(size_type n)                 { _vec.resize(n);      }
    void resize(size_type n, value_type val) { _vec.resize(n, val); }

    bool operator==(DiaSourceVector const & v) { return _vec == v._vec; }
    bool operator!=(DiaSourceVector const & v) { return _vec != v._vec; }

private :

    LSST_PERSIST_FORMATTER(lsst::afw::formatters::DiaSourceVectorFormatter);

    Vector _vec;
};

#endif // SWIG


}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_DIASOURCE_H


