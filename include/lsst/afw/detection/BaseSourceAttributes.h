#ifndef LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H
#define LSST_AFW_DETECTION_BASE_SOURCE_ATTRIBUTES_H

#include <bitset>
#include "boost/cstdint.hpp"

namespace boost {
namespace serialization {
    class access;
}}

namespace lsst{
namespace afw {
namespace detection {

#ifndef SWIG
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;
#endif

template<int const numNullableFields>
class BaseSourceAttributes
{
public:
    virtual ~BaseSourceAttributes(){};
    
    // getters
    int64_t getId()               const { return _id;         }
    int64_t getAmpExposureId()    const { return _ampExposureId;    }
    int8_t  getFilterId()         const { return _filterId;         }
    int64_t getObjectId()         const { return _objectId;         }
    int64_t getMovingObjectId()   const { return _movingObjectId;   }
    int32_t getProcHistoryId()    const { return _procHistoryId;    }
    double  getRa()               const { return _ra;               }
    double  getDec()              const { return _dec;              }
    float   getRaErr4wcs()        const { return _raErr4wcs;        }
    float   getDecErr4wcs()       const { return _decErr4wcs;       }
    float   getRaErr4detection()  const { return _raErr4detection;  }
    float   getDecErr4detection() const { return _decErr4detection; }
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
    double  getPsfMag()           const { return _psfMag;           }
    float   getPsfMagErr()        const { return _psfMagErr;        }
    double  getApMag()            const { return _apMag;            }
    float   getApMagErr()         const { return _apMagErr;         }
    double  getModelMag()         const { return _modelMag;         }
    float   getModelMagErr()      const { return _modelMagErr;      }
    double  getInstMag()          const { return _instMag;          }   
    double  getInstMagErr()       const { return _instMagErr;       }   
    double  getNonGrayCorrMag()   const { return _nonGrayCorrMag;   }   
    double  getNonGrayCorrMagErr()const { return _nonGrayCorrMagErr;}   
    double  getAtmCorrMag()       const { return _atmCorrMag;       }   
    double  getAtmCorrMagErr()    const { return _atmCorrMagErr;    }           
    float   getApDia()            const { return _apDia;            }
    float   getSnr()              const { return _snr;              }
    float   getChi2()             const { return _chi2;             }    
    int16_t getFlag4association() const { return _flag4association; }
    int16_t getFlag4detection()   const { return _flag4detection;   }
    int16_t getFlag4wcs()         const { return _flag4wcs;         }
    
    
    // setters
    void setId              (int64_t const id        ) {
        set(_id, id);         
    }
    void setAmpExposureId   (int64_t const ampExposureId   ) {
        set(_ampExposureId, ampExposureId);
    }
    void setFilterId        (int8_t  const filterId        ) {
        set(_filterId, filterId);         
    }
    void setObjectId        (int64_t const objectId) {
        set(_objectId, objectId);
    }
    void setMovingObjectId (int64_t const movingObjectId) {
    	set(_movingObjectId, movingObjectId);
    }
    void setProcHistoryId (int32_t const procHistoryId   ) {
        set(_procHistoryId, procHistoryId);   
    }   
    void setRa              (double  const ra              ) {
        set(_ra, ra);               
    }
    void setDec             (double  const dec             ) {
        set(_dec, dec);             
    }
    void setRaErr4wcs       (float   const raErr4wcs       ) {
        set(_raErr4wcs, raErr4wcs);       
    }
    void setDecErr4wcs      (float   const decErr4wcs      ) {
        set(_decErr4wcs, decErr4wcs);
    }
    void setRaErr4detection (float   const raErr4detection ) {
        set(_raErr4detection, raErr4detection); 
    }
    void setDecErr4detection(float   const decErr4detection) {
        set(_decErr4detection, decErr4detection);
    }
    void setXFlux           (double  const xFlux           ) {
        set(_xFlux, xFlux);           
    }
    void setXFluxErr        (double  const xFluxErr        ) {
        set(_xFluxErr, xFluxErr);           
    }   
    void setYFlux           (double  const yFlux           ) {
        set(_yFlux, yFlux);           
    }   
    void setYFluxErr        (double  const yFluxErr        ) {
        set(_yFluxErr, yFluxErr);           
    }   
    void setRaFlux          (double  const raFlux          ) {
        set(_raFlux, raFlux);           
    }
    void setRaFluxErr       (double  const raFluxErr       ) {
        set(_raFluxErr, raFluxErr);           
    }   
    void setDecFlux         (double  const decFlux         ) {
        set(_decFlux, decFlux);           
    }   
    void setDecFluxErr      (double  const decFluxErr      ) {
        set(_decFluxErr, decFluxErr);           
    }   
    void setXPeak           (double  const xPeak           ) {
        set(_xPeak, xPeak);           
    }
    void setYPeak           (double  const yPeak           ) {
        set(_yPeak, yPeak);           
    }   
    void setRaPeak          (double  const raPeak          ) {
        set(_raPeak, raPeak);           
    }   
    void setDecPeak         (double  const decPeak         ) {
        set(_decPeak, decPeak);           
    }   
    void setXAstrom         (double  const xAstrom         ) {
        set(_xAstrom, xAstrom);           
    }
    void setXastromErr      (double  const xAstromErr      ) {
        set(_xAstromErr, xAstromErr);           
    }   
    void setYAstrom         (double  const yAstrom         ) {
        set(_yAstrom, yAstrom);           
    }   
    void setYAstromErr      (double  const yAstromErr      ) {
        set(_yAstromErr, yAstromErr);           
    }   
    void setRaAstrom        (double  const raAstrom        ) {
        set(_raAstrom, raAstrom);           
    }
    void setRaAstromErr     (double  const raAstromErr     ) {
        set(_raAstromErr, raAstromErr);           
    }   
    void setDecAstrom       (double  const decAstrom       ) {
        set(_decAstrom, decAstrom);           
    }   
    void setDecAstromErr    (double  const decAstromErr    ) {
        set(_decAstromErr, decAstromErr);           
    }         
    void setTaiMidPoint     (double  const taiMidPoint     ) {
        set(_taiMidPoint, taiMidPoint);     
    }
    void setTaiRange        (float   const taiRange        ) {
        set(_taiRange, taiRange);         
    }
    void setFwhmA           (float   const fwhmA           ) {
        set(_fwhmA, fwhmA);
    }
    void setFwhmB           (float   const fwhmB           ) {
        set(_fwhmB, fwhmB);
    }
    void setFwhmTheta       (float   const fwhmTheta       ) {
        set(_fwhmTheta, fwhmTheta);
    }
    void setPsfMag          (double  const psfMag          ) {
        set(_psfMag, psfMag);           
    }
    void setPsfMagErr       (float   const psfMagErr       ) {
        set(_psfMagErr, psfMagErr);       
    }
    void setApMag           (double  const apMag           ) {
        set(_apMag, apMag);           
    }
    void setApMagErr        (float   const apMagErr        ) {
        set(_apMagErr, apMagErr);         
    }
    void setModelMag        (double  const modelMag        ) {
        set(_modelMag, modelMag);         
    }
    void setModelMagErr     (float   const modelMagErr     ) {
        set(_modelMagErr, modelMagErr);     
    }
    void setInstMag         (double  const instMag         ) {
        set(_instMag, instMag);         
    }
    void setInstMagErr      (double  const instMagErr      ) {
        set(_instMagErr, instMagErr);     
    }
    void setNonGrayCorrMag  (double  const nonGrayCorrMag  ) {
        set(_nonGrayCorrMag, nonGrayCorrMag);         
    }
    void setNonGrayCorrMagErr(double  const nonGrayCorrMagErr) {
        set(_nonGrayCorrMagErr, nonGrayCorrMagErr);     
    }
    void setAtmCorrMag       (double  const atmCorrMag         ) {
        set(_atmCorrMag, atmCorrMag);         
    }
    void setAtmCorrMagErr      (double  const atmCorrMagErr      ) {
        set(_atmCorrMagErr, atmCorrMagErr);     
    }       
    void setApDia           (float   const apDia           ) {
        set(_apDia, apDia);
    }
    void setSnr             (float   const snr             ) {
        set(_snr, snr);             
    }
    void setChi2            (float   const chi2            ) {
        set(_chi2, chi2);             
    }   
    void setFlag4association(int16_t const flag4association) {
        set(_flag4association, flag4association);
    }
    void setFlag4detection  (int16_t const flag4detection  ) {
        set(_flag4detection, flag4detection);
    }
    void setFlag4wcs        (int16_t const flag4wcs        ) {
        set(_flag4wcs, flag4wcs);
    }
    
protected:
    
    std::bitset<numNullableFields> _nulls;
    
    template<typename T> 
    inline bool areEqual(T const & a, T const & b, int const field = -1) const {
    
    	bool null = isNull(field);
    	
        return ( a == b || null);
    }
    
    template<typename T>
    inline void set(T &dest, T const & src, int const field = -1) {
		setNotNull(field);			
        dest = src;
    }
    
    inline bool isNull(int const field) const { 
    	if(field >= 0 && field < numNullableFields)    	
	    	return _nulls.test(field); 
	    else return false;
	}

    inline void setNotNull(int const field) {
		if(field >= 0 && field < numNullableFields)
			_nulls.reset(field);
    }
    
    inline void setNull(int const field) {
		if(field >= 0 && field < numNullableFields)
			_nulls.set(field);
    }
    
    inline void setNull (int const field, bool const null) { 
		if(field >= 0 && field < numNullableFields)
		  	_nulls.set(field, null);   
	}
    inline void setNull() { _nulls.set();}
    inline void setNotNull() {_nulls.reset();}    

    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version);            
    
    int64_t _id;               // BIGINT        NOT NULL
    int64_t _ampExposureId;    // BIGINT        NULL
    int64_t _objectId;         // BIGINT        NULL
    int64_t _movingObjectId;   // BIGINT        NULL
    double  _ra;               // DOUBLE        NOT NULL
    double  _dec;              // DOUBLE        NOT NULL
    double  _xFlux;            // DOUBLE        NULL
    double  _xFluxErr;         // DOUBLE        NULL
    double  _yFlux;            // DOUBLE        NULL
    double  _yFluxErr;         // DOUBLE        NULL
    double  _raFlux;           // DOUBLE        NULL
    double  _raFluxErr;        // DOUBLE        NULL
    double  _decFlux;         // DOUBLE        NULL
    double  _decFluxErr;      // DOUBLE        NULL
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
    double  _decAstrom;       // DOUBLE        NULL
    double  _decAstromErr;    // DOUBLE        NULL
    double  _taiMidPoint;      // DOUBLE        NOT NULL
    double  _psfMag;           // DOUBLE        NOT NULL
    double  _apMag;            // DOUBLE        NOT NULL
    double  _modelMag;         // DOUBLE        NOT NULL
    double  _instMag;     	 // DOUBLE	     
    double  _nonGrayCorrMag;   // DOUBLE
    double  _atmCorrMag;       // DOUBLE
    double  _instMagErr;       // DOUBLE
    double  _nonGrayCorrMagErr;// DOUBLE
    double  _atmCorrMagErr;	 // DOUBLE
    float   _raErr4detection;  // FLOAT
    float   _decErr4detection; // FLOAT
    float   _raErr4wcs;        // FLOAT
    float   _decErr4wcs;       // FLOAT
    float   _taiRange;         // FLOAT
    float   _fwhmA;            // FLOAT
    float   _fwhmB;            // FLOAT
    float   _fwhmTheta;        // FLOAT
    float   _psfMagErr;        // FLOAT
    float   _apMagErr;         // FLOAT
    float   _modelMagErr;	   // FLOAT
    float   _apDia;            // FLOAT(0)      NULL
    float   _snr;              // FLOAT(0)      NOT NULL
    float   _chi2;             // FLOAT(0)      NOT NULL
    int32_t _procHistoryId;    // INTEGER       NOT NULL    
    int16_t _flag4association; // SMALLINT      NULL
    int16_t _flag4detection;   // SMALLINT      NULL
    int16_t _flag4wcs;         // SMALLINT      NULL    
    int8_t  _filterId;         // TINYINT       NOT NULL
    
    
    friend class boost::serialization::access;
};


}}} //namespace lsst::afw::detection

#endif
