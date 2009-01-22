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

enum SharedNullableField {
    OBJECT_ID,
    MOVING_OBJECT_ID,
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
    NON_GRAY_CORR_MAG,
    NON_GRAY_CORR_MAG_ERR,
    ATM_CORR_MAG,        
    ATM_CORR_MAG_ERR,
    AP_DIA,       
    FLAG_4_ASSOCIATION,
    FLAG_4_DETECTION,
    FLAG_4_WCS,
    NUM_SHARED_NULLABLE_FIELDS
};

template<int numNullableFields>
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
        set(_objectId, objectId,  OBJECT_ID);
    }
    void setMovingObjectId (int64_t const movingObjectId) {
    	set(_movingObjectId, movingObjectId,  MOVING_OBJECT_ID);
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
    void setXFlux (double const xFlux) { 
        set(_xFlux, xFlux,  X_FLUX);            
    }
    void setXFluxErr (double const xFluxErr) { 
        set(_xFluxErr, xFluxErr,  X_FLUX_ERR);            
    }    
    void setYFlux (double const yFlux) { 
        set(_yFlux, yFlux,  Y_FLUX);            
    }    
    void setYFluxErr (double const yFluxErr) { 
        set(_yFluxErr, yFluxErr,  Y_FLUX_ERR);            
    }    
    void setRaFlux (double const raFlux) { 
        set(_raFlux, raFlux,  RA_FLUX);            
    }
    void setRaFluxErr (double const raFluxErr) { 
        set(_raFluxErr, raFluxErr,  RA_FLUX_ERR);            
    }    
    void setDecFlux (double const decFlux) { 
        set(_decFlux, decFlux,  DEC_FLUX);
    }    
    void setDecFluxErr (double const decFluxErr) { 
        set(_decFluxErr, decFluxErr,  DEC_FLUX_ERR);            
    }    
    void setXPeak (double const xPeak) { 
        set(_xPeak, xPeak,  X_PEAK);            
    }
    void setYPeak (double const yPeak) { 
        set(_yPeak, yPeak,  Y_PEAK);            
    }    
    void setRaPeak (double const raPeak) { 
        set(_raPeak, raPeak,  RA_PEAK);            
    }    
    void setDecPeak (double const decPeak) { 
        set(_decPeak, decPeak,  DEC_PEAK);            
    }    
    void setXAstrom (double const xAstrom) { 
        set(_xAstrom, xAstrom,  X_ASTROM);            
    }
    void setXAstromErr (double const xAstromErr) { 
        set(_xAstromErr, xAstromErr,  X_ASTROM_ERR);            
    }    
    void setYAstrom (double const yAstrom) { 
        set(_yAstrom, yAstrom,  Y_ASTROM);            
    }    
    void setYAstromErr (double const yAstromErr) { 
        set(_yAstromErr, yAstromErr,  Y_ASTROM_ERR);            
    }    
    void setRaAstrom (double const raAstrom) { 
        set(_raAstrom, raAstrom,  RA_ASTROM);            
    }
    void setRaAstromErr (double const raAstromErr) { 
        set(_raAstromErr, raAstromErr,  RA_ASTROM_ERR);            
    }    
    void setDecAstrom (double const decAstrom) { 
        set(_decAstrom, decAstrom,  DEC_ASTROM);            
    }    
    void setDecAstromErr (double const decAstromErr) { 
        set(_decAstromErr, decAstromErr,  DEC_ASTROM_ERR);            
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
    void setNonGrayCorrMag (double const nonGrayCorrMag) { 
        set(_nonGrayCorrMag, nonGrayCorrMag,  NON_GRAY_CORR_MAG);         
    }
    void setNonGrayCorrMagErr(double const nonGrayCorrMagErr) { 
        set(_nonGrayCorrMagErr, nonGrayCorrMagErr,  NON_GRAY_CORR_MAG_ERR);      
    }
    void setAtmCorrMag (double const atmCorrMag) { 
        set(_atmCorrMag, atmCorrMag,  ATM_CORR_MAG);         
    }
    void setAtmCorrMagErr (double const atmCorrMagErr) { 
        set(_atmCorrMagErr, atmCorrMagErr,  ATM_CORR_MAG_ERR);      
    }     
    void setApDia (float const apDia) {
        set(_apDia, apDia,  AP_DIA);
    }
    void setSnr             (float   const snr             ) {
        set(_snr, snr);             
    }
    void setChi2            (float   const chi2            ) {
        set(_chi2, chi2);             
    }   
    void setFlag4association(int16_t const flag4association) {
        set(_flag4association, flag4association,  FLAG_4_ASSOCIATION);
    }
    void setFlag4detection (int16_t const flag4detection) {
        set(_flag4detection, flag4detection,  FLAG_4_DETECTION);
    }
    void setFlag4wcs (int16_t const flag4wcs) {
        set(_flag4wcs, flag4wcs,  FLAG_4_WCS);
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
protected:
    
    std::bitset<numNullableFields> _nulls;
    
	BaseSourceAttributes(): 
		_id(0), _ampExposureId(0), _filterId(0),
        _objectId(0), _movingObjectId(0), _procHistoryId(0),
        _ra(0.0), _dec(0.0), 
        _raErr4detection(0.0), _decErr4detection(0.0),
        _raErr4wcs(0.0), _decErr4wcs(0.0),
        _xFlux(0.0), _xFluxErr(0.0),
        _yFlux(0.0), _yFluxErr(0.0),
        _raFlux(0.0),_raFluxErr(0.0),
        _decFlux(0.0), _decFluxErr(0.0),
        _xPeak(0.0), _yPeak(0.0), _raPeak(0.0), _decPeak(0.0),
        _xAstrom(0.0), _xAstromErr(0.0),
        _yAstrom(0.0), _yAstromErr(0.0),
        _raAstrom(0.0), _raAstromErr(0.0),
        _decAstrom(0.0), _decAstromErr(0.0),
        _taiMidPoint(0.0), _taiRange(0.0),
        _fwhmA(0.0), _fwhmB(0.0), _fwhmTheta(0.0),
        _psfMag(0.0), _psfMagErr(0.0),
        _apMag(0.0), _apMagErr(0.0),
        _modelMag(0.0), _modelMagErr(0.0),
        _instMag(0.0), _instMagErr(0.0),
        _nonGrayCorrMag(0.0), _nonGrayCorrMagErr(0.0),
        _atmCorrMag(0.0), _atmCorrMagErr(0.0),
        _apDia(0.0), 
        _snr(0.0), _chi2(0.0),
		_flag4association(0), _flag4detection(0), _flag4wcs(0)
    {
    	setNull();
    }
	
	/*
    BaseSourceAttributes(BaseSourceAttributes const & other) :
     	_id(other._id),
        _ampExposureId(other._ampExposureId), 
        _filterId(other._filterId),
        _objectId(other._objectId), 
        _movingObjectId(other._movingObjectId), 
        _procHistoryId(other._procHistoryId),
        _ra(other._ra), 
        _dec(other._dec), 
        _raErr4detection(other._raErr4detection), 
        _decErr4detection(other._decErr4detection),
        _raErr4wcs(other._raErr4wcs), 
        _decErr4wcs(other._decErr4wcs),
        _xFlux(other._xFlux), 
        _xFluxErr(other._xFluxErr),
        _yFlux(other._yFlux), 
        _yFluxErr(other._yFluxErr),
        _raFlux(other._raFlux),
        _raFluxErr(other._raFluxErr),
        _decFlux(other._decFlux), 
        _decFluxErr(other._decFluxErr),
        _xPeak(other._xPeak), 
        _yPeak(other._yPeak), 
        _raPeak(other._raPeak), 
        _decPeak(other._decPeak),
        _xAstrom(other._xAstrom), 
        _xAstromErr(other._xAstromErr),
        _yAstrom(other._yAstrom), 
        _yAstromErr(other._yAstromErr),
        _raAstrom(other._raAstrom), 
        _raAstromErr(other._raAstromErr),
        _decAstrom(other._decAstrom), 
        _decAstromErr(other._decAstromErr),
        _taiMidPoint(other._taiMidPoint), 
        _taiRange(other._taiRange),
        _fwhmA(other._fwhmA), 
        _fwhmB(other._fwhmB), 
        _fwhmTheta(other._fwhmTheta),
        _psfMag(other._psfMag), 
        _psfMagErr(other._psfMagErr),
        _apMag(other._apMag), 
        _apMagErr(other._apMagErr),
        _modelMag(other._modelMag), 
        _modelMagErr(other._modelMagErr),
        _instMag(other._instMag), 
        _instMagErr(other._instMagErr),
        _nonGrayCorrMag(other._nonGrayCorrMag), 
        _nonGrayCorrMagErr(other._nonGrayCorrMagErr),
        _atmCorrMag(other._atmCorrMag), 
        _atmCorrMagErr(other._atmCorrMagErr),
        _apDia(other._apDia), 
        _snr(other._snr), 
        _chi2(other._chi2),
        _flag4association(other._flag4association), 
        _flag4detection(other._flag4detection), 
        _flag4wcs(other._flag4wcs)
    {}	
	*/
	
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
    
 

    template <class Archive> 
    void serialize(Archive & ar, unsigned int const version) {
        ar & _id;
        ar & _ampExposureId;
        ar & _filterId;
        ar & _objectId;
        ar & _movingObjectId;
        ar & _procHistoryId;
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
        ar & _psfMag;
        ar & _psfMagErr;
        ar & _apMag;
        ar & _apMagErr;
        ar & _modelMag;
        ar & _modelMagErr;
        ar & _instMag;
        ar & _instMagErr;
        ar & _nonGrayCorrMag;
        ar & _nonGrayCorrMagErr;
        ar & _atmCorrMag;
        ar & _atmCorrMagErr;
        ar & _apDia;        
        ar & _snr;
        ar & _chi2;
        ar & _flag4association;
        ar & _flag4detection;
        ar & _flag4wcs;
        
        bool b;
        if (Archive::is_loading::value) {
            for (int i = 0; i < numNullableFields; ++i) {
                ar & b;
                _nulls.set(i, b);
            }
        } else {
            for (int i = 0; i < numNullableFields; ++i) {
                b = isNull(i);
                ar & b;
            }
        }    
    }
            
    
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
