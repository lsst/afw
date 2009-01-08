// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  The C++ representation of a Deep Detection Source.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_DETECTION_SOURCE_H
#define LSST_AFW_DETECTION_SOURCE_H

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
        class SourceVectorFormatter;
    }
namespace detection {

#ifndef SWIG
using boost::int16_t;
using boost::int32_t;
using boost::int64_t;
#endif

namespace source_detail {

/*! An integer id for each nullable field in Source. */
enum SourceNullableField {
    AMP_EXPOSURE_ID = 0,
    OBJECT_ID,
    MOVING_OBJECT_ID,
    RA_ERR_4_DETECTION,
    DEC_ERR_4_DETECTION,
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
    TAI_RANGE,
    PETRO_MAG,
    PETRO_MAG_ERR,
    NON_GRAY_CORR_MAG,
    NON_GRAY_CORR_MAG_ERR,
    ATM_CORR_MAG,        
    ATM_CORR_MAG_ERR,
    AP_DIA,
    SKY,
    SKY_ERR,        
    FLAG_4_ASSOCIATION,
    FLAG_4_DETECTION,
    FLAG_4_WCS,
    NUM_NULLABLE_FIELDS
};

} //namespace source_detail


template class BaseSourceAttributes<source_detail::NUM_NULLABLE_FIELDS>;
typedef BaseSourceAttributes<source_detail::NUM_NULLABLE_FIELDS> SourceBase;

class Source 
	: public SourceBase {
public :

    typedef boost::shared_ptr<Source> Ptr;


    Source() {setNull();}
    virtual ~Source(){};

    // getters
    int64_t getSourceId()     const { return _id; 			  }
    double getPetroMag()      const { return _petroMag;       }
    float  getPetroMagErr()   const { return _petroMagErr;    }    
    float  getSky()           const { return _sky;            }
    float  getSkyErr()        const { return _skyErr;         }

    // setters
    void setSourceId( int64_t const sourceId) {setId(sourceId);}
    
    void setPetroMag (double const petroMag) { 
        set(_petroMag, petroMag, source_detail::PETRO_MAG);         
    }
    void setPetroMagErr (float const petroMagErr) { 
        set(_petroMagErr, petroMagErr, source_detail::PETRO_MAG_ERR);    
    }
    void setSky (float const sky) { 
        set(_sky, sky, source_detail::SKY);       
    }
    void setSkyErr (float const skyErr) {
        set(_skyErr, skyErr, source_detail::SKY_ERR);
    }   
    
    //overloaded setters
    void setAmpExposureId (int64_t const ampExposureId) { 
        set(_ampExposureId, ampExposureId, source_detail::AMP_EXPOSURE_ID);
    }
    void setObjectId (int64_t const objectId) {
        set(_objectId, objectId, source_detail::OBJECT_ID);
    }
    void setMovingObjectId (int64_t const movingObjectId) {
    	set(_movingObjectId, movingObjectId, source_detail::MOVING_OBJECT_ID);
    }
    void setRaErr4detection (float const raErr4detection) { 
        set(_raErr4detection, raErr4detection, source_detail::RA_ERR_4_DETECTION);  
    }
    void setDecErr4detection(float const decErr4detection) { 
        set(_decErr4detection, decErr4detection, source_detail::DEC_ERR_4_DETECTION); 
    }
    void setXFlux (double const xFlux) { 
        set(_xFlux, xFlux, source_detail::X_FLUX);            
    }
    void setXFluxErr (double const xFluxErr) { 
        set(_xFluxErr, xFluxErr, source_detail::X_FLUX_ERR);            
    }    
    void setYFlux (double const yFlux) { 
        set(_yFlux, yFlux, source_detail::Y_FLUX);            
    }    
    void setYFluxErr (double const yFluxErr) { 
        set(_yFluxErr, yFluxErr, source_detail::Y_FLUX_ERR);            
    }    
    void setRaFlux (double const raFlux) { 
        set(_raFlux, raFlux, source_detail::RA_FLUX);            
    }
    void setRaFluxErr (double const raFluxErr) { 
        set(_raFluxErr, raFluxErr, source_detail::RA_FLUX_ERR);            
    }    
    void setDecFlux (double const decFlux) { 
        set(_decFlux, decFlux, source_detail::DEC_FLUX);
    }    
    void setDecFluxErr (double const decFluxErr) { 
        set(_decFluxErr, decFluxErr, source_detail::DEC_FLUX_ERR);            
    }    
    void setXPeak (double const xPeak) { 
        set(_xPeak, xPeak, source_detail::X_PEAK);            
    }
    void setYPeak (double const yPeak) { 
        set(_yPeak, yPeak, source_detail::Y_PEAK);            
    }    
    void setRaPeak (double const raPeak) { 
        set(_raPeak, raPeak, source_detail::RA_PEAK);            
    }    
    void setDecPeak (double const decPeak) { 
        set(_decPeak, decPeak, source_detail::DEC_PEAK);            
    }    
    void setXAstrom (double const xAstrom) { 
        set(_xAstrom, xAstrom, source_detail::X_ASTROM);            
    }
    void setXastromErr (double const xAstromErr) { 
        set(_xAstromErr, xAstromErr, source_detail::X_ASTROM_ERR);            
    }    
    void setYAstrom (double const yAstrom) { 
        set(_yAstrom, yAstrom, source_detail::Y_ASTROM);            
    }    
    void setYAstromErr (double const yAstromErr) { 
        set(_yAstromErr, yAstromErr, source_detail::Y_ASTROM_ERR);            
    }    
    void setRaAstrom (double const raAstrom) { 
        set(_raAstrom, raAstrom, source_detail::RA_ASTROM);            
    }
    void setRaAstromErr (double const raAstromErr) { 
        set(_raAstromErr, raAstromErr, source_detail::RA_ASTROM_ERR);            
    }    
    void setDecAstrom (double const decAstrom) { 
        set(_decAstrom, decAstrom, source_detail::DEC_ASTROM);            
    }    
    void setDecAstromErr (double const decAstromErr) { 
        set(_decAstromErr, decAstromErr, source_detail::DEC_ASTROM_ERR);            
    }         
    void setTaiRange (float const taiRange) { 
        set(_taiRange, taiRange, source_detail::TAI_RANGE);         
    }
    void setNonGrayCorrMag (double const nonGrayCorrMag) { 
        set(_nonGrayCorrMag, nonGrayCorrMag, source_detail::NON_GRAY_CORR_MAG);         
    }
    void setNonGrayCorrMagErr(double const nonGrayCorrMagErr) { 
        set(_nonGrayCorrMagErr, nonGrayCorrMagErr, source_detail::NON_GRAY_CORR_MAG_ERR);      
    }
    void setAtmCorrMag (double const atmCorrMag) { 
        set(_atmCorrMag, atmCorrMag, source_detail::ATM_CORR_MAG);         
    }
    void setAtmCorrMagErr (double const atmCorrMagErr) { 
        set(_atmCorrMagErr, atmCorrMagErr, source_detail::ATM_CORR_MAG_ERR);      
    }        
    void setApDia (float const apDia) {
        set(_apDia, apDia, source_detail::AP_DIA);
    }
    void setFlag4association(int16_t const flag4association) {
        set(_flag4association, flag4association, source_detail::FLAG_4_ASSOCIATION);
    }
    void setFlag4detection (int16_t const flag4detection) {
        set(_flag4detection, flag4detection, source_detail::FLAG_4_DETECTION);
    }
    void setFlag4wcs (int16_t const flag4wcs) {
        set(_flag4wcs, flag4wcs, source_detail::FLAG_4_WCS);
    }
    
    bool operator==(Source const & d) const;

private :
    double _petroMag;         // DOUBLE        NULL    
    float  _petroMagErr;      // FLOAT(0)      NULL            
    float  _sky;              // FLOAT(0)      NULL
    float  _skyErr;           // FLOAT(0)      NULL    

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        ar & _petroMag;
        ar & _petroMagErr;
        ar & _sky;
        ar & _skyErr;

		SourceBase::serialize(ar, version);
    }

    friend class boost::serialization::access;
    friend class formatters::SourceVectorFormatter;   
};

inline bool operator!=(Source const & lhs, Source const & rhs) {
	return !(lhs==rhs);
}


class PersistableSourceVector : public lsst::daf::base::Persistable {
    typedef std::vector<Source> SourceVector;
public:
    PersistableSourceVector() {}
    PersistableSourceVector(SourceVector const & sources)
        : _sources(sources) {}
        
    SourceVector & getSources() {return _sources; }
    SourceVector getSources() const {return _sources; } 
    
    void setSources(SourceVector const & sources) {_sources = sources; }
private:
    LSST_PERSIST_FORMATTER(lsst::afw::formatters::SourceVectorFormatter);
    SourceVector _sources;
}; 

}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_SOURCE_H

