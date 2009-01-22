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


/*! An integer id for each nullable field in Source. */
enum SourceNullableField {
    AMP_EXPOSURE_ID = NUM_SHARED_NULLABLE_FIELDS,
    RA_ERR_4_DETECTION,
    DEC_ERR_4_DETECTION,
    TAI_RANGE,
    PETRO_MAG,
    PETRO_MAG_ERR,
    SKY,
    SKY_ERR,        
    NUM_SOURCE_NULLABLE_FIELDS
};

class Source 
	: public BaseSourceAttributes< NUM_SOURCE_NULLABLE_FIELDS> {
public :

    typedef boost::shared_ptr<Source> Ptr;


    Source();
    Source(Source const & other);
    
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
        set(_petroMag, petroMag,  PETRO_MAG);         
    }
    void setPetroMagErr (float const petroMagErr) { 
        set(_petroMagErr, petroMagErr,  PETRO_MAG_ERR);    
    }
    void setSky (float const sky) { 
        set(_sky, sky,  SKY);       
    }
    void setSkyErr (float const skyErr) {
        set(_skyErr, skyErr,  SKY_ERR);
    }   
    
    //overloaded setters
    void setAmpExposureId (int64_t const ampExposureId) { 
        set(_ampExposureId, ampExposureId,  AMP_EXPOSURE_ID);
    }
    void setRaErr4detection (float const raErr4detection) { 
        set(_raErr4detection, raErr4detection,  RA_ERR_4_DETECTION);  
    }
    void setDecErr4detection(float const decErr4detection) { 
        set(_decErr4detection, decErr4detection,  DEC_ERR_4_DETECTION); 
    }
    void setTaiRange (float const taiRange) { 
        set(_taiRange, taiRange,  TAI_RANGE);         
    }

    
    bool operator==(Source const & d) const;

private :
    double _petroMag;         // DOUBLE        NULL    
    float  _petroMagErr;      // FLOAT(0)      NULL            
    float  _sky;              // FLOAT(0)      NULL
    float  _skyErr;           // FLOAT(0)      NULL    

    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version) {
        ar & _petroMag;
        ar & _petroMagErr;
        ar & _sky;
        ar & _skyErr;

		BaseSourceAttributes< NUM_SOURCE_NULLABLE_FIELDS>::serialize(ar, version);
    }

    friend class boost::serialization::access;
    friend class formatters::SourceVectorFormatter;   
};

inline bool operator!=(Source const & lhs, Source const & rhs) {
	return !(lhs==rhs);
}

 
typedef std::vector<Source> SourceVector;

class PersistableSourceVector : public lsst::daf::base::Persistable {
public:
	typedef boost::shared_ptr<PersistableSourceVector> Ptr;
    PersistableSourceVector() {}
    PersistableSourceVector(SourceVector const & sources)
        : _sources(sources) {}
        
    SourceVector getSources() const {return _sources; }
    void setSources(SourceVector const & sources) {_sources = sources; }
    
    bool operator==(SourceVector const & other) const {
        if(_sources.size() != other.size())
        	return false;
        	        
    	SourceVector::size_type i;
        for(i =0; i<_sources.size();i++) {
			if(_sources[i] != other[i])
				return false;			
    	}
    	
    	return true;
    }
    
    bool operator==(PersistableSourceVector const & other) const {
    	return other==_sources;
    }
private:

    LSST_PERSIST_FORMATTER(lsst::afw::formatters::SourceVectorFormatter);
    SourceVector _sources;
}; 


}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_SOURCE_H

