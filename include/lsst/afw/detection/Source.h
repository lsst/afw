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

/*! An integer id for each nullable field in Source. */
enum SourceNullableField {
    AMP_EXPOSURE_ID = NUM_SHARED_NULLABLE_FIELDS,
    RA_ERR_FOR_DETECTION,
    DEC_ERR_FOR_DETECTION,
    TAI_RANGE,
    PETRO_MAG,
    PETRO_MAG_ERR,
    SKY,
    SKY_ERR,        
    NUM_SOURCE_NULLABLE_FIELDS
};


/**
 * In-code representation of an entry in the Source catalog for
 *   persisting/retrieving Sources
 */
class Source 
    : public BaseSourceAttributes< NUM_SOURCE_NULLABLE_FIELDS> {
public :
    typedef boost::shared_ptr<Source> Ptr;

    Source();
    Source(Source const & other);  
    virtual ~Source(){};

    // getters
    boost::int64_t getSourceId() const { return _id; }
    double getPetroMag() const { return _petroMag; }
    float  getPetroMagErr() const { return _petroMagErr; }    
    float  getSky() const { return _sky; }
    float  getSkyErr() const { return _skyErr; }

    // setters
    void setSourceId( boost::int64_t const sourceId) {setId(sourceId);}
    
    void setPetroMag(double const petroMag) { 
        set(_petroMag, petroMag, PETRO_MAG);         
    }
    void setPetroMagErr(float const petroMagErr) { 
        set(_petroMagErr, petroMagErr, PETRO_MAG_ERR);    
    }
    void setSky(float const sky) { 
        set(_sky, sky, SKY);       
    }
    void setSkyErr (float const skyErr) {
        set(_skyErr, skyErr, SKY_ERR);
    }   
    
    //overloaded setters
    //Because these fields are not NULLABLE in all sources, 
    //  special behavior must be defined in the derived class
    void setAmpExposureId (boost::int64_t const ampExposureId) { 
        set(_ampExposureId, ampExposureId, AMP_EXPOSURE_ID);
    }
    void setRaErrForDetection (float const raErrForDetection) { 
        set(_raErrForDetection, raErrForDetection, RA_ERR_FOR_DETECTION);  
    }
    void setDecErrForDetection(float const decErrForDetection) { 
        set(_decErrForDetection, decErrForDetection, DEC_ERR_FOR_DETECTION); 
    }
    void setTaiRange (float const taiRange) { 
        set(_taiRange, taiRange, TAI_RANGE);         
    }

    
    bool operator==(Source const & d) const;

private :
    double _petroMag;  
    float  _petroMagErr;
    float  _sky;
    float  _skyErr;

    template <typename Archive> 
    void serialize(Archive & ar, unsigned int const version) {
        ar & _petroMag;
        ar & _petroMagErr;
        ar & _sky;
        ar & _skyErr;

        BaseSourceAttributes<NUM_SOURCE_NULLABLE_FIELDS>::serialize(ar, version);
    }

    friend class boost::serialization::access;
    friend class formatters::SourceVectorFormatter;   
};

inline bool operator!=(Source const & lhs, Source const & rhs) {
    return !(lhs==rhs);
}


typedef std::vector<Source::Ptr> SourceVector;
typedef boost::shared_ptr<SourceVector> SourceVectorPtr;
 
class PersistableSourceVector : public lsst::daf::base::Persistable {
public:
    typedef boost::shared_ptr<PersistableSourceVector> Ptr;
    PersistableSourceVector() {}
    PersistableSourceVector(SourceVector const & sources)
        : _sources(sources) {}
    ~PersistableSourceVector(){_sources.clear();}
        
    SourceVector getSources() const {return _sources; }
    void setSources(SourceVector const & sources) {_sources = sources; }
    
    bool operator==(SourceVector const & other) const {
        if (_sources.size() != other.size())
            return false;
                    
        SourceVector::size_type i;
        for (i = 0; i < _sources.size(); ++i) {
            if (*_sources[i] != *other[i])
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

