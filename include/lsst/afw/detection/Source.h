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

// forward declarations for formatters

/**
    \brief Contains attributes for Deep Detection Source records.

   This class is useful
   when an unadorned data structure is required (e.g. for placement into shared memory) or
   is all that is necessary.

    The C++ fields are derived from the LSST DC3 MySQL schema, which is reproduced below:

    \code
    CREATE TABLE Source
    (
        sourceId            BIGINT      NOT NULL,
        ampExposureId       BIGINT      NULL,
        filterId            TINYINT     NOT NULL,
        objectId            BIGINT      NULL,
        movingObjectId      BIGINT      NULL,
        procHistoryId       INTEGER     NOT NULL,
        ra                  DOUBLE      NOT NULL,
        decl                DOUBLE      NOT NULL,
        raErr4wcs           FLOAT(0)    NOT NULL,
        decErr4wcs          FLOAT(0)    NOT NULL,
        raErr4detection     FLOAT(0)    NULL,
        decErr4detection    FLOAT(0)    NULL,
        xFlux               DOUBLE      NULL,
        xFluxErr            DOUBLE      NULL,
        yFlux               DOUBLE      NULL,
        yFluxErr            DOUBLE      NULL,
        raFlux              DOUBLE      NULL,
        raFluxErr           DOUBLE      NULL,
        declFlux            DOUBLE      NULL,
        declFluxErr         DOUBLE      NULL,
        xPeak               DOUBLE      NULL,
        yPeak               DOUBLE      NULL,
        raPeak              DOUBLE      NULL,
        declPeak            DOUBLE      NULL,
        xAstrom             DOUBLE      NULL,
        xAstromErr          DOUBLE      NULL,
        yAstrom             DOUBLE      NULL,
        yAstromErr          DOUBLE      NULL,
        raAstrom            DOUBLE      NULL,
        raAstromErr         DOUBLE      NULL,
        declAstrom          DOUBLE      NULL,
        declAstromErr       DOUBLE      NULL,
        taiMidPoint         DOUBLE      NOT NULL,
        taiRange            FLOAT(0)    NULL,
        fwhmA               FLOAT(0)    NOT NULL,
        fwhmB               FLOAT(0)    NOT NULL,
        fwhmTheta           FLOAT(0)    NOT NULL,
        psfMag              DOUBLE      NOT NULL,
        psfMagErr           FLOAT(0)    NOT NULL,
        apMag               DOUBLE      NOT NULL,
        apMagErr            FLOAT(0)    NOT NULL,
        modelMag            DOUBLE      NOT NULL,
        modelMagErr         FLOAT(0)    NOT NULL,
        petroMag            DOUBLE      NULL,
        petroMagErr         FLOAT(0)    NULL,
        apDia               FLOAT(0)    NULL,
        snr                 FLOAT(0)    NOT NULL,
        chi2                FLOAT(0)    NOT NULL,
        sky                 FLOAT(0)    NULL,
        skyErr              FLOAT(0)    NULL,
        flag4association    SMALLINT    NULL,
        flag4detection      SMALLINT    NULL,
        flag4wcs            SMALLINT    NULL,
        PRIMARY KEY (sourceId),
        KEY (ampExposureId),
        KEY (filterId),
        KEY (movingObjectId),
        KEY (objectId),
        KEY (procHistoryId)
    ) TYPE=MyISAM;
    \endcode
    
    Note that the C++ fields are listed in a different order than their 
    corresponding database columns: fields are sorted by type size to minimize 
    the number of padding bytes that the compiler must insert to meet field 
    alignment requirements.
*/

class Source : public BaseSourceAttributes {

public :

    typedef boost::shared_ptr<Source> Ptr;

    /*! An integer id for each nullable field. */
    enum NullableField {
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
        SKY_ERR        
        FLAG_4_ASSOCIATION,
        FLAG_4_DETECTION,
        FLAG_4_WCS,
        NUM_NULLABLE_FIELDS
    };

    Source();
    virtual ~Source();

    // getters
    double * getModelMag()      const { return _modelMag;       }
    float  * getModelMagErr()   const { return _modelMagErr;    }    
    float  * getSky()           const { return _sky;            }
    float  * getSkyErr()        const { return _skyErr;         }

    // setters
    void setPetroMag        (double  const petroMag        ) { 
        set(_petroMag, petroMag);         
    }
    void setPetroMagErr     (float   const petroMagErr     ) { 
        set(_petroMagErr, petroMagErr);    
    }
    void setSky             (float   const sky             ) { 
        set(_sky, sky);       
    }
    void setSkyErr          (float   const skyErr          ) {
        set(_skyErr, skyErr);
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
    
    bool operator==(Source const & d) const;

private :
    double * _petroMag;         // DOUBLE        NULL    
    float  * _petroMagErr;      // FLOAT(0)      NULL            
    float  * _sky;              // FLOAT(0)      NULL
    float  * _skyErr;           // FLOAT(0)      NULL    

    template <typename Archive> void serialize(Archive & ar, unsigned int const version) {
        serializeData(ar, version, _petroMag);
        serializeData(ar, version, _petroMagErr);
        serializeData(ar, version, _sky);
        serializeData(ar, version, _skyErr);

        bool b;
        
        //go through list of nullable fields,
        //store/retrieve true if field is NULL
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
    friend class formatters::SourceVectorFormatter;
};

inline bool operator!=(Source const & d1, Source const & d2) {
    return !(d1 == d2);
}


class PersistableSourceVector : public Persistable {
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

