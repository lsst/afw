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
*/

class Source : public SourceBase {

public :

    typedef boost::shared_ptr<Source> Ptr;

    /*! An integer id for each field. */
    enum FieldId {
        //Fields added by Source
        PETRO_MAG = SourceBase::NUM_FIELDS,
        PETRO_MAG_ERR,
        SKY,
        SKY_ERR,        
        NUM_FIELDS
    };

    Source();
    

    // getters
    int64_t getSourceId() const { return getId(); }
    
    double  getPetroMag() const { return get(PETRO_MAG); }
    float   getPetroMagErr() const { return _modelMagErr;      }    
    float   getSky() const { return get(SKY); }
    float   getSkyErr() const { return get(SKY_ERR); }

    //setters
    void setSourceId(int64_t const & sourceId) {setId(sourceId); }
    
    void setPetroMag (double const petroMag) { set(PETRO_MAG, petroMag); }    
    void setPetroMagErr (float const petroMagErr) { set(PETRO_MAG_ERR, petroMagErr); }
    void setSky (float const sky) { set(SKY, sky); }
    void setSkyErr (float const skyErr) { set(SKY_ERR, skyErr); }

    bool operator==(Source const & d) const {
        return SourceBase::operator==(*static_cast<SourceBase*>(&d));
    }
private :

    friend class boost::serialization::access;
    friend class formatters::SourceVectorFormatter;
};

inline bool operator!=(Source const & d1, Source const & d2) {
    return !(d1 == d2);
}

}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_SOURCE_H

