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
class DiaSource : public SourceBase {

public :

    typedef boost::shared_ptr<DiaSource> Ptr;

    /*! An integer id for each field. */
    enum FieldId {
        //Fields added by DiaSource
        SC_ID = SourceBase::NUM_FIELDS,
        SSM_ID,
        LENGTH_DEG,
        FLUX,
        FLUX_ERR,
        REF_MAG,
        IXX,
        IXX_ERR,
        IYY,
        IYY_ERR,       
        IXY,
        IXY_ERR,
        VAL_X1,
        VAL_X2,
        VAL_Y1,
        VAL_Y2,
        VAL_XY,
        OBS_CODE,
        IS_SYNTHETIC,
        MOPS_STATUS,
        NUM_FIELDS                
    };

    DiaSource();

    // getters
    int32_t getScId() const { return get(SC_ID); }
    int64_t getSsmId() const { return get(SSM_ID); }
    double  getLengthDeg() const { return get(LENGTH_DEG);}
    float   getFlux() const { return get(FLUX);}
    float   getFluxErr() const { return get(FLUX_ERR);}
    float   getRefMag() const { return get(REF_MAG);}
    float   getIxx() const { return get(IXX);}
    float   getIxxErr() const { return get(IXX_ERR);}
    float   getIyy() const { return get(IYY);}
    float   getIyyErr() const { return get(IYY_ERR);}
    float   getIxy() const { return get(IXY); }
    float   getIxyErr() const { return get(IXY_ERR); }
    double  getValX1() const { return get(VAL_X1); }
    double  getValX2() const { return get(VAL_X2); }
    double  getValY1() const { return get(VAL_Y1); }
    double  getValY2() const { return get(VAL_Y2); }
    double  getValXY() const { return get(VAL_XY); } 
    char    getObsCode() const { return get(OBS_CODE); } 
    char    isSynthetic() const { return get(IS_SYNTHETIC); }
    char    getMopsStatus() const { return get(MOPS_STATUS); }


    // setters
    void setScId (int32_t const scId) { set(SC_ID, scId); }
    void setSsmId (int64_t const ssmId) { set(SSM_ID, ssmId); }
    void setLengthDeg (double const lengthDeg) { set(LENGTH_DEG, lengthDeg); }
    void setFlux (double  const flux) { set(FLUX, flux); }
    void setFluxErr(double  const fluxErr) { set(FLUX_ERR, fluxErr); }
    void setRefMag (float const refMag) { set(REF_MAG, refMag); }
    void setIxx (float const ixx) { set(IXX, ixx); }
    void setIxxErr (float const ixxErr) { set(IXX_ERR, ixxErr); }         
    void setIyy (float const iyy) { set(IYY, iyy); }     
    void setIyyErr (float const iyyErr) { set(IYY_ERR, iyyErr); }         
    void setIxy (float const ixy) { set(IXY, ixy); }      
    void setIxyErr(float const ixyErr) { set(IXY_ERR, ixyERR); }         
    void setValX1 (double const valX1) { set(VAL_X1, valX1); }
    void setValX2 (double const valX2) { set(VAL_X2, valX2); }
    void setValY1 (double const valY1) { set(VAL_Y1, valY1); }
    void setValY2 (double const valY2) { set(VAL_Y2, valY2); }
    void setValXY (double const valXY) { set(VAL_XY, valXY); }         
    void setObsCode (char const obsCode) { set(OBS_CODE, obsCode); } 
    void setIsSynthetic (char const isSynthetic) { set(IS_SYNTHETIC, isSynthetic); } 
    void setMopsStatus (char const mopsStatus) { set(MOPS_STATUS, mopsStatus); }

    bool operator==(DiaSource const & d) const {
        return SourceBase::operator==(*static_cast<SourceBase*>(&d));
    }

private :

    friend class boost::serialization::access;
    friend class formatters::DiaSourceVectorFormatter;
};

inline bool operator!=(DiaSource const & d1, DiaSource const & d2) {
    return !(d1 == d2);
}

}}}  // namespace lsst::afw::detection

#endif // LSST_AFW_DETECTION_DIASOURCE_H


