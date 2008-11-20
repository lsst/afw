// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for DiaSource instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/any.hpp"
#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/DiaSourceFormatters.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/DiaSource.h"

namespace ex = lsst::pex::exceptions;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::DiaSource;
using lsst::afw::detection::DiaSourceVector;

namespace form = lsst::afw::formatters;

// -- DiaSourceVectorFormatter ----------------

form::DiaSourceVectorFormatter::DiaSourceVectorFormatter(Policy::Ptr const & policy) 
    : lsst::daf::persistence::Formatter(typeid(*this)), _policy(policy) {}
    
DiaSourceVectorFormatter::~DiaSourceVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::DiaSourceVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new DiaSourceVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::DiaSourceVectorFormatter::registration(
    "DiaSourceVector",
    typeid(DiaSourceVector),
    createInstance
);


/*!
    \internal   Generates a unique identifier for a DiaSource given the id of the
                originating visit, the id of the originating ccd, and the sequence
                number of the DiaSource within that slice.
 */
inline static int64_t generateDiaSourceId(unsigned short seqNum, int ccdId, int64_t visitId) {
    return (visitId << 24) + (ccdId << 16) + seqNum;
}


/*!
    Inserts a single DiaSource into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::DiaSourceVectorFormatter::insertRow(T & db, DiaSource const & d) {

    db.template setColumn<int64_t>("diaSourceId",      d._diaSourceId);
    db.template setColumn<int64_t>("ampExposureId",    d._ampExposureId);
    db.template setColumn<int64_t>("ssmId",            d.ssmId);

    db.template setColumn<double> ("ra",               d._ra);
    db.template setColumn<double> ("decl",             d._dec);
    db.template setColumn<double> ("taiMidPoint",      d._taiMidPoint);
    db.template setColumn<double> ("lengthDeg",        d._lengthDeg);        
    db.template setColumn<double> ("psfMag",           d._psfMag);
    db.template setColumn<double> ("apMag",            d._apMag);
    db.template setColumn<double> ("modelMag",         d._modelMag);
    db.template setColumn<double> ("valx1",            d._valX1);
    db.template setColumn<double> ("valx2",            d._valX2);
    db.template setColumn<double> ("valy1",            d._valY1);
    db.template setColumn<double> ("valy2",            d._valY2);
    db.template setColumn<double> ("valxy",            d._valXY);                
    
    db.template setColumn<float>  ("raErr4detection",  d._raErr4detection);
    db.template setColumn<float>  ("decErr4detection", d._decErr4detection);
    db.template setColumn<float>  ("taiRange",         d._taiRange);
    db.template setColumn<float>  ("fwhmA",            d._fwhmA);
    db.template setColumn<float>  ("fwhmB",            d._fwhmB);
    db.template setColumn<float>  ("fwhmTheta",        d._fwhmTheta);
    db.template setColumn<float>  ("flux",             d._flux);
    db.template setColumn<float>  ("fluxErr",          d._fluxErr);
    db.template setColumn<float>  ("psfMagErr",        d._psfMagErr);
    db.template setColumn<float>  ("apMagErr",         d._apMagErr);
    db.template setColumn<float>  ("snr",              d._snr);
    db.template setColumn<float>  ("chi2",             d._chi2);
    
    db.template setColumn<int32_t>("procHistoryId",    d._procHistoryId);
    db.template setColumn<int32_t>("scId",             d._scId);
    db.template setColumn<char>   ("filterId",         static_cast<char>(d._filterId));


    if (!d.isNull(DiaSource::OBJECT_ID)) {
        db.template setColumn<int64_t>("objectId", d._objectId);
    } else {
        db.setColumnToNull("objectId");
    }
    if (!d.isNull(DiaSource::MOVING_OBJECT_ID)) {
        db.template setColumn<int64_t>("movingObjectId", d._movingObjectId);
    } else {
        db.setColumnToNull("movingObjectId");
    }
    if (!d.isNull(DiaSource::RA_ERR_4_WCS)) {
        db.template setColumn<float>("raErr4wcs", d._raErr4wcs);
    } else {
        db.setColumnToNull("raErr4wcs");
    }
    if (!d.isNull(DiaSource::DEC_ERR_4_WCS)) {
        db.template setColumn<float>("decErr4wcs", d._decErr4wcs);
    } else {
        db.setColumnToNull("decErr4wcs");
    }
    if (!d.isNull(DiaSource::X_FLUX)) {
        db.template setColumn<double>("xFlux", d._xFlux);
    } else {
        db.setColumnToNull("xFlux");
    }    
    if (!d.isNull(DiaSource::X_FLUX_ERR)) {
        db.template setColumn<double>("xFluxErr", d._xFluxErr);
    } else {
        db.setColumnToNull("xFluxErr");
    }        
    if (!d.isNull(DiaSource::Y_FLUX)) {
        db.template setColumn<double>("yFlux", d._yFlux);
    } else {
        db.setColumnToNull("yFlux");
    }    
    if (!d.isNull(DiaSource::Y_FLUX_ERR)) {
        db.template setColumn<double>("yFluxErr", d._yFluxErr);
    } else {
        db.setColumnToNull("yFluxErr");
    } 
    if (!d.isNull(DiaSource::RA_FLUX)) {
        db.template setColumn<double>("xFlux", d._raFlux);
    } else {
        db.setColumnToNull("raFlux");
    }    
    if (!d.isNull(DiaSource::RA_FLUX_ERR)) {
        db.template setColumn<double>("raFluxErr", d._raFluxErr);
    } else {
        db.setColumnToNull("raFluxErr");
    }        
    if (!d.isNull(DiaSource::DEC_FLUX)) {
        db.template setColumn<double>("decFlux", d._decFlux);
    } else {
        db.setColumnToNull("decFlux");
    }    
    if (!d.isNull(DiaSource::DEC_FLUX_ERR)) {
        db.template setColumn<double>("decFluxErr", d._decFluxErr);
    } else {
        db.setColumnToNull("decFluxErr");
    }                     
    if (!d.isNull(DiaSource::X_PEAK)) {
        db.template setColumn<double>("xPeak", d._xPeak);
    } else {
        db.setColumnToNull("xPeak");
    }        
    if (!d.isNull(DiaSource::Y_PEAK)) {
        db.template setColumn<double>("yPeak", d._yPeak);
    } else {
        db.setColumnToNull("yPeak");
    } 
    if (!d.isNull(DiaSource::RA_PEAK)) {
        db.template setColumn<double>("raPeak", d._raPeak);
    } else {
        db.setColumnToNull("raPeak");
    }     
    if (!d.isNull(DiaSource::DEC_PEAK)) {
        db.template setColumn<double>("decPeak", d._decPeak);
    } else {
        db.setColumnToNull("decPeak");
    }     
    if (!d.isNull(DiaSource::X_ASTROM)) {
        db.template setColumn<double>("xAstrom", d._xAstrom);
    } else {
        db.setColumnToNull("xAstrom");
    }    
    if (!d.isNull(Source::X_ASTROM_ERR)) {
        db.template setColumn<double>("xAstromErr", d._xAstromErr);
    } else {
        db.setColumnToNull("xAstromErr");
    }        
    if (!d.isNull(DiaSource::Y_ASTROM)) {
        db.template setColumn<double>("yAstrom", d._yAstrom);
    } else {
        db.setColumnToNull("yAstrom");
    }    
    if (!d.isNull(DiaSource::Y_ASTROM_ERR)) {
        db.template setColumn<double>("yAstromErr", d._yAstromErr);
    } else {
        db.setColumnToNull("yAstromErr");
    } 
    if (!d.isNull(DiaSource::RA_ASTROM)) {
        db.template setColumn<double>("xAstrom", d._raAstrom);
    } else {
        db.setColumnToNull("raAstrom");
    }    
    if (!d.isNull(DiaSource::RA_ASTROM_ERR)) {
        db.template setColumn<double>("raAstromErr", d._raAstromErr);
    } else {
        db.setColumnToNull("raAstromErr");
    }        
    if (!d.isNull(DiaSource::DEC_ASTROM)) {
        db.template setColumn<double>("decAstrom", d._decAstrom);
    } else {
        db.setColumnToNull("decAstrom");
    }    
    if (!d.isNull(DiaSource::DEC_ASTROM_ERR)) {
        db.template setColumn<double>("decAstromErr", d._decAstromErr);
    } else {
        db.setColumnToNull("decAstromErr");
    }    
    if (!d.isNull(DiaSource::MODEL_MAG_ERR)) {
        db.template setColumn<float>("modelMagErr", d._modelMagErr);
    } else {
        db.setColumnToNull("modelMagErr");
    }              
    if (!d.isNull(DiaSource::AP_DIA)) {
        db.template setColumn<float>("apDia", d._apDia);
    } else {
        db.setColumnToNull("apDia");
    }
    if (!d.isNull(DiaSource::REF_MAG)) {
        db.template setColumn<float>("refMag", d._refMag);
    } else {
        db.setColumnToNull("refMag");
    }        
    if (!d.isNull(DiaSource::IXX)) {
        db.template setColumn<float>("Ixx", d._ixx);
    } else {
        db.setColumnToNull("Ixx");
    }
    if (!d.isNull(DiaSource::IXX_ERR)) {
        db.template setColumn<float>("IxxErr", d._ixxErr);
    } else {
        db.setColumnToNull("IxxErr");
    }
    if (!d.isNull(DiaSource::IYY)) {
        db.template setColumn<float>("Iyy", d._iyy);
    } else {
        db.setColumnToNull("Iyy");
    }
    if (!d.isNull(DiaSource::IYY_ERR)) {
        db.template setColumn<float>("IyyErr", d._iyyErr);
    } else {
        db.setColumnToNull("IyyErr");
    }
    if (!d.isNull(DiaSource::IXY)) {
        db.template setColumn<float>("Ixy", d._ixy);
    } else {
        db.setColumnToNull("Ixy");
    }
    if (!d.isNull(DiaSource::IXY_ERR)) {
        db.template setColumn<float>("IxyErr", d._ixyErr);
    } else {
        db.setColumnToNull("IxyErr");
    }
    if(!d.isNull(DiaSource::OBS_CODE)) {
        db.template setColumn<char[3]> ("obsCode", d._obsCode);
    } else {
        db.setColumnToNull("obsCode");
    }
    if (!d.isNull(DiaSource::IS_SYNTHETIC)) {
        db.template setColumn<char>("isSynthetic", d._isSynthetic);
    } else {
        db.setColumnToNull("isSynthetic");
    }    
    if (!d.isNull(DiaSource::MOPS_STATUS)) {
        db.template setColumn<char>("mopsStatus", d._mopsStatus);
    } else {
        db.setColumnToNull("mopsStatus");
    }        
    if (!d.isNull(DiaSource::FLAG_4_ASSOCIATION)) {
        db.template setColumn<int16_t>("flag4association", d._flag4association);
    } else {
        db.setColumnToNull("flag4association");
    }
    if (!d.isNull(DiaSource::FLAG_4_DETECTION)) {
        db.template setColumn<int16_t>("flag4detection", d._flag4detection);
    } else {
        db.setColumnToNull("flag4detection");
    }
    if (!d.isNull(DiaSource::FLAG_4_WCS)) {
        db.template setColumn<int16_t>("flag4wcs", d._flag4wcs);
    } else {
        db.setColumnToNull("flag4wcs");
    }
    db.insertRow();
}

//! \cond
template void form::DiaSourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    DiaSource const &);
template void form::DiaSourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, DiaSource const &);
//! \endcond


/*! Prepares for reading DiaSource instances from a database table. */
void form::DiaSourceVectorFormatter::setupFetch(DbStorage & db, DiaSource & d) {
    db.outParam("diaSourceId",      &(d._diaSourceId));
    db.outParam("ampExposureId",    &(d._ampExposureId));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",         &(d._objectId));
    db.outParam("movingObjectId",   &(d._movingObjectId));
    db.outParam("procHistoryId",    &(d._procHistoryId));    
    db.outParam("scId",             &(d._scId));
    db.outParam("ra",               &(d._ra));
    db.outParam("ssmId",            &(d._ssmId));
    db.outParam("decl",             &(d._dec));
    db.outParam("raErr4detection",  &(d._raErr4detection));
    db.outParam("decErr4detection", &(d._decErr4detection));
    db.outParam("raErr4wcs",        &(d._raErr4wcs));
    db.outParam("decErr4wcs",       &(d._decErr4wcs));
    db.outParam("xFlux",            &(d._xFlux));
    db.outParam("xFluxErr",         &(d._xFluxErr));
    db.outParam("yFlux",            &(d._yFlux));    
    db.outParam("yFluxErr",         &(d._yFluxErr));
    db.outParam("raFlux",           &(d._raFlux));
    db.outParam("raFluxErr",        &(d._raFluxErr));
    db.outParam("decFlux",          &(d._decFlux));    
    db.outParam("decFluxErr",       &(d._decFluxErr));
    db.outParam("xPeak",            &(d._xPeak));
    db.outParam("yPeak",            &(d._yPeak));
    db.outParam("raPeak",           &(d._raPeak));
    db.outParam("decPeak",          &(d._decPeak));            
    db.outParam("xAstrom",          &(d._xAstrom));
    db.outParam("xAstromErr",       &(d._xAstromErr));    
    db.outParam("yAstrom",          &(d._yAstrom));
    db.outParam("yAstromErr",       &(d._yAstromErr));  
    db.outParam("raAstrom",         &(d._raAstrom));
    db.outParam("raAstromErr",      &(d._raAstromErr));    
    db.outParam("decAstrom",        &(d._decAstrom));
    db.outParam("decAstromErr",     &(d._decAstromErr));       
    db.outParam("taiMidPoint",      &(d._taiMidPoint));
    db.outParam("taiRange",         &(d._taiRange));
    db.outParam("fwhmA",            &(d._fwhmA));
    db.outParam("fwhmB",            &(d._fwhmB));
    db.outParam("fwhmTheta",        &(d._fwhmTheta));
    db.outParam("lengthDeg",        &(d._lengthDeg));
    db.outParam("flux",             &(d._flux));
    db.outParam("fluxErr",          &(d._fluxErr));
    db.outParam("psfMag",           &(d._psfMag));
    db.outParam("psfMagErr",        &(d._psfMagErr));
    db.outParam("apMag",            &(d._apMag));
    db.outParam("apMagErr",         &(d._apMagErr));
    db.outParam("modelMag",         &(d._modelMag));
    db.outParam("modelMagErr",      &(d._modelMagErr));
    db.outParam("apDia",            &(d._apDia));
    db.outParam("refMag",           &(d._refMag));
    db.outParam("Ixx",              &(d._ixx));
    db.outParam("IxxErr",           &(d._ixxErr));
    db.outParam("Iyy",              &(d._iyy));
    db.outParam("IyyErr",           &(d._iyyErr));
    db.outParam("Ixy",              &(d._ixy));
    db.outParam("IxyErr",           &(d._ixyErr));
    db.outParam("snr",              &(d._snr));
    db.outParam("chi2",             &(d._chi2));
    db.outParam("valx1",            &(d._valX1));
    db.outParam("valx2",            &(d._valX2));
    db.outParam("valy1",            &(d._valY1));
    db.outParam("valy2",            &(d._valY2));
    db.outParam("valxy",            &(d._valXY));  
    db.outParam("obsCode",          &(d._obsCode));
    db.outParam("isSynthetic",      &(d._isSynthetic));
    db.outParam("mopsStatus",       &(d._mopsStatus));              
    db.outParam("flag4association", &(d._flag4association));
    db.outParam("flag4detection",   &(d._flag4detection));
    db.outParam("flag4wcs",         &(d._flag4wcs));
}


template <class Archive>
void form::DiaSourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {
    DiaSourceVector * p = dynamic_cast<DiaSourceVector *>(persistable);
    archive & boost::serialization::base_object<Persistable>(*p);
    DiaSourceVector::size_type sz;

    if (Archive::is_loading::value) {
        DiaSource data;
        archive & sz;
        p->reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            p->push_back(data);
        }
    } else {
        sz = p->size();
        archive & sz;
        DiaSourceVector::iterator const end(p->end());
        for (DiaSourceVector::iterator i = p->begin(); i != end; ++i) {
            archive & *i;
        }
    }
}

template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);
//template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
//    boost::archive::binary_oarchive &, unsigned int const, Persistable *
//);
//template void DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
//    boost::archive::binary_iarchive &, unsigned int const, Persistable *
//);


void form::DiaSourceVectorFormatter::write( Persistable const *   persistable,
                        Storage::Ptr          storage,
                        lsst::daf::base::DataProperty::PtrType additionalData) {
                        
    if (persistable == 0) {
        throw ex::InvalidParameter("No Persistable provided");
    }
    if (!storage) {
        throw ex::InvalidParameter("No Storage provided");
    }

    DiaSourceVector const * p = dynamic_cast<DiaSourceVector const *>(persistable);
    if (p == 0) {
        throw ex::Runtime("Persistable was not of concrete type DiaSourceVector");
    }

    // Assume all have ids or none do.
    if (p->begin()->_diaDiaSourceId == 0 &&
        (!_policy || !_policy->exists("GenerateIds") ||
         _policy->getBool("GenerateIds"))) {
        DiaSourceVector* v    = const_cast<DiaSourceVector*>(p);
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ccdExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
            throw ex::InvalidParameter("ccdId out of range");
        }
        for (DiaSourceVector::iterator i = v->begin(); i != v->end(); ++i) {
            i->_diaSourceId    = generateDiaSourceId(seq, ccdId, visitId);
            i->_ccdExposureId  = ccdExposureId;
            ++seq;
            if (seq == 0) { // Overflowed
                throw ex::Runtime("Too many DiaSources");
            }
        }
    }

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage * bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw ex::Runtime("Didn't get BoostStorage");
        }
        bs->getOArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        std::string itemName(getItemName(additionalData));
        std::string name(getVisitSliceTableName(_policy, additionalData));
        std::string model = extractPolicyString(
            _policy,
            itemName + ".templateTableName",
            itemName + "Template"
        );
        bool mayExist = !extractOptionalFlag(additionalData, itemName + ".isPerSliceTable");
        if (typeid(*storage) == typeid(DbStorage)) {
            DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            DiaSourceVector::const_iterator const end(p->end());
            for (DiaSourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            DiaSourceVector::const_iterator const end(p->end());
            for (DiaSourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported"); 
    }
}


Persistable* form::DiaSourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::DataProperty::PtrType additionalData
) {
    std::auto_ptr<DiaSourceVector> p(new DiaSourceVector);

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage* bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw ex::Runtime("Didn't get BoostStorage");
        }
        bs->getIArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
        if (db == 0) {
            throw ex::Runtime("Didn't get DbStorage");
        }
        std::vector<std::string> tables;
        getAllVisitSliceTableNames(tables, _policy, additionalData);

        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            DiaSource data;
            setupFetch(*db, data);
            db->query();
            data.setNotNull();
            while (db->next()) {
                if (db->columnIsNull( 0)) { throw ex::Runtime("null column \"diaSourceId\"");      }
                if (db->columnIsNull( 1)) { throw ex::Runtime("null column \"exposureId\"");       }
                if (db->columnIsNull( 2)) { throw ex::Runtime("null column \"filterId\"");         }
                if (db->columnIsNull( 3)) { data.setNull(DiaSource::OBJECT_ID);                    }
                if (db->columnIsNull( 4)) { data.setNull(DiaSource::MOVING_OBJECT_ID);             }
                if (db->columnIsNull( 5)) { throw ex::Runtime("null column \"procHistoryId\"");    }                
                if (db->columnIsNull( 6)) { throw ex::Runtime("null column \"scId\"");             }
                if (db->columnIsNull( 7)) { throw ex::Runtime("null column \"ra\"");               }
                if (db->columnIsNull( 8)) { data.setNULL(DiaSource::SSM_ID);                       }
                if (db->columnIsNull( 9)) { throw ex::Runtime("null column \"decl\"");             }                
                if (db->columnIsNull(10)) { throw ex::Runtime("null column \"raErr4detection\"");  }
                if (db->columnIsNull(11)) { throw ex::Runtime("null column \"decErr4detection\""); }
                if (db->columnIsNull(12)) { data.setNull(DiaSource::RA_ERR_4_WCS);                 }
                if (db->columnIsNull(13)) { data.setNull(DiaSource::DEC_ERR_4_WCS);                }
                if (db->columnIsNull(14)) { data.setNull(DiaSource::X_FLUX); }
                if (db->columnIsNull(15)) { data.setNull(DiaSource::X_FLUX_ERR); }
                if (db->columnIsNull(16)) { data.setNull(DiaSource::Y_FLUX); }
                if (db->columnIsNull(17)) { data.setNull(DiaSource::Y_FLUX_ERR); }
                if (db->columnIsNull(18)) { data.setNull(DiaSource::RA_FLUX); }
                if (db->columnIsNull(19)) { data.setNull(DiaSource::RA_FLUX_ERR); }
                if (db->columnIsNull(20)) { data.setNull(DiaSource::DEC_FLUX); }
                if (db->columnIsNull(21)) { data.setNull(DiaSource::DEC_FLUX_ERR); }
                if (db->columnIsNull(22)) { data.setNull(DiaSource::X_PEAK); }
                if (db->columnIsNull(23)) { data.setNull(DiaSource::Y_PEAK); }
                if (db->columnIsNull(24)) { data.setNull(DiaSource::RA_PEAK); }
                if (db->columnIsNull(25)) { data.setNull(DiaSource::DEC_PEAK); }                                                  
                if (db->columnIsNull(26)) { data.setNull(DiaSource::X_ASTROM); }
                if (db->columnIsNull(27)) { data.setNull(DiaSource::X_ASTROM_ERR); }
                if (db->columnIsNull(28)) { data.setNull(DiaSource::Y_ASTROM); }
                if (db->columnIsNull(29)) { data.setNull(DiaSource::Y_ASTROM_ERR); }
                if (db->columnIsNull(30)) { data.setNull(DiaSource::RA_ASTROM); }
                if (db->columnIsNull(31)) { data.setNull(DiaSource::RA_ASTROM_ERR); }
                if (db->columnIsNull(32)) { data.setNull(DiaSource::DEC_ASTROM); }
                if (db->columnIsNull(33)) { data.setNull(DiaSource::DEC_ASTROM_ERR); }
                if (db->columnIsNull(34)) { throw ex::Runtime("null column \"taiMidPoint\"");      }                
                if (db->columnIsNull(35)) { throw ex::Runtime("null column \"taiRange\"");         }                
                if (db->columnIsNull(36)) { throw ex::Runtime("null column \"fwhmA\"");            }
                if (db->columnIsNull(37)) { throw ex::Runtime("null column \"fwhmB\"");            }
                if (db->columnIsNull(38)) { throw ex::Runtime("null column \"fwhmTheta\"");        }
                if (db->columnIsNull(39)) { throw ex::Runtime("null column \"lengthDeg\"");        }               
                if (db->columnIsNull(40)) { throw ex::Runtime("null column \"flux\"");             }
                if (db->columnIsNull(41)) { throw ex::Runtime("null column \"fluxErr\"");          }
                if (db->columnIsNull(42)) { throw ex::Runtime("null column \"psfMag\"");           }
                if (db->columnIsNull(43)) { throw ex::Runtime("null column \"psfMagErr\"");        }
                if (db->columnIsNull(44)) { throw ex::Runtime("null column \"apMag\"");            }
                if (db->columnIsNull(45)) { throw ex::Runtime("null column \"apMagErr\"");         }
                if (db->columnIsNull(46)) { throw ex::Runtime("null column \"modelMag\"");         }
                if (db->columnIsNull(47)) { data.setNull(DiaSource::MODEL_MAG_ERR);                }
                if (db->columnIsNull(48)) { data.setNull(DiaSource::AP_DIA);                       }
                if (db->columnIsNull(49)) { data.setNull(DiaSource::REF_MAG);                      }                
                if (db->columnIsNull(50)) { data.setNull(DiaSource::IXX);                          }
                if (db->columnIsNull(51)) { data.setNull(DiaSource::IXX_ERR);                      }
                if (db->columnIsNull(52)) { data.setNull(DiaSource::IYY);                          }
                if (db->columnIsNull(53)) { data.setNull(DiaSource::IYY_ERR);                      }
                if (db->columnIsNull(54)) { data.setNull(DiaSource::IXY);                          }
                if (db->columnIsNull(55)) { data.setNull(DiaSource::IXY_ERR);                      }                
                if (db->columnIsNull(56)) { throw ex::Runtime("null column \"snr\"");              }
                if (db->columnIsNull(57)) { throw ex::Runtime("null column \"chi2\"");             }                
                if (db->columnIsNull(58)) { throw ex::Runtime("null column \"valx1\"");            }
                if (db->columnIsNull(59)) { throw ex::Runtime("null column \"valx2\"");            }
                if (db->columnIsNull(60)) { throw ex::Runtime("null column \"valy1\"");            }
                if (db->columnIsNull(61)) { throw ex::Runtime("null column \"valy2\"");            }
                if (db->columnIsNull(62)) { throw ex::Runtime("null column \"valxy\"");            }
                if (db->columnIsNull(63)) { data.setNull(DiaSource::OBS_CODE);                     }
                if (db->columnIsNull(64)) { data.setNull(DiaSource::IS_SYNTHETIC);                 }
                if (db->columnIsNull(65)) { data.setNull(DiaSource::MOPS_STATUS);                  }                              
                if (db->columnIsNull(66)) { data.setNull(DiaSource::FLAG_4_ASSOCIATION);           }
                if (db->columnIsNull(67)) { data.setNull(DiaSource::FLAG_4_DETECTION);             }
                if (db->columnIsNull(68)) { data.setNull(DiaSource::FLAG_4_WCS);                   }
                p->push_back(data);
                data.setNotNull(); // clear out null markers
            }
            db->finishQuery();
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported");
    }
    return p.release();
}


void form::DiaSourceVectorFormatter::update(Persistable*, Storage::Ptr, lsst::daf::base::DataProperty::PtrType) {
    throw ex::Runtime("DiaSourceVectorFormatter: updates not supported");
}

}}} // namespace lsst::afw::formatters
