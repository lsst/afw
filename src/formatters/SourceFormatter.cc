// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for Source instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/any.hpp"
#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/SourceFormatters.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/Source.h"

namespace ex = lsst::pex::exceptions;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::Source;
using lsst::afw::detection::SourceVector;

namespace form = lsst::afw::formatters;

// -- SourceVectorFormatter ----------------

form::SourceVectorFormatter::SourceVectorFormatter(Policy::Ptr const & policy) :
    lsst::daf::persistence::Formatter(typeid(*this)),
    _policy(policy)
{}


form::SourceVectorFormatter::~SourceVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::SourceVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new SourceVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::SourceVectorFormatter::registration(
    "SourceVector",
    typeid(SourceVector),
    createInstance
);


/*!
    \internal   Generates a unique identifier for a Source given the id of the
                originating visit, the id of the originating ccd, and the sequence
                number of the Source within that slice.
 */
inline static int64_t generateSourceId(unsigned short seqNum, int ccdId, int64_t visitId) {
    return (visitId << 24) + (ccdId << 16) + seqNum;
}


/*!
    Inserts a single Source into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::SourceVectorFormatter::insertRow(T & db, Source const & d) {

    db.template setColumn<int64_t>("sourceId",         d._sourceId);
    db.template setColumn<int64_t>("ampExposureId",    d._ampExposureId);
    db.template setColumn<double> ("ra",               d._ra);
    db.template setColumn<double> ("decl",             d._dec);
    db.template setColumn<double> ("taiMidPoint",      d._taiMidPoint);
    db.template setColumn<double> ("psfMag",           d._psfMag);
    db.template setColumn<double> ("apMag",            d._apMag);
    db.template setColumn<double> ("modelMag",         d._modelMag);            
    db.template setColumn<float>  ("raErr4wcs",        d._raErr4wcs);
    db.template setColumn<float>  ("decErr4wcs",       d._decErr4wcs);    
    db.template setColumn<float>  ("taiRange",         d._taiRange); 
    db.template setColumn<float>  ("fwhmA",            d._fwhmA);
    db.template setColumn<float>  ("fwhmB",            d._fwhmB);
    db.template setColumn<float>  ("fwhmTheta",        d._fwhmTheta);
    db.template setColumn<float>  ("psfMagErr",        d._psfMagErr);
    db.template setColumn<float>  ("apMagErr",         d._apMagErr);
    db.template setColumn<float>  ("modelMagErr",      d._modelMagErr);
    db.template setColumn<float>  ("snr",              d._snr);
    db.template setColumn<float>  ("chi2",             d._chi2);
    db.template setColumn<int32_t>("procHistoryID",    d._procHistoryId);
    db.template setColumn<char>   ("filterId",         static_cast<char>(d._filterId));

    if (!d.isNull(Source::OBJECT_ID)) {
        db.template setColumn<int64_t>("objectId", d._objectId);
    } else {
        db.setColumnToNull("objectId");
    }
    if (!d.isNull(Source::MOVING_OBJECT_ID)) {
        db.template setColumn<int64_t>("movingObjectId", d._movingObjectId);
    } else {
        db.setColumnToNull("movingObjectId");
    }
    if (!d.isNull(Source::RA_ERR_4_DETECTION)) {
        db.template setColumn<float>("raErr4wcs", d._raErr4detection);
    } else {
        db.setColumnToNull("raErr4wcs");
    }
    if (!d.isNull(Source::DEC_ERR_4_DETECTION)) {
        db.template setColumn<float>("decErr4wcs", d._decErr4detection);
    } else {
        db.setColumnToNull("decErr4wcs");
    }
    if (!d.isNull(Source::X_FLUX)) {
        db.template setColumn<double>("xFlux", d._xFlux);
    } else {
        db.setColumnToNull("xFlux");
    }    
    if (!d.isNull(Source::X_FLUX_ERR)) {
        db.template setColumn<double>("xFluxErr", d._xFluxErr);
    } else {
        db.setColumnToNull("xFluxErr");
    }        
    if (!d.isNull(Source::Y_FLUX)) {
        db.template setColumn<double>("yFlux", d._yFlux);
    } else {
        db.setColumnToNull("yFlux");
    }    
    if (!d.isNull(Source::Y_FLUX_ERR)) {
        db.template setColumn<double>("yFluxErr", d._yFluxErr);
    } else {
        db.setColumnToNull("yFluxErr");
    } 
    if (!d.isNull(Source::RA_FLUX)) {
        db.template setColumn<double>("xFlux", d._raFlux);
    } else {
        db.setColumnToNull("raFlux");
    }    
    if (!d.isNull(Source::RA_FLUX_ERR)) {
        db.template setColumn<double>("raFluxErr", d._raFluxErr);
    } else {
        db.setColumnToNull("raFluxErr");
    }        
    if (!d.isNull(Source::DEC_FLUX)) {
        db.template setColumn<double>("decFlux", d._decFlux);
    } else {
        db.setColumnToNull("decFlux");
    }    
    if (!d.isNull(Source::DEC_FLUX_ERR)) {
        db.template setColumn<double>("decFluxErr", d._decFluxErr);
    } else {
        db.setColumnToNull("decFluxErr");
    }                     
    if (!d.isNull(Source::X_PEAK)) {
        db.template setColumn<double>("xPeak", d._xPeak);
    } else {
        db.setColumnToNull("xPeak");
    }        
    if (!d.isNull(Source::Y_PEAK)) {
        db.template setColumn<double>("yPeak", d._yPeak);
    } else {
        db.setColumnToNull("yPeak");
    } 
    if (!d.isNull(Source::RA_PEAK)) {
        db.template setColumn<double>("raPeak", d._raPeak);
    } else {
        db.setColumnToNull("raPeak");
    }     
    if (!d.isNull(Source::DEC_PEAK)) {
        db.template setColumn<double>("decPeak", d._decPeak);
    } else {
        db.setColumnToNull("decPeak");
    }     
    if (!d.isNull(Source::X_ASTROM)) {
        db.template setColumn<double>("xAstrom", d._xAstrom);
    } else {
        db.setColumnToNull("xAstrom");
    }    
    if (!d.isNull(Source::X_ASTROM_ERR)) {
        db.template setColumn<double>("xAstromErr", d._xAstromErr);
    } else {
        db.setColumnToNull("xAstromErr");
    }        
    if (!d.isNull(Source::Y_ASTROM)) {
        db.template setColumn<double>("yAstrom", d._yAstrom);
    } else {
        db.setColumnToNull("yAstrom");
    }    
    if (!d.isNull(Source::Y_ASTROM_ERR)) {
        db.template setColumn<double>("yAstromErr", d._yAstromErr);
    } else {
        db.setColumnToNull("yAstromErr");
    } 
    if (!d.isNull(Source::RA_ASTROM)) {
        db.template setColumn<double>("xAstrom", d._raAstrom);
    } else {
        db.setColumnToNull("raAstrom");
    }    
    if (!d.isNull(Source::RA_ASTROM_ERR)) {
        db.template setColumn<double>("raAstromErr", d._raAstromErr);
    } else {
        db.setColumnToNull("raAstromErr");
    }        
    if (!d.isNull(Source::DEC_ASTROM)) {
        db.template setColumn<double>("decAstrom", d._decAstrom);
    } else {
        db.setColumnToNull("decAstrom");
    }    
    if (!d.isNull(Source::DEC_ASTROM_ERR)) {
        db.template setColumn<double>("decAstromErr", d._decAstromErr);
    } else {
        db.setColumnToNull("decAstromErr");
    }                         
    if (!d.isNull(Source::TAI_RANGE)) {
        db.template setColumn<float>("taiRange", d._taiRange);
    } else {
        db.setColumnToNull("taiRange");
    }                         
    if (!d.isNull(Source::PETRO_MAG)) {
        db.template setColumn<double>("petroMag", d._petroMag);
    } else {
        db.setColumnToNull("petroMag");
    }     
    if (!d.isNull(Source::PETRO_MAG_ERR)) {
        db.template setColumn<float>("petroMagErr", d._petroMagErr);
    } else {
        db.setColumnToNull("petroMagErr");
    }  
    if (!d.isNull(Source::AP_DIA)) {
        db.template setColumn<float>("apDia", d._apDia);
    } else {
        db.setColumnToNull("apDia");
    }
    if (!d.isNull(Source::SKY)) {
        db.template setColumn<float>("sky", d._sky);
    } else {
        db.setColumnToNull("sky");
    }
    if (!d.isNull(Source::SKY_ERR)) {
        db.template setColumn<float>("skyErr", d._skyErr);
    } else {
        db.setColumnToNull("skyErr");
    }

    if (!d.isNull(Source::FLAG_4_ASSOCIATION)) {
        db.template setColumn<int16_t>("flag4association", d._flag4association);
    } else {
        db.setColumnToNull("flag4association");
    }
    if (!d.isNull(Source::FLAG_4_DETECTION)) {
        db.template setColumn<int16_t>("flag4detection", d._flag4detection);
    } else {
        db.setColumnToNull("flag4detection");
    }
    if (!d.isNull(Source::FLAG_4_WCS)) {
        db.template setColumn<int16_t>("flag4wcs", d._flag4wcs);
    } else {
        db.setColumnToNull("flag4wcs");
    }
    db.insertRow();
}

//! \cond
template void form::SourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    Source const &);
template void form::SourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, Source const &);
//! \endcond


/*! Prepares for reading Source instances from a database table. */
void form::SourceVectorFormatter::setupFetch(DbStorage & db, Source & d) {
    db.outParam("sourceId",         &(d._sourceId));
    db.outParam("ampExposureId",    &(d._ampExposureId));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",         &(d._objectId));
    db.outParam("movingObjectId",   &(d._movingObjectId));
    db.outParam("procHistoryId",    &(d._procHistoryId));
    db.outParam("ra",               &(d._ra));
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
    db.outParam("psfMag",           &(d._psfMag));
    db.outParam("psfMagErr",        &(d._psfMagErr));
    db.outParam("apMag",            &(d._apMag));
    db.outParam("apMagErr",         &(d._apMagErr));
    db.outParam("modelMag",         &(d._modelMag));
    db.outParam("modelMagErr",      &(d._modelMagErr));
    db.outParam("petroMag",         &(d._petroMag));
    db.outParam("petroMagErr",      &(d._petroMagErr));    
    db.outParam("apDia",            &(d._apDia));    
    db.outParam("snr",              &(d._snr));
    db.outParam("chi2",             &(d._chi2));
    db.outParam("sky",              &(d._sky));
    db.outParam("skyErr",           &(d._skyErr));    
    db.outParam("flag4association", &(d._flag4association));
    db.outParam("flag4detection",   &(d._flag4detection));
    db.outParam("flag4wcs",         &(d._flag4wcs));
}


template <class Archive>
void form::SourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {
    SourceVector * p = dynamic_cast<SourceVector *>(persistable);
    archive & boost::serialization::base_object<Persistable>(*p);
    SourceVector::size_type sz;

    if (Archive::is_loading::value) {
        Source data;
        archive & sz;
        p->reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            p->push_back(data);
        }
    } else {
        sz = p->size();
        archive & sz;
        SourceVector::iterator const end(p->end());
        for (SourceVector::iterator i = p->begin(); i != end; ++i) {
            archive & *i;
        }
    }
}

template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);
//template void SourceVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
//    boost::archive::binary_oarchive &, unsigned int const, Persistable *
//);
//template void SourceVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
//    boost::archive::binary_iarchive &, unsigned int const, Persistable *
//);


void form::SourceVectorFormatter::write(
    Persistable const *   persistable,
    Storage::Ptr          storage,
    lsst::daf::base::DataProperty::PtrType additionalData
) {
    if (persistable == 0) {
        throw ex::InvalidParameter("No Persistable provided");
    }
    if (!storage) {
        throw ex::InvalidParameter("No Storage provided");
    }

    SourceVector const * p = dynamic_cast<SourceVector const *>(persistable);
    if (p == 0) {
        throw ex::Runtime("Persistable was not of concrete type SourceVector");
    }

    // Assume all have ids or none do.
    if (p->begin()->_sourceId == 0 &&
        (!_policy || !_policy->exists("GenerateIds") ||
         _policy->getBool("GenerateIds"))) {
        SourceVector* v    = const_cast<SourceVector*>(p);
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ccdExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
            throw ex::InvalidParameter("ccdId out of range");
        }
        for (SourceVector::iterator i = v->begin(); i != v->end(); ++i) {
            i->setSourceId( generateSourceId(seq, ccdId, visitId));
            i->setAmpExposureId(ccdExposureId);
            ++seq;
            if (seq == 0) { // Overflowed
                throw ex::Runtime("Too many Sources");
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
            SourceVector::const_iterator const end(p->end());
            for (SourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            SourceVector::const_iterator const end(p->end());
            for (SourceVector::const_iterator i = p->begin(); i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported"); 
    }
}


Persistable* form::SourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::DataProperty::PtrType additionalData
) {
    std::auto_ptr<SourceVector> p(new SourceVector);

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
            Source data;
            setupFetch(*db, data);
            db->query();
            data.setNotNull();
            while (db->next()) {
                if (db->columnIsNull( 0)) { throw ex::Runtime("null column \"sourceId\""); }
                if (db->columnIsNull( 1)) { data.setNull(Source::AMP_EXPOSURE_ID); }
                if (db->columnIsNull( 2)) { throw ex::Runtime("null column \"filterId\""); }
                if (db->columnIsNull( 3)) { data.setNull(Source::OBJECT_ID); }
                if (db->columnIsNull( 4)) { data.setNull(Source::MOVING_OBJECT_ID); }
                if (db->columnIsNull( 5)) { throw ex::Runtime("null column \"procHistoryId\""); }            
                if (db->columnIsNull( 6)) { throw ex::Runtime("null column \"ra\""); }
                if (db->columnIsNull( 7)) { throw ex::Runtime("null column \"decl\""); }
                if (db->columnIsNull( 8)) { throw ex::Runtime("null column \"raErr4wcs\""); }
                if (db->columnIsNull( 9)) { throw ex::Runtime("null column \"decErr4wcs\""); }
                if (db->columnIsNull(10)) { data.setNull(Source::RA_ERR_4_DETECTION); }
                if (db->columnIsNull(11)) { data.setNull(Source::DEC_ERR_4_DETECTION); }
                if (db->columnIsNull(12)) { data.setNull(Source::X_FLUX); }
                if (db->columnIsNull(13)) { data.setNull(Source::X_FLUX_ERR); }
                if (db->columnIsNull(14)) { data.setNull(Source::Y_FLUX); }
                if (db->columnIsNull(15)) { data.setNull(Source::Y_FLUX_ERR); }
                if (db->columnIsNull(16)) { data.setNull(Source::RA_FLUX); }
                if (db->columnIsNull(17)) { data.setNull(Source::RA_FLUX_ERR); }
                if (db->columnIsNull(18)) { data.setNull(Source::DEC_FLUX); }
                if (db->columnIsNull(19)) { data.setNull(Source::DEC_FLUX_ERR); }
                if (db->columnIsNull(20)) { data.setNull(Source::X_PEAK); }
                if (db->columnIsNull(21)) { data.setNull(Source::Y_PEAK); }
                if (db->columnIsNull(22)) { data.setNull(Source::RA_PEAK); }
                if (db->columnIsNull(23)) { data.setNull(Source::DEC_PEAK); }                                                  
                if (db->columnIsNull(24)) { data.setNull(Source::X_ASTROM); }
                if (db->columnIsNull(25)) { data.setNull(Source::X_ASTROM_ERR); }
                if (db->columnIsNull(26)) { data.setNull(Source::Y_ASTROM); }
                if (db->columnIsNull(27)) { data.setNull(Source::Y_ASTROM_ERR); }
                if (db->columnIsNull(28)) { data.setNull(Source::RA_ASTROM); }
                if (db->columnIsNull(29)) { data.setNull(Source::RA_ASTROM_ERR); }
                if (db->columnIsNull(30)) { data.setNull(Source::DEC_ASTROM); }
                if (db->columnIsNull(31)) { data.setNull(Source::DEC_ASTROM_ERR); }
                if (db->columnIsNull(32)) { throw ex::Runtime("null column \"taiMidPoint\""); }
                if (db->columnIsNull(33)) { data.setNull(Source::TAI_RANGE); }
                if (db->columnIsNull(34)) { throw ex::Runtime("null column \"fwhmA\"");            }
                if (db->columnIsNull(35)) { throw ex::Runtime("null column \"fwhmB\"");            }
                if (db->columnIsNull(36)) { throw ex::Runtime("null column \"fwhmTheta\"");        }
                if (db->columnIsNull(37)) { throw ex::Runtime("null column \"psfMag\"");           }
                if (db->columnIsNull(38)) { throw ex::Runtime("null column \"psfMagErr\"");        }
                if (db->columnIsNull(39)) { throw ex::Runtime("null column \"apMag\"");            }
                if (db->columnIsNull(40)) { throw ex::Runtime("null column \"apMagErr\"");         }
                if (db->columnIsNull(41)) { throw ex::Runtime("null column \"modelMag\"");         }
                if (db->columnIsNull(42)) { throw ex::Runtime("null column \"modelMagErr\"");      }
                if (db->columnIsNull(43)) { data.setNull(Source::PETRO_MAG); }
                if (db->columnIsNull(44)) { data.setNull(Source::PETRO_MAG_ERR); }                
                if (db->columnIsNull(45)) { data.setNull(Source::AP_DIA);                       }                
                if (db->columnIsNull(46)) { throw ex::Runtime("null column \"snr\"");              }
                if (db->columnIsNull(47)) { throw ex::Runtime("null column \"chi2\"");             }
                if (db->columnIsNull(48)) { data.setNull(Source::SKY); }
                if (db->columnIsNull(49)) { data.setNull(Source::SKY_ERR); }                                
                if (db->columnIsNull(50)) { data.setNull(Source::FLAG_4_ASSOCIATION);           }
                if (db->columnIsNull(51)) { data.setNull(Source::FLAG_4_DETECTION);             }
                if (db->columnIsNull(52)) { data.setNull(Source::FLAG_4_WCS);                   }
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


void form::SourceVectorFormatter::update(Persistable*, Storage::Ptr, lsst::daf::base::DataProperty::PtrType) {
    throw ex::Runtime("SourceVectorFormatter: updates not supported");
}
