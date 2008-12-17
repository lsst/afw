// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for Source instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/format.hpp"
#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/SourceFormatter.h"
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
using lsst::afw::detection::PersistableSourceVector;

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
    "PersistableSourceVector",
    typeid(PersistableSourceVector),
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
    setColumn(db, "sourceId",           d._sourceId);
    setColumn(db, "ampExposureId",      d._ampExposureId);
    setColumn(db, "filterId",           static_cast<char*>(d._filterId));
    setColumn(db, "objectId",           d._objectId);
    setColumn(db, "movingObjectId",     d._movingObjectId);
    setColumn(db, "procHistoryID",      d._procHistoryId);
    setColumn(db, "ra",                 d._ra);
    setColumn(db, "raErr4detection",    d._raErr4detection);
    setColumn(db, "raErr4wcs",          d._raErr4wcs);    
    setColumn(db, "decl",               d._dec);
    setColumn(db, "decErr4detection",   d._decErr4detection);
    setColumn(db, "decErr4wcs",         d._decErr4wcs);    
    setColumn(db, "xFlux",              d._xFlux);
    setColumn(db, "xFluxErr",           d._xFluxErr);
    setColumn(db, "yFlux",              d._yFlux);
    setColumn(db, "yFluxErr",           d._yFluxErr);
    setColumn(db, "xFlux",              d._raFlux);
    setColumn(db, "raFluxErr",          d._raFluxErr);
    setColumn(db, "decFlux",            d._decFlux);
    setColumn(db, "decFluxErr",         d._decFluxErr);
    setColumn(db, "xPeak",              d._xPeak);
    setColumn(db, "yPeak",              d._yPeak);
    setColumn(db, "raPeak",             d._raPeak);
    setColumn(db, "decPeak",            d._decPeak);
    setColumn(db, "xAstrom",            d._xAstrom);
    setColumn(db, "xAstromErr",         d._xAstromErr);
    setColumn(db, "yAstrom",            d._yAstrom);
    setColumn(db, "yAstromErr",         d._yAstromErr);
    setColumn(db, "raAstrom",           d._raAstrom);
    setColumn(db, "raAstromErr",        d._raAstromErr);
    setColumn(db, "decAstrom",          d._decAstrom);
    setColumn(db, "decAstromErr",       d._decAstromErr);        
    setColumn(db, "taiMidPoint",        d._taiMidPoint);
    setColumn(db, "taiRange",           d._taiRange);
    setColumn(db, "fwhmA",              d._fwhmA);
    setColumn(db, "fwhmB",              d._fwhmB);
    setColumn(db, "fwhmTheta",          d._fwhmTheta);
    setColumn(db, "psfMag",             d._psfMag);
    setColumn(db, "psfMagErr",          d._psfMagErr);
    setColumn(db, "apMag",              d._apMag);
    setColumn(db, "apMagErr",           d._apMagErr);
    setColumn(db, "modelMag",           d._modelMag);            
    setColumn(db, "modelMagErr",        d._modelMagErr);
    setColumn(db, "petroMag",           d._petroMag);
    setColumn(db, "petroMagErr",        d._petroMagErr);    
    setColumn(db, "instMag",            d._instMag);
    setColumn(db, "instMagErr",         d._instMagErr);
    setColumn(db, "nonGrayCorrMag",     d._nonGrayCorrMag);
    setColumn(db, "nonGrayCorrMagErr",  d._nonGrayCorrMagErr);    
    setColumn(db, "atmCorrMag",         d._atmCorrMag);
    setColumn(db, "atmCorrMagErr",      d._atmCorrMagErr);
    setColumn(db, "apDia",              d._apDia);
    setColumn(db, "snr",                d._snr);
    setColumn(db, "chi2",               d._chi2);
    setColumn(db, "sky",                d._sky);
    setColumn(db, "skyErr",             d._skyErr);
    setColumn(db, "flag4association",   d._flag4association);
    setColumn(db, "flag4detection",     d._flag4detection);
    setColumn(db, "flag4wcs",           d._flag4wcs);

    db.insertRow();
}

//! \cond
template void form::SourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    Source const &);
template void form::SourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, Source const &);
//! \endcond


/*! Prepares for reading Source instances from a database table. */
void form::SourceVectorFormatter::setupFetch(DbStorage & db, Source & d) {
    d.setAllNotNull();
    db.outParam("sourceId",         d._id);
    db.outParam("ampExposureId",    d._ampExposureId);
    db.outParam("filterId",         reinterpret_cast<char *>(d._filterId));
    db.outParam("objectId",         d._objectId);
    db.outParam("movingObjectId",   d._movingObjectId);
    db.outParam("procHistoryId",    d._procHistoryId);
    db.outParam("ra",               d._ra);
    db.outParam("raErr4detection",  d._raErr4detection);
    db.outParam("raErr4wcs",        d._raErr4wcs);
    db.outParam("decl",             d._dec);
    db.outParam("decErr4detection", d._decErr4detection);
    db.outParam("decErr4wcs",       d._decErr4wcs);
    db.outParam("xFlux",            d._xFlux);
    db.outParam("xFluxErr",         d._xFluxErr);
    db.outParam("yFlux",            d._yFlux);    
    db.outParam("yFluxErr",         d._yFluxErr);
    db.outParam("raFlux",           d._raFlux);
    db.outParam("raFluxErr",        d._raFluxErr);
    db.outParam("decFlux",          d._decFlux);    
    db.outParam("decFluxErr",       d._decFluxErr);
    db.outParam("xPeak",            d._xPeak);
    db.outParam("yPeak",            d._yPeak);
    db.outParam("raPeak",           d._raPeak);
    db.outParam("decPeak",          d._decPeak);            
    db.outParam("xAstrom",          d._xAstrom);
    db.outParam("xAstromErr",       d._xAstromErr);    
    db.outParam("yAstrom",          d._yAstrom);
    db.outParam("yAstromErr",       d._yAstromErr);  
    db.outParam("raAstrom",         d._raAstrom);
    db.outParam("raAstromErr",      d._raAstromErr);    
    db.outParam("decAstrom",        d._decAstrom);
    db.outParam("decAstromErr",     d._decAstromErr);    
    db.outParam("taiMidPoint",      d._taiMidPoint);
    db.outParam("taiRange",         d._taiRange);
    db.outParam("fwhmA",            d._fwhmA);
    db.outParam("fwhmB",            d._fwhmB);
    db.outParam("fwhmTheta",        d._fwhmTheta);
    db.outParam("psfMag",           d._psfMag);
    db.outParam("psfMagErr",        d._psfMagErr);
    db.outParam("apMag",            d._apMag);
    db.outParam("apMagErr",         d._apMagErr);
    db.outParam("modelMag",         d._modelMag);
    db.outParam("modelMagErr",      d._modelMagErr);
    db.outParam("petroMag",         d._petroMag);
    db.outParam("petroMagErr",      d._petroMagErr);
    db.outParam("instMag",          d._instMag);
    db.outParam("instMagErr",       d._instMagErr);
    db.outParam("nonGrayCorrMag",   d._nonGrayCorrMag);
    db.outParam("nonGrayCorrMagErr",d._nonGrayCorrMagErr);    
    db.outParam("atmCorrMag",       d._atmCorrMag);
    db.outParam("atmCorrMagErr",    d._atmCorrMagErr);     
    db.outParam("apDia",            d._apDia);    
    db.outParam("snr",              d._snr);
    db.outParam("chi2",             d._chi2);
    db.outParam("sky",              d._sky);
    db.outParam("skyErr",           d._skyErr);    
    db.outParam("flag4association", d._flag4association);
    db.outParam("flag4detection",   d._flag4detection);
    db.outParam("flag4wcs",         d._flag4wcs);
}


template <class Archive>
void form::SourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {  
    PersistableSourceVector * p = dynamic_cast<PersistableSourceVector *>(persistable);
    
    archive & boost::serialization::base_object<Persistable>(*p);
    
    SourceVector::size_type sz;

    if (Archive::is_loading::value) {        
        Source data;
        archive >> sz;
        SourceVector sourceVector(sz);
        for (; sz > 0; --sz) {
            archive >> data;
            sourceVector.push_back(data);
        }
        p->setSources(sourceVector);
    } else {
        SourceVector sourceVector = p->getSources();
        sz = sourceVector->size();
        archive << sz;
        SourceVector::iterator i = sourceVector.begin();
        SourceVector::iterator const end(sourceVector.end());
        for ( ; i != end; ++i) {
            archive <<  *i;
        }
    }
}

template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);


void form::SourceVectorFormatter::write(
    Persistable const *   persistable,
    Storage::Ptr          storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    if (persistable == 0) {
        throw ex::InvalidParameter("No Persistable provided");
    }
    if (!storage) {
        throw ex::InvalidParameter("No Storage provided");
    }

    PersistableSourceVector const * p = dynamic_cast<PersistableSourceVector const *>(persistable);
    if (p == 0) {
        throw ex::Runtime("Persistable was not of concrete type SourceVector");
    }
    SourceVector sourceVector = p->getSources();   

    // Assume all have ids or none do.
    if (sourceVector.begin()->_sourceId == 0 && 
        (!_policy || !_policy->exists("GenerateIds") || _policy->getBool("GenerateIds"))
    ) {
     
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ampExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
            throw ex::InvalidParameter("ampExposureId out of range");
        }
        
        SourceVector::iterator i = sourceVector.begin();
        for ( ; i != sourceVector.end(); ++i) {
            i->setId(generateSourceId(seq, ccdId, visitId));
            i->setAmpExposureId(ampExposureId);
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
            
            SourceVector::const_iterator i(sourceVector.being());
            SourceVector::const_iterator const end(sourceVector.end());
            for ( ; i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw ex::Runtime("Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);

            SourceVector::const_iterator i(sourceVector.being());
            SourceVector::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
        throw ex::InvalidParameter("Storage type is not supported"); 
    }
}


Persistable* form::SourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {   
    std::auto_ptr<PersistableSourceVector> p();

    
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

        SourceVector sourceVector();
        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            Source data;
            setupFetch(*db, data);
            db->query();
            while (db->next()) {
                if (db->columnIsNull(SOURCE_ID)) { throw ex::Runtime("null column \"sourceId\""); }
                if (db->columnIsNull(AMP_EXPOSURE_ID)) { data.setNull(Source::AMP_EXPOSURE_ID); }
                if (db->columnIsNull(FILTER_ID)) { throw ex::Runtime("null column \"filterId\""); }
                if (db->columnIsNull(OBJECT_ID)) { data.setNull(Source::OBJECT_ID); }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { data.setNull(Source::MOVING_OBJECT_ID); }
                if (db->columnIsNull(PROC_HIUSTORY_ID)) { throw ex::Runtime("null column \"procHistoryId\""); }            
                if (db->columnIsNull(RA)) { throw ex::Runtime("null column \"ra\""); }
                if (db->columnIsNull(DECL)) { throw ex::Runtime("null column \"decl\""); }
                if (db->columnIsNull(RA_ERR_4_WCS)) { throw ex::Runtime("null column \"raErr4wcs\""); }
                if (db->columnIsNull(DEC_ERR_4_WCS)) { throw ex::Runtime("null column \"decErr4wcs\""); }
                if (db->columnIsNull(RA_ERR_4_DETECTION)) { data.setNull(Source::RA_ERR_4_DETECTION); }
                if (db->columnIsNull(DEC_ERR_4_DETECTION)) { data.setNull(Source::DEC_ERR_4_DETECTION); }
                if (db->columnIsNull(X_FLUX)) { data.setNull(Source::X_FLUX); }
                if (db->columnIsNull(X_FLUX_ERR)) { data.setNull(Source::X_FLUX_ERR); }
                if (db->columnIsNull(Y_FLUX)) { data.setNull(Source::Y_FLUX); }
                if (db->columnIsNull(Y_FLUX_ERR)) { data.setNull(Source::Y_FLUX_ERR); }
                if (db->columnIsNull(RA_FLUX)) { data.setNull(Source::RA_FLUX); }
                if (db->columnIsNull(RA_FLUX_ERR)) { data.setNull(Source::RA_FLUX_ERR); }
                if (db->columnIsNull(DEC_FLUX)) { data.setNull(Source::DEC_FLUX); }
                if (db->columnIsNull(DEC_FLUX_ERR)) { data.setNull(Source::DEC_FLUX_ERR); }
                if (db->columnIsNull(X_PEAK)) { data.setNull(Source::X_PEAK); }
                if (db->columnIsNull(Y_PEAK)) { data.setNull(Source::Y_PEAK); }
                if (db->columnIsNull(RA_PEAK)) { data.setNull(Source::RA_PEAK); }
                if (db->columnIsNull(DEC_PEAK)) { data.setNull(Source::DEC_PEAK); }                                                  
                if (db->columnIsNull(X_ASTROM)) { data.setNull(Source::X_ASTROM); }
                if (db->columnIsNull(X_ASTROM_ERR)) { data.setNull(Source::X_ASTROM_ERR); }
                if (db->columnIsNull(Y_ASTROM)) { data.setNull(Source::Y_ASTROM); }
                if (db->columnIsNull(Y_ASTROM_ERR)) { data.setNull(Source::Y_ASTROM_ERR); }
                if (db->columnIsNull(RA_ASTROM)) { data.setNull(Source::RA_ASTROM); }
                if (db->columnIsNull(RA_ASTROM_ERR)) { data.setNull(Source::RA_ASTROM_ERR); }
                if (db->columnIsNull(DEC_ASTROM)) { data.setNull(Source::DEC_ASTROM); }
                if (db->columnIsNull(DEC_ASTROM_ERR)) { data.setNull(Source::DEC_ASTROM_ERR); }
                if (db->columnIsNull(TAI_MIDPOINT)) { throw ex::Runtime("null column \"taiMidPoint\""); }
                if (db->columnIsNull(TAI_RANGE)) { data.setNull(Source::TAI_RANGE); }
                if (db->columnIsNull(FWHM_A)) { throw ex::Runtime("null column \"fwhmA\"");            }
                if (db->columnIsNull(FWHM_B)) { throw ex::Runtime("null column \"fwhmB\"");            }
                if (db->columnIsNull(FWHM_THETA)) { throw ex::Runtime("null column \"fwhmTheta\"");        }
                if (db->columnIsNull(PSF_MAG)) { throw ex::Runtime("null column \"psfMag\"");           }
                if (db->columnIsNull(PSF_MAG_ERR)) { throw ex::Runtime("null column \"psfMagErr\"");        }
                if (db->columnIsNull(AP_MAG)) { throw ex::Runtime("null column \"apMag\"");  }
                if (db->columnIsNull(AP_MAG_ERR)) { throw ex::Runtime("null column \"apMagErr\"");  }
                if (db->columnIsNull(MODEL_MAG)) { throw ex::Runtime("null column \"modelMag\""); }
                if (db->columnIsNull(MODEL_MAG_ERR)) { throw ex::Runtime("null column \"modelMagErr\""); }
                if (db->columnIsNull(PETRO_MAG)) { data.setNull(Source::PETRO_MAG); }
                if (db->columnIsNull(PETRO_MAG_ERR)) { data.setNull(Source::PETRO_MAG_ERR); }    
                if (db->columnIsNull(INST_MAG)) { throw ex::Runtime("null column \"instMag\""); }
                if (db->columnIsNull(INST_MAG_ERR)) { throw ex::Runtime("null column \"instMagErr\""); }                            
                if (db->columnIsNull(NON_GRAY_CORR_MAG)) { data.setNull(Source::NON_GRAY_CORR_MAG); }
                if (db->columnIsNull(NON_GRAY_CORR_MAG_ERR)) { data.setNull(Source::NON_GRAY_CORR_MAG_ERR); }                
                if (db->columnIsNull(ATM_CORR_MAG)) { data.setNull(Source::ATM_CORR_MAG); }
                if (db->columnIsNull(ATM_CORR_MAG_ERR)) { data.setNull(Source::ATM_CORR_MAG_ERR); }                                                
                if (db->columnIsNull(AP_DIA)) { data.setNull(Source::AP_DIA); }                
                if (db->columnIsNull(SNR)) { throw ex::Runtime("null column \"snr\""); }
                if (db->columnIsNull(CHI2)) { throw ex::Runtime("null column \"chi2\""); }
                if (db->columnIsNull(SKY)) { data.setNull(Source::SKY); }
                if (db->columnIsNull(SKY_ERR)) { data.setNull(Source::SKY_ERR); }                                
                if (db->columnIsNull(FLAG_4_ASSOCIATION)) { data.setNull(Source::FLAG_4_ASSOCIATION);}
                if (db->columnIsNull(FLAG_4_DETECTION)) { data.setNull(Source::FLAG_4_DETECTION); }
                if (db->columnIsNull(FLAG_4_WCS)) { data.setNull(Source::FLAG_4_WCS); }
                sourceVector.push_back(data);
                
                //pointers may have changed,
                //need to reset outParam locations for next fetch 
                setupFetch(*db, data);
            }
            db->finishQuery();
        }
        p->setSources(sourceVector);
    } else {
        throw ex::InvalidParameter("Storage type is not supported");
    }
    

    return p.release();
}


void form::SourceVectorFormatter::update(Persistable*, 
    Storage::Ptr, lsst::daf::base::PropertySet::Ptr
) {
    throw ex::Runtime("SourceVectorFormatter: updates not supported");
}
