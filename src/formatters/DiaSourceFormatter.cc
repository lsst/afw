// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for DiaSource instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/format.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/DiaSourceFormatter.h"
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
using lsst::afw::detection::PersistableDiaSourceVector;

namespace form = lsst::afw::formatters;

// -- DiaSourceVectorFormatter ----------------

form::DiaSourceVectorFormatter::DiaSourceVectorFormatter(Policy::Ptr const & policy) 
    : lsst::daf::persistence::Formatter(typeid(*this)), _policy(policy) {}
    
DiaSourceVectorFormatter::~DiaSourceVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::DiaSourceVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new DiaSourceVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::DiaSourceVectorFormatter::registration(
    "PersistableDiaSourceVector",
    typeid(PersistableDiaSourceVector),
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
    setColumn(db, "diaSourceId", &(d._id));
    setColumn(db, "ampExposureId", &(d._ampExposureId));
    setColumn(db. "diaSource2Id", &(d._diaSource2Id), d.isNull(Field::DIA_SOURCE_2_ID));
    setColumn(db, "filterId", static_cast<char*>(&(d._filterId)));
    setColumn(db, "objectId", &(d._objectId), d.isNull(Field::OBJECT_ID));
    setColumn(db, "movingObjectId", &(d._movingObjectId), d.isNull(Field::MOVING_OBJECT_ID));
    setColumn(db, "procHistoryID", &(d._procHistoryId));
    setColumn(db, "scId", &(d._scId));
    setColumn(db, "ssmId", &(d.ssmId), d.isNull(Field::SSM_ID));        
    setColumn(db, "ra", &(d._ra));
    setColumn(db, "raErr4detection", &(d._raErr4detection));
    setColumn(db, "raErr4wcs", &(d._raErr4wcs), d.isNull(Field::RA_ERR_4_WCS));    
    setColumn(db, "decl", &(d._dec));
    setColumn(db, "decErr4detection", &(d._decErr4detection));
    setColumn(db, "decErr4wcs", (d._decErr4wcs), d.isNull(Field::DEC_ERR_4_WCS));    
    setColumn(db, "xFlux", &(d._xFlux), d.isNull(Field::X_FLUX));
    setColumn(db, "xFluxErr", &(d._xFluxErr), d.isNull(Field::X_FLUX_ERR));
    setColumn(db, "yFlux", &(d._yFlux), d.isNull(Field::Y_FLUX));
    setColumn(db, "yFluxErr", &(d._yFluxErr), d.isNull(Field::Y_FLUX_ERR));
    setColumn(db, "xFlux", &(d._raFlux), d.isNull(Field::RA_FLUX));
    setColumn(db, "raFluxErr", &(d._raFluxErr), d.isNull(Field::RA_FLUX_ERR));
    setColumn(db, "decFlux", &(d._decFlux), d.isNull(Field::DEC_FLUX));
    setColumn(db, "decFluxErr", &(d._decFluxErr), d.isNull(Field::DEC_FLUX_ERR));
    setColumn(db, "xPeak", &(d._xPeak), d.isNull(Field::X_PEAK));
    setColumn(db, "yPeak", &(d._yPeak), d.isNull(Field::Y_PEAK));
    setColumn(db, "raPeak", &(d._raPeak), d.isNull(Field::RA_PEAK));
    setColumn(db, "decPeak", &(d._decPeak), d.isNull(Field::DEC_PEAK));
    setColumn(db, "xAstrom", &(d._xAstrom), d.isNull(Field::X_ASTROM));
    setColumn(db, "xAstromErr", &(d._xAstromErr), d.isNull(Field::X_ASTROM_ERR));
    setColumn(db, "yAstrom", &(d._yAstrom), d.isNull(Field::Y_ASTROM));
    setColumn(db, "yAstromErr", &(d._yAstromErr), d.isNull(Field::Y_ASTROM_ERR));
    setColumn(db, "raAstrom", &(d._raAstrom), d.isNull(Field::RA_ASTROM));
    setColumn(db, "raAstromErr", &d._raAstromErr), d.isNull(Field::RA_ASTROM_ERR));
    setColumn(db, "decAstrom", (d._decAstrom), d.isNull(Field::DEC_ASTROM));
    setColumn(db, "decAstromErr", &(d._decAstromErr), d.isNull(Field::DEC_ASTROM_ERR));        
    setColumn(db, "taiMidPoint", &(d._taiMidPoint));
    setColumn(db, "taiRange", &(d._taiRange));
    setColumn(db, "fwhmA", &(d._fwhmA));
    setColumn(db, "fwhmB", &(d._fwhmB));
    setColumn(db, "fwhmTheta", &(d._fwhmTheta));
    setColumn(db, "lengthDeg", &(d._lengthDeg));
    setColumn(db, "flux", &(d._flux));
    setColumn(db, "fluxErr", &(d._fluxErr));         
    setColumn(db, "psfMag", &(d._psfMag));
    setColumn(db, "psfMagErr", &(d._psfMagErr));
    setColumn(db, "apMag", &(d._apMag));
    setColumn(db, "apMagErr", &(d._apMagErr));
    setColumn(db, "modelMag", &(d._modelMag));            
    setColumn(db, "modelMagErr", &(d._modelMagErr), d.isNull(Field::MODEL_MAG_ERR));   
    setColumn(db, "instMag", &(d._instMag));
    setColumn(db, "instMagErr", &(d._instMagErr));
    setColumn(db, "nonGrayCorrMag", &(d._nonGrayCorrMag), d.isNull(Field::NON_GRAY_CORR_MAG));
    setColumn(db, "nonGrayCorrMagErr", &(d._nonGrayCorrMagErr), d.isNull(Field::NON_GRAY_CORR_MAG_ERR));    
    setColumn(db, "atmCorrMag", &(d._atmCorrMag), d.isNull(Field::ATM_CORR_MAG));
    setColumn(db, "atmCorrMagErr", &(d._atmCorrMagErr), d.isNull(Field::ATM_CORR_MAG_ERR));
    setColumn(db, "apDia", &(d._apDia), d.isNull(Field::AP_DIA));
    setColumn(db, "refMag", &(d._refMag), d.isNull(Field::REF_MAG));
    setColumn(db, "Ixx", &(d._ixx), d.isNull(Field::IXX));
    setColumn(db, "IxxErr", &(d._ixxErr), d.isNull(Field::IXX_ERR));
    setColumn(db, "Iyy", &(d._iyy), d.isNull(Field::IYY));
    setColumn(db, "IyyErr",&(d._iyyErr), d.isNull(Field::IYY_ERR));
    setColumn(db, "Ixy", &(d._ixy), d.isNull(Field::IXY));
    setColumn(db, "IxyErr", &(d._ixyErr), d.isNull(Field::IXY_ERR));        
    setColumn(db, "snr", &(d._snr));
    setColumn(db, "chi2", &(d._chi2));
    setColumn(db, "valx1", &(d._valX1));
    setColumn(db, "valx2", &(d._valX2));
    setColumn(db, "valy1", &(d._valY1));
    setColumn(db, "valy2", &(d._valY2));
    setColumn(db, "valxy", &(d._valXY));        
    setColumn(db, "obsCode", &(d._obsCode), d.isNull(Field::OBS_CODE));
    setColumn(db, "isSynthetic", &(d._isSynthetic), d.isNull(Field::IS_SYNTHETIC));
    setColumn(db, "mopsStatus", &(d._mopsStatus), d.isNull(Field::MOPS_STATUS));      
    setColumn(db, "flag4association", &(d._flag4association, d.isNull(Field::FLAG_4_ASSOCIATION)));
    setColumn(db, "flag4detection", &(d._flag4detection), d.isNull(Field::FLAG_4_DETECTION));
    setColumn(db, "flag4wcs", &(d._flag4wcs, d.isNull(Field::FLAG_4_WCS)));
    setColumn(db, "flagClassification", &(d._flagClassification), d.isNull(Field::FLAG_CLASSIFICATION));  
  
    db.insertRow();
}

//! \cond
template void form::DiaSourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    DiaSource const &);
template void form::DiaSourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, DiaSource const &);
//! \endcond


/*! Prepares for reading DiaSource instances from a database table. */
void form::DiaSourceVectorFormatter::setupFetch(DbStorage & db, DiaSource & d) {
    db.outParam("diaSourceId",      &(d._id));
    db.outParam("ampExposureId",    &(d._ampExposureId));
    db.outParam("diaSource2Id",     &(d._diaSource2Id));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",         &(d._objectId));
    db.outParam("movingObjectId",   &(d._movingObjectId));
    db.outParam("procHistoryId",    &(d._procHistoryId));
    db.outParam("scId",             &(d._scId));
    db.outParam("ssmId",            &(d.ssmId));      
    db.outParam("ra",               &(d._ra));
    db.outParam("raErr4detection",  &(d._raErr4detection));
    db.outParam("raErr4wcs",        &(d._raErr4wcs));
    db.outParam("decl",             &(d._dec));
    db.outParam("decErr4detection", &(d._decErr4detection));
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
    db.outParam("lengthDeg",        &(d._lengthDeg);
    db.outParam("flux",             &(d._flux));
    db.outParam("fluxErr",          &(d._fluxErr));      
    db.outParam("psfMag",           &(d._psfMag));
    db.outParam("psfMagErr",        &(d._psfMagErr));
    db.outParam("apMag",            &(d._apMag));
    db.outParam("apMagErr",         &(d._apMagErr));
    db.outParam("modelMag",         &(d._modelMag));
    db.outParam("modelMagErr",      &(d._modelMagErr));
    db.outParam("instMag",          &(d._instMag));
    db.outParam("instMagErr",       &(d._instMagErr));
    db.outParam("nonGrayCorrMag",   &(d._nonGrayCorrMag));
    db.outParam("nonGrayCorrMagErr",&(d._nonGrayCorrMagErr));    
    db.outParam("atmCorrMag",       &(d._atmCorrMag));
    db.outParam("atmCorrMagErr",    &(d._atmCorrMagErr));     
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
    db.outParam("flagClassification", &(d._flagClassification));
}

template <class Archive>
void form::DiaSourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const version,
    Persistable *      persistable
) {
    PersistableDiaSourceVector * p = dynamic_cast<PersistableDiaSourceVector *>(persistable);
    
    archive & boost::serialization::base_object<Persistable>(*p);
    
    DiaSourceVector::size_type sz;

    if (Archive::is_loading::value) {        
        DiaSource data;
        archive >> sz;
        DiaSourceVector sourceVector(sz);
        for (; sz > 0; --sz) {
            archive >> data;
            sourceVector.push_back(data);
        }
        p->setSources(sourceVector);
    } else {
        DiaSourceVector sourceVector = p->getSources();
        sz = sourceVector->size();
        archive << sz;
        DiaSourceVector::iterator i = sourceVector.begin();
        DiaSourceVector::iterator const end(sourceVector.end());
        for ( ; i != end; ++i) {
            archive <<  *i;
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
    lsst::daf::base::PropertySet::Ptr additionalData
) {                       
    if (persistable == 0) {
		throw LSST_EXCEPT(ex::InvalidParameterException, "No Persistable provided");
    }
    if (!storage) {
		throw LSST_EXCEPT(ex::InvalidParameterException, "No Storage provided");
    }

    PersistableDiaSourceVector const * p = dynamic_cast<PersistableDiaSourceVector const *>(persistable);
    if (p == 0) {
		throw LSST_EXCEPT(ex::RuntimeErrorException, "Persistable was not of concrete type SourceVector");
    }
    DiaSourceVector sourceVector = p->getSources();   

    // Assume all have ids or none do.
    if (sourceVector.begin()->_sourceId == 0 && 
        (!_policy || !_policy->exists("GenerateIds") || _policy->getBool("GenerateIds"))
    ) {
     
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ampExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
			throw LSST_EXCEPT(ex::InvalidParameterException, "ampExposureId out of range");
        }
        
        DiaSourceVector::iterator i = sourceVector.begin();
        for ( ; i != sourceVector.end(); ++i) {
            i->setId(generateDiaSourceId(seq, ccdId, visitId));
            i->setAmpExposureId(ampExposureId);
            ++seq;
            if (seq == 0) { // Overflowed
    			throw LSST_EXCEPT(ex::RuntimeErrorException, "Too many Sources");
            }
        }
    }

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage * bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
    		throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get BoostStorage");
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
        		throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);
            
            DiaSourceVector::const_iterator i(sourceVector.being());
            DiaSourceVector::const_iterator const end(sourceVector.end());
            for ( ; i != end; ++i) {
                insertRow<DbStorage>(*db, *i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
    			throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);

            DiaSourceVector::const_iterator i(sourceVector.being());
            DiaSourceVector::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, *i);
            }
        }
    } else {
		throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported"); 
    }
}


Persistable* form::DiaSourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    std::auto_ptr<PersistableDiaSourceVector> p();

    if (typeid(*storage) == typeid(BoostStorage)) {
        BoostStorage* bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get BoostStorage");
        }
        bs->getIArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
        if (db == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbStorage");
        }
        std::vector<std::string> tables;
        getAllVisitSliceTableNames(tables, _policy, additionalData);

        DiaSourceVector sourceVector();
        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            DiaSource data;
            setupFetch(*db, data);
            db->query();
            while (db->next()) {
                if (db->columnIsNull(DIA_SOURCE_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"diaSourceId\""); 
                }
                if (db->columnIsNull(AMP_EXPOSURE_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"ampExposureId\""); 
                }
                if (db->columnIsNull(DIA_SOURCE_2_ID)) { data.setNull(DiaSource::DIA_SOURCE_2_ID); }
                if (db->columnIsNull(FILTER_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"filterId\""); 
                }
                if (db->columnIsNull(OBJECT_ID)) { data.setNull(DiaSource::OBJECT_ID); }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { data.setNull(DiaSource::MOVING_OBJECT_ID); }
                if (db->columnIsNull(PROC_HIUSTORY_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"procHistoryId\""); 
                }
                if (db->columnIsNull(SC_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"scId\""); 
                }                
                if (db->columnIsNull(SSM_ID)) { data.setNULL(DiaSource::SSM_ID); }
                if (db->columnIsNull(RA)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"ra\""); 
                }                
                if (db->columnIsNull(RA_ERR_4_DETECTION)) { 
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"raErr4detection\"");
                }
                if (db->columnIsNull(RA_ERR_4_WCS)) { data.setNull(DiaSource::RA_ERR_4_WCS); }
                if (db->columnIsNull(DECL)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decl\""); 
                }
                if (db->columnIsNull(DEC_ERR_4_DETECTION)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decErr4detection\""); 
                }                
                if (db->columnIsNull(DEC_ERR_4_WCS)) { data.setNull(DiaSource::DEC_ERR_4_WCS);                }
                if (db->columnIsNull(X_FLUX)) { data.setNull(DiaSource::X_FLUX); }
                if (db->columnIsNull(X_FLUX_ERR)) { data.setNull(DiaSource::X_FLUX_ERR); }
                if (db->columnIsNull(Y_FLUX)) { data.setNull(DiaSource::Y_FLUX); }
                if (db->columnIsNull(Y_FLUX_ERR)) { data.setNull(DiaSource::Y_FLUX_ERR); }
                if (db->columnIsNull(RA_FLUX)) { data.setNull(DiaSource::RA_FLUX); }
                if (db->columnIsNull(RA_FLUX_ERR)) { data.setNull(DiaSource::RA_FLUX_ERR); }
                if (db->columnIsNull(DEC_FLUX)) { data.setNull(DiaSource::DEC_FLUX); }
                if (db->columnIsNull(DEC_FLUX_ERR)) { data.setNull(DiaSource::DEC_FLUX_ERR); }
                if (db->columnIsNull(X_PEAK)) { data.setNull(DiaSource::X_PEAK); }
                if (db->columnIsNull(Y_PEAK)) { data.setNull(DiaSource::Y_PEAK); }
                if (db->columnIsNull(RA_PEAK)) { data.setNull(DiaSource::RA_PEAK); }
                if (db->columnIsNull(DEC_PEAK)) { data.setNull(DiaSource::DEC_PEAK); }
                if (db->columnIsNull(X_ASTROM)) { data.setNull(DiaSource::X_ASTROM); }
                if (db->columnIsNull(X_ASTROM_ERR)) { data.setNull(DiaSource::X_ASTROM_ERR); }
                if (db->columnIsNull(Y_ASTROM)) { data.setNull(DiaSource::Y_ASTROM); }
                if (db->columnIsNull(Y_ASTROM_ERR)) { data.setNull(DiaSource::Y_ASTROM_ERR); }
                if (db->columnIsNull(RA_ASTROM)) { data.setNull(DiaSource::RA_ASTROM); }
                if (db->columnIsNull(RA_ASTROM_ERR)) { data.setNull(DiaSource::RA_ASTROM_ERR); }
                if (db->columnIsNull(DEC_ASTROM)) { data.setNull(DiaSource::DEC_ASTROM); }
                if (db->columnIsNull(DEC_ASTROM_ERR)) { data.setNull(DiaSource::DEC_ASTROM_ERR); }
                if (db->columnIsNull(TAI_MIDPOINT)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"taiMidPoint\""); 
                }
                if (db->columnIsNull(TAI_RANGE)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"taiRange\""); 
                }
                if (db->columnIsNull(FWHM_A)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmA\""); 
                }
                if (db->columnIsNull(FWHM_B)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmB\""); 
                }
                if (db->columnIsNull(FWHM_THETA)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmTheta\""); 
                }
                if (db->columnIsNull(LENGTH_DEG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"lengthDeg\"");        
                }               
                if (db->columnIsNull(FLUX)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"flux\"");             
                }
                if (db->columnIsNull(FLUX_ERR)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fluxErr\"");          
                }                
                if (db->columnIsNull(PSF_MAG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"psfMag\"");           
                }
                if (db->columnIsNull(PSF_MAG_ERR)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"psfMagErr\"");        
                }
                if (db->columnIsNull(AP_MAG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"apMag\"");  
                }
                if (db->columnIsNull(AP_MAG_ERR)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"apMagErr\"");  
                }
                if (db->columnIsNull(MODEL_MAG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"modelMag\""); 
                }
                if (db->columnIsNull(MODEL_MAG_ERR)) { data.setNull(DiaSource::MODEL_MAG_ERR);  } 
                if (db->columnIsNull(INST_MAG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMag\""); 
                }
                if (db->columnIsNull(INST_MAG_ERR)) { 
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMagErr\""); 
                }                            
                if (db->columnIsNull(NON_GRAY_CORR_MAG)) { data.setNull(DiaSource::NON_GRAY_CORR_MAG); }
                if (db->columnIsNull(NON_GRAY_CORR_MAG_ERR)) { data.setNul;(DiaSource::NON_GRAY_CORR_MAG_ERR);}
                if (db->columnIsNull(ATM_CORR_MAG)) { data.setNull(DiaSource::ATM_CORR_MAG); }
                if (db->columnIsNull(ATM_CORR_MAG_ERR)) { data.setNull(DiaSource::ATM_CORR_MAG_ERR); }
                if (db->columnIsNull(AP_DIA)) { data.setNull(DiaSource::AP_DIA); }
                if (db->columnIsNull(REF_MAG)) { data.setNull(DiaSource::REF_MAG);}                
                if (db->columnIsNull(IXX)) { data.setNull(DiaSource::IXX);}
                if (db->columnIsNull(IXX_ERR)) { data.setNull(DiaSource::IXX_ERR);}
                if (db->columnIsNull(IYY)) { data.setNull(DiaSource::IYY); }
                if (db->columnIsNull(IYY_ERR)) { data.setNull(DiaSource::IYY_ERR);}
                if (db->columnIsNull(IXY)) { data.setNull(DiaSource::IXY);}
                if (db->columnIsNull(IXY_ERR)) { data.setNull(DiaSource::IXY_ERR); }
                if (db->columnIsNull(SNR)) { 
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"snr\""); 
                }
                if (db->columnIsNull(CHI2)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"chi2\""); 
                }
                if (db->columnIsNull(VAL_X1)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"valx1\"");
                }
                if (db->columnIsNull(VAL_X2)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"valx2\""); 
                }
                if (db->columnIsNull(VAL_Y1)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"valy1\""); 
                }
                if (db->columnIsNull(VAL_Y2)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"valy2\""); 
                }
                if (db->columnIsNull(VAL_XY)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"valxy\""); 
                }                
                if (db->columnIsNull(OBS_CODE)) { data.setNull(DiaSource::OBS_CODE); }
                if (db->columnIsNull(IS_SYNTHETIC)) { data.setNull(DiaSource::IS_SYNTHETIC); }
                if (db->columnIsNull(MOPS_STATUS)) { data.setNull(DiaSource::MOPS_STATUS); }
                if (db->columnIsNull(FLAG_4_ASSOCIATION)) { data.setNull(DiaSource::FLAG_4_ASSOCIATION);}
                if (db->columnIsNull(FLAG_4_DETECTION)) { data.setNull(DiaSource::FLAG_4_DETECTION); }
                if (db->columnIsNull(FLAG_4_WCS)) { data.setNull(DiaSource::FLAG_4_WCS); }       
                if (db->columnIsNull(FLAG_CLASSIFICATION)) {data.setNull(DiaSource::FLAG_CLASSIFICATION; }     

				//add source to vector
                sourceVector.push_back(data);

                //reset nulls for next source
                data.setNotNull();
            }
            db->finishQuery();
        }
        p->setSources(sourceVector);
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported");
    }
    return p.release();
}


void form::DiaSourceVectorFormatter::update(Persistable*, 
    Storage::Ptr, lsst::daf::base::DataProperty::PtrType
) {
	throw LSST_EXCEPT(ex::RuntimeErrorException, "DiaSourceVectorFormatter: updates not supported");
}

}}} // namespace lsst::afw::formatters
