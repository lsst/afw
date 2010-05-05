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
namespace det = lsst::afw::detection;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::DiaSource;
using lsst::afw::detection::DiaSourceSet;
using lsst::afw::detection::PersistableDiaSourceVector;

namespace form = lsst::afw::formatters;

// -- DiaSourceVectorFormatter ----------------

form::DiaSourceVectorFormatter::DiaSourceVectorFormatter(Policy::Ptr const & policy) 
    : lsst::daf::persistence::Formatter(typeid(this)), _policy(policy) {}
    
form::DiaSourceVectorFormatter::~DiaSourceVectorFormatter() {}


lsst::daf::persistence::Formatter::Ptr form::DiaSourceVectorFormatter::createInstance(Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new DiaSourceVectorFormatter(policy));
}


lsst::daf::persistence::FormatterRegistration form::DiaSourceVectorFormatter::registration(
    "PersistableDiaSourceVector",
    typeid(PersistableDiaSourceVector),
    createInstance
);


/*!
    \internal   Generates a unique identifier for a DiaSource given the ampExposureId of the
                originating amplifier and the sequence number of the DiaSource within the amplifier.
 */
inline static int64_t generateDiaSourceId(unsigned short seqNum, int64_t ampExposureId) {
    return (ampExposureId << 16) + seqNum;
}

template <typename T, typename F>
inline static void insertFp(T & db, F const & val, char const * const col, bool isNull=false) {
    if (isNull || isnan(val)) {
        db.setColumnToNull(col);
    } else if (isinf(val)) {
        F replacement = (val > 0.0) ? std::numeric_limits<F>::max() :
                                     -std::numeric_limits<F>::max();
        db.template setColumn<F>(col, replacement);
    } else {
        db.template setColumn<F>(col, val);
    }
}


/*!
    Inserts a single DiaSource into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::DiaSourceVectorFormatter::insertRow(T & db, DiaSource const & d) {
    db.template setColumn<int64_t>("diaSourceId", d._id);
    db.template setColumn<int64_t>("ampExposureId", d._ampExposureId);

    if (!d.isNull(det::DIA_SOURCE_TO_ID)) {
        db.template setColumn<int64_t>("diaSourceToId", d._diaSourceToId);
    } else {
        db.setColumnToNull("diaSourceToId");
    }

    db.template setColumn<char>("filterId", static_cast<char>(d._filterId));

    if (!d.isNull(det::OBJECT_ID)) {
        db. template setColumn<int64_t>("objectId", d._objectId);
    } else {
        db.setColumnToNull("objectId");
    }

    if (!d.isNull(det::MOVING_OBJECT_ID)) {
        db. template setColumn<int64_t>("movingObjectId", d._movingObjectId);
    } else {
        db.setColumnToNull("movingObjectId");
    }

    insertFp(db, d._ra, "ra");
    insertFp(db, d._raErrForDetection, "raErrForDetection", d.isNull(det::RA_ERR_FOR_DETECTION));
    insertFp(db, d._raErrForWcs, "raErrForWcs", d.isNull(det::RA_ERR_FOR_WCS));
    insertFp(db, d._dec, "decl");
    insertFp(db, d._decErrForDetection, "declErrForDetection", d.isNull(det::DEC_ERR_FOR_DETECTION));
    insertFp(db, d._decErrForWcs, "declErrForWcs", d.isNull(det::DEC_ERR_FOR_WCS));

    insertFp(db, d._xAstrom, "xAstrom");
    insertFp(db, d._xAstromErr, "xAstromErr", d.isNull(det::X_ASTROM_ERR));
    insertFp(db, d._yAstrom, "yAstrom");
    insertFp(db, d._yAstromErr, "yAstromErr", d.isNull(det::Y_ASTROM_ERR));

    insertFp(db, d._taiMidPoint, "taiMidPoint");
    insertFp(db, d._taiRange, "taiRange");

    insertFp(db, d._psfFlux, "psfFlux");
    insertFp(db, d._psfFluxErr, "psfFluxErr", d.isNull(det::PSF_FLUX_ERR));
    insertFp(db, d._apFlux, "apFlux");
    insertFp(db, d._apFluxErr, "apFluxErr", d.isNull(det::AP_FLUX_ERR));
    insertFp(db, d._modelFlux, "modelFlux");
    insertFp(db, d._modelFluxErr, "modelFluxErr", d.isNull(det::MODEL_FLUX_ERR));
    insertFp(db, d._instFlux, "instFlux");
    insertFp(db, d._instFluxErr, "instFluxErr", d.isNull(det::INST_FLUX_ERR));

    insertFp(db, d._apDia, "apDia", d.isNull(det::AP_DIA));
   
    insertFp(db, d._ixx, "Ixx", d.isNull(det::IXX));
    insertFp(db, d._ixxErr, "IxxErr", d.isNull(det::IXX_ERR));
    insertFp(db, d._iyy, "Iyy", d.isNull(det::IYY));
    insertFp(db, d._iyyErr, "IyyErr", d.isNull(det::IYY_ERR));
    insertFp(db, d._ixy, "Ixy", d.isNull(det::IXY));
    insertFp(db, d._ixyErr, "IxyErr", d.isNull(det::IXY_ERR));

    insertFp(db, d._snr, "snr");
    insertFp(db, d._chi2, "chi2");

    if (!d.isNull(det::FLAG_FOR_DETECTION)) {
        db.template setColumn<boost::int64_t>("flagForDetection", d._flagForDetection);
    } else {
        db.setColumnToNull("flagForDetection");
    } 
    if (!d.isNull(det::FLAG_CLASSIFICATION)) {
        db.template setColumn<boost::int64_t>("flagForClassification", d._flagClassification);
    } else {
        db.setColumnToNull("flagForClassification");
    }

    #if 0
    //not defined in DC3a. Keep for DC3b
    db.template setColumn<int32_t>("procHistoryID", d._procHistoryId);
    db.template setColumn<int32_t>("scId", d._scId);
    
    if (!d.isNull(det::SSM_ID))
        db. template setColumn<int64_t>("ssmId", d._ssmId);        
    else db.setColumnToNull("ssmId");    

    if (!d.isNull(det::X_FLUX))
        db. template setColumn<double>("xFlux", d._xFlux);
    else db.setColumnToNull("xFlux");

    if (!d.isNull(det::X_FLUX_ERR))    
        db. template setColumn<double>("xFluxErr", d._xFluxErr);
    else db.setColumnToNull("xFluxErr");
    
    if (!d.isNull(det::Y_FLUX))
        db. template setColumn<double>("yFlux", d._yFlux);
    else db.setColumnToNull("yFlux");
    
    if (!d.isNull(det::Y_FLUX_ERR))
        db. template setColumn<double>("yFluxErr", d._yFluxErr);
    else db.setColumnToNull("yFluxErr");
    
    if (!d.isNull(det::RA_FLUX))
        db. template setColumn<double>("raFlux", d._raFlux);
    else db.setColumnToNull("raFlux");
    
    if (!d.isNull(det::RA_FLUX_ERR))
        db. template setColumn<double>("raFluxErr", d._raFluxErr);
    else db.setColumnToNull("raFluxErr");
    
    if (!d.isNull(det::DEC_FLUX))
        db. template setColumn<double>("declFlux", d._decFlux);
    else db.setColumnToNull("declFlux");
    
    if (!d.isNull(det::DEC_FLUX_ERR))
        db. template setColumn<double>("declFluxErr", d._decFluxErr);
    else db.setColumnToNull("declFluxErr");
    
    if (!d.isNull(det::X_PEAK))
        db. template setColumn<double>("xPeak", d._xPeak);
    else db.setColumnToNull("xPeak");
    
    if (!d.isNull(det::Y_PEAK))
        db. template setColumn<double>("yPeak", d._yPeak);
    else db.setColumnToNull("yPeak");
    
    if (!d.isNull(det::RA_PEAK))
        db. template setColumn<double>("raPeak", d._raPeak);
    else db.setColumnToNull("raPeak");
        
    if (!d.isNull(det::DEC_PEAK))    
        db. template setColumn<double>("declPeak", d._decPeak);
    else db.setColumnToNull("declPeak");
    
    if (!d.isNull(det::RA_ASTROM))
        db. template setColumn<double>("raAstrom", d._raAstrom);
    else db.setColumnToNull("raAstrom");
    
    if (!d.isNull(det::RA_ASTROM_ERR))
        db. template setColumn<double>("raAstromErr", d._raAstromErr);
    else db.setColumnToNull("raAstromErr");
    
    if (!d.isNull(det::DEC_ASTROM))
        db. template setColumn<double>("declAstrom", d._decAstrom);
    else db.setColumnToNull("declAstrom");
    
    if (!d.isNull(det::DEC_ASTROM_ERR))
        db. template setColumn<double>("declAstromErr", d._decAstromErr);        
    else db.setColumnToNull("declAstromErr");

    db.template setColumn<double>("lengthDeg", d._lengthDeg);

    if (!d.isNull(det::NON_GRAY_CORR_FLUX))
        db.template setColumn<double>("nonGrayCorrFlux", d._nonGrayCorrFlux);
    else db.setColumnToNull("nonGrayCorrFlux");
    
    if (!d.isNull(det::NON_GRAY_CORR_FLUX_ERR)) {
        db.template setColumn<double>("nonGrayCorrFluxErr", 
                d._nonGrayCorrFluxErr);
    }
    else db.setColumnToNull("nonGrayCorrFluxErr");
        
    if (!d.isNull(det::ATM_CORR_FLUX))
        db.template setColumn<double>("atmCorrFlux", d._atmCorrFlux);
    else db.setColumnToNull("atmCorrFlux");
    
    if (!d.isNull(det::ATM_CORR_FLUX_ERR))
        db.template setColumn<double>("atmCorrFluxErr", d._atmCorrFluxErr);
    else db.setColumnToNull("atmCorrFluxErr");

    if (!d.isNull(det::REF_FLUX))
        db.template setColumn<float>("refFlux", d._refFlux);
    else db.setColumnToNull("refFlux");

    db.template setColumn<double>("valx1", d._valX1);
    db.template setColumn<double>("valx2", d._valX2);
    db.template setColumn<double>("valy1", d._valY1);
    db.template setColumn<double>("valy2", d._valY2);
    db.template setColumn<double>("valxy", d._valXY);        

    if (!d.isNull(det::OBS_CODE))
        db.template setColumn<std::string>("obsCode", d._obsCode);
    else db.setColumnToNull("obsCode");
    
    if (!d.isNull(det::IS_SYNTHETIC))
        db.template setColumn<char>("isSynthetic", d._isSynthetic);
    else db.setColumnToNull("isSynthetic");
    
    if (!d.isNull(det::MOPS_STATUS))
        db.template setColumn<char>("mopsStatus", d._mopsStatus);      
    else db.setColumnToNull("mopsStatus");
    
    if (!d.isNull(det::FLAG_FOR_ASSOCIATION)) {
        db.template setColumn<boost::int16_t>("flagForAssociation", 
                d._flagForAssociation);
    }
    else db.setColumnToNull("flagForAssociation");

    if (!d.isNull(det::FLAG_FOR_WCS))
        db.template setColumn<boost::int16_t>("flagForWcs", d._flagForWcs);
    else db.setColumnToNull("flagForWcs");
    #endif

    db.insertRow();
}

//! \cond
template void form::DiaSourceVectorFormatter::insertRow<DbStorage>   (DbStorage &,    DiaSource const &);
template void form::DiaSourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage &, DiaSource const &);
//! \endcond


/*! Prepares for reading DiaSource instances from a database table. */
void form::DiaSourceVectorFormatter::setupFetch(DbStorage & db, DiaSource & d) {
    db.outParam("diaSourceId",          &(d._id));
    db.outParam("ampExposureId",        &(d._ampExposureId));
    db.outParam("diaSourceToId",        &(d._diaSourceToId));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",             &(d._objectId));
    db.outParam("movingObjectId",       &(d._movingObjectId));
    db.outParam("xAstrom",              &(d._xAstrom));
    db.outParam("xAstromErr",           &(d._xAstromErr));    
    db.outParam("yAstrom",              &(d._yAstrom));
    db.outParam("yAstromErr",           &(d._yAstromErr));  
    db.outParam("ra",                   &(d._ra));
    db.outParam("raErrForDetection",    &(d._raErrForDetection));
    db.outParam("raErrForWcs",          &(d._raErrForWcs));
    db.outParam("decl",                 &(d._dec));
    db.outParam("declErrForDetection",  &(d._decErrForDetection));
    db.outParam("declErrForWcs",        &(d._decErrForWcs));
    db.outParam("taiMidPoint",          &(d._taiMidPoint));
    db.outParam("taiRange",             &(d._taiRange)); 
    db.outParam("Ixx",                  &(d._ixx));
    db.outParam("IxxErr",               &(d._ixxErr));
    db.outParam("Iyy",                  &(d._iyy));
    db.outParam("IyyErr",               &(d._iyyErr));
    db.outParam("Ixy",                  &(d._ixy));
    db.outParam("IxyErr",               &(d._ixyErr)); 
    db.outParam("psfFlux",              &(d._psfFlux));
    db.outParam("psfFluxErr",           &(d._psfFluxErr));
    db.outParam("apFlux",               &(d._apFlux));
    db.outParam("apFluxErr",            &(d._apFluxErr));
    db.outParam("modelFlux",            &(d._modelFlux));
    db.outParam("modelFluxErr",         &(d._modelFluxErr));
    db.outParam("instFlux",             &(d._instFlux));
    db.outParam("instFluxErr",          &(d._instFluxErr));
    db.outParam("apDia",                &(d._apDia));      
    db.outParam("flagForClassification",&(d._flagClassification));
    db.outParam("flagForDetection",     &(d._flagForDetection));
    db.outParam("snr",                  &(d._snr));
    db.outParam("chi2",                 &(d._chi2));

    #if 0
    //not defined in DC3a. Keep for DC3b
    db.outParam("procHistoryId",      &(d._procHistoryId));
    db.outParam("scId",               &(d._scId));
    db.outParam("ssmId",              &(d._ssmId));  
    db.outParam("xFlux",              &(d._xFlux));
    db.outParam("xFluxErr",           &(d._xFluxErr));
    db.outParam("yFlux",              &(d._yFlux));    
    db.outParam("yFluxErr",           &(d._yFluxErr));
    db.outParam("raFlux",             &(d._raFlux));
    db.outParam("raFluxErr",          &(d._raFluxErr));
    db.outParam("declFlux",           &(d._decFlux));    
    db.outParam("declFluxErr",        &(d._decFluxErr));
    db.outParam("xPeak",              &(d._xPeak));
    db.outParam("yPeak",              &(d._yPeak));
    db.outParam("raPeak",             &(d._raPeak));
    db.outParam("declPeak",           &(d._decPeak));            
    db.outParam("raAstrom",           &(d._raAstrom));
    db.outParam("raAstromErr",        &(d._raAstromErr));    
    db.outParam("declAstrom",         &(d._decAstrom));
    db.outParam("declAstromErr",      &(d._decAstromErr));
    db.outParam("lengthDeg",          &(d._lengthDeg));
    db.outParam("nonGrayCorrFlux",    &(d._nonGrayCorrFlux));
    db.outParam("nonGrayCorrFluxErr", &(d._nonGrayCorrFluxErr));    
    db.outParam("atmCorrFlux",        &(d._atmCorrFlux));
    db.outParam("atmCorrFluxErr",     &(d._atmCorrFluxErr)); 
    db.outParam("refFlux",            &(d._refFlux));
    db.outParam("flag4wcs",           &(d._flagForWcs));
    db.outParam("flag4association",   &(d._flagForAssociation));
    db.outParam("valx1",              &(d._valX1));
    db.outParam("valx2",              &(d._valX2));
    db.outParam("valy1",              &(d._valY1));
    db.outParam("valy2",              &(d._valY2));
    db.outParam("valxy",              &(d._valXY));  
    db.outParam("obsCode",            &(d._obsCode));
    db.outParam("isSynthetic",        &(d._isSynthetic));
    db.outParam("mopsStatus",         &(d._mopsStatus));  
    #endif

    
}

template <class Archive>
void form::DiaSourceVectorFormatter::delegateSerialize(
    Archive &          archive,
    unsigned int const,
    Persistable *      persistable
) {
    PersistableDiaSourceVector * p = dynamic_cast<PersistableDiaSourceVector *>(persistable);
    
    archive & boost::serialization::base_object<Persistable>(*p);
    
    DiaSourceSet::size_type sz;

    if (Archive::is_loading::value) {        
        DiaSource data;
        archive & sz;
        p->_sources.clear();
        p->_sources.reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            DiaSource::Ptr sourcePtr(new DiaSource(data));
            p->_sources.push_back(sourcePtr);
        }
    } else {
        sz = p->_sources.size();
        archive & sz;
        DiaSourceSet::iterator i = p->_sources.begin();
        DiaSourceSet::iterator const end(p->_sources.end());
        for ( ; i != end; ++i) {
            archive &  **i;
        }
    }
}

template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);

/** 
 * Persist a collection of DiaSource to BoostStorage, DbStorage or DbTsvStorage
 */
void form::DiaSourceVectorFormatter::write( Persistable const * persistable,
    Storage::Ptr storage,
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
    DiaSourceSet sourceVector = p->getSources();   

    // Assume all have ids or none do.
    if ((*sourceVector.begin())->getId() == 0 && 
        (!_policy || !_policy->exists("GenerateIds") || _policy->getBool("GenerateIds"))
    ) {
        unsigned short seq = 1;
        boost::int64_t ampExposureId = extractAmpExposureId(additionalData);
        if (sourceVector.size() >= 65536) {
            throw LSST_EXCEPT(ex::RangeErrorException, "too many DiaSources per-amp: "
                "sequence number overflows 16 bits, causing unique-id conflicts");
        }
        
        DiaSourceSet::iterator i = sourceVector.begin();
        for ( ; i != sourceVector.end(); ++i) {
            (*i)->setId(generateDiaSourceId(seq, ampExposureId));
            (*i)->setAmpExposureId(ampExposureId);
            ++seq;
            if (seq == 0) { // Overflowed
                throw LSST_EXCEPT(ex::RuntimeErrorException, "Too many Sources");
            }
        }
    }

    if (typeid(*storage) == typeid(BoostStorage)) {
        //persist to BoostStorage
        BoostStorage * bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get BoostStorage");
        }
        
        //call serializeDelegate
        bs->getOArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        std::string itemName(getItemName(additionalData));
        std::string name(getTableName(_policy, additionalData));
        std::string model = _policy->getString(itemName + ".templateTableName");
        if (typeid(*storage) == typeid(DbStorage)) {
            //handle persisting to DbStorage
            DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
            if (db == 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbStorage");
            }
            db->createTableFromTemplate(name, model, true);
            db->setTableForInsert(name);
            
            DiaSourceSet::const_iterator i(sourceVector.begin());
            DiaSourceSet::const_iterator const end(sourceVector.end());
            for ( ; i != end; ++i) {
                insertRow<DbStorage>(*db, **i);
            }
        } else {
            //handle persisting to DbTsvStorage
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, true);
            db->setTableForInsert(name);

            DiaSourceSet::const_iterator i(sourceVector.begin());
            DiaSourceSet::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, **i);
            }
        }
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported"); 
    }
}

template <typename T>
static inline void handleNullFp(DbStorage * db, int col, T & val) {
    if (db->columnIsNull(col)) {
        val = std::numeric_limits<T>::quiet_NaN();
    }
}
template <typename T>
static inline void handleNullFp(DbStorage * db, DiaSource & src, int col, T & val, int flag) {
    if (db->columnIsNull(col)) {
        src.setNull(flag);
        val = std::numeric_limits<T>::quiet_NaN(); 
    } 
}


/** 
 * Retrieve a collection of DiaSource from BoostStorage, DbStorage or DbTsvStorage
 */
Persistable* form::DiaSourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    std::auto_ptr<PersistableDiaSourceVector> p(new PersistableDiaSourceVector());

    if (typeid(*storage) == typeid(BoostStorage)) {
        //handle retrieval from BoostStorage
        BoostStorage* bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get BoostStorage");
        }
        //calls serializeDelegate
        bs->getIArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) || typeid(*storage) == typeid(DbTsvStorage)) {
        //handle retrieval from DbStorage, DbTsvStorage
        DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
        if (db == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbStorage");
        }
        
        //get a list of tables from policy and additionalData
        std::vector<std::string> tables = getAllSliceTableNames(_policy, additionalData);

        DiaSourceSet sourceVector;
        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            DiaSource data;
            
            //set target for query output
            setupFetch(*db, data);
            
            //perform query
            db->query();
            
            //Loop over every value in the returned query
            //add a DiaSource to sourceVector
            data.setNotNull();
            while (db->next()) {
                //Now validate each column. 
                //If NULLABLE column is null, set that field null in resulting DiaSource
                //else if NON-NULLABLE column is null, throw exception
                if (db->columnIsNull(DIA_SOURCE_ID)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"diaSourceId\""); 
                }
                if (db->columnIsNull(AMP_EXPOSURE_ID)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"ampExposureId\""); 
                }
                if (db->columnIsNull(DIA_SOURCE_TO_ID)) { 
                    data.setNull(det::DIA_SOURCE_TO_ID); 
                }
                if (db->columnIsNull(FILTER_ID)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"filterId\""); 
                }
                if (db->columnIsNull(OBJECT_ID)) { 
                    data.setNull(det::OBJECT_ID); 
                }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { 
                    data.setNull(det::MOVING_OBJECT_ID); 
                }

                handleNullFp(db, RA, data._ra);
                handleNullFp(db, DECL, data._dec);
                handleNullFp(db, data, RA_ERR_FOR_WCS,
                             data._raErrForWcs, det::RA_ERR_FOR_WCS);
                handleNullFp(db, data, DEC_ERR_FOR_WCS,
                             data._decErrForWcs, det::DEC_ERR_FOR_WCS);
                handleNullFp(db, data, RA_ERR_FOR_DETECTION,
                             data._raErrForDetection, det::RA_ERR_FOR_DETECTION);
                handleNullFp(db, data, DEC_ERR_FOR_DETECTION,
                             data._decErrForDetection, det::DEC_ERR_FOR_DETECTION);

                handleNullFp(db, X_ASTROM, data._xAstrom);
                handleNullFp(db, data, X_ASTROM_ERR, data._xAstromErr, det::X_ASTROM_ERR);
                handleNullFp(db, Y_ASTROM, data._yAstrom);
                handleNullFp(db, data, Y_ASTROM_ERR, data._yAstromErr, det::Y_ASTROM_ERR);

                handleNullFp(db, TAI_MID_POINT, data._taiMidPoint);
                handleNullFp(db, TAI_RANGE, data._taiRange);

                handleNullFp(db, data, IXX, data._ixx, det::IXX);
                handleNullFp(db, data, IXX_ERR, data._ixxErr, det::IXX_ERR);
                handleNullFp(db, data, IYY, data._iyy, det::IYY);
                handleNullFp(db, data, IYY_ERR, data._iyyErr, det::IYY_ERR);
                handleNullFp(db, data, IXY, data._ixy, det::IXY);
                handleNullFp(db, data, IXY_ERR, data._ixyErr, det::IXY_ERR);

                handleNullFp(db, PSF_FLUX, data._psfFlux);
                handleNullFp(db, data, PSF_FLUX_ERR, data._psfFluxErr, det::PSF_FLUX_ERR);
                handleNullFp(db, AP_FLUX, data._apFlux);
                handleNullFp(db, data, AP_FLUX_ERR, data._apFluxErr, det::AP_FLUX_ERR);
                handleNullFp(db, MODEL_FLUX, data._modelFlux);
                handleNullFp(db, data, MODEL_FLUX_ERR, data._modelFluxErr, det::MODEL_FLUX_ERR);
                handleNullFp(db, INST_FLUX, data._instFlux);
                handleNullFp(db, data, INST_FLUX_ERR, data._instFluxErr, det::INST_FLUX_ERR);

                handleNullFp(db, data, AP_DIA, data._apDia, det::AP_DIA);
                handleNullFp(db, SNR, data._snr);
                handleNullFp(db, CHI2, data._chi2);

                if (db->columnIsNull(FLAG_FOR_CLASSIFICATION)) {
                    data.setNull(det::FLAG_CLASSIFICATION); 
                }    
                if (db->columnIsNull(FLAG_FOR_DETECTION)) { 
                    data.setNull(det::FLAG_FOR_DETECTION); 
                }

                //following are not in DC3a schema. set to null.
                data.setNull(det::SSM_ID);
                data.setNull(det::X_FLUX);
                data.setNull(det::X_FLUX_ERR);
                data.setNull(det::Y_FLUX);
                data.setNull(det::Y_FLUX_ERR);
                data.setNull(det::RA_FLUX);
                data.setNull(det::RA_FLUX_ERR);
                data.setNull(det::DEC_FLUX);
                data.setNull(det::DEC_FLUX_ERR);
                data.setNull(det::X_PEAK);
                data.setNull(det::Y_PEAK);
                data.setNull(det::RA_PEAK);
                data.setNull(det::DEC_PEAK);
                data.setNull(det::RA_ASTROM);
                data.setNull(det::RA_ASTROM_ERR);
                data.setNull(det::DEC_ASTROM);
                data.setNull(det::DEC_ASTROM_ERR);
                data.setNull(det::NON_GRAY_CORR_FLUX);
                data.setNull(det::NON_GRAY_CORR_FLUX_ERR);
                data.setNull(det::ATM_CORR_FLUX);
                data.setNull(det::ATM_CORR_FLUX_ERR);
                data.setNull(det::REF_FLUX);
                data.setNull(det::OBS_CODE);
                data.setNull(det::IS_SYNTHETIC);
                data.setNull(det::MOPS_STATUS);
                data.setNull(det::FLAG_FOR_ASSOCIATION);
                data.setNull(det::FLAG_FOR_WCS);
                data._xFlux = std::numeric_limits<double>::quiet_NaN();
                data._xFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._yFlux = std::numeric_limits<double>::quiet_NaN();
                data._yFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._raFlux = std::numeric_limits<double>::quiet_NaN();
                data._raFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._decFlux = std::numeric_limits<double>::quiet_NaN();
                data._decFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._xPeak = std::numeric_limits<double>::quiet_NaN();
                data._yPeak = std::numeric_limits<double>::quiet_NaN();
                data._raPeak = std::numeric_limits<double>::quiet_NaN();
                data._decPeak = std::numeric_limits<double>::quiet_NaN();
                data._raAstrom = std::numeric_limits<double>::quiet_NaN();
                data._raAstromErr = std::numeric_limits<float>::quiet_NaN();
                data._decAstrom = std::numeric_limits<double>::quiet_NaN();
                data._decAstromErr = std::numeric_limits<float>::quiet_NaN();
                data._nonGrayCorrFlux = std::numeric_limits<double>::quiet_NaN();
                data._nonGrayCorrFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._atmCorrFlux = std::numeric_limits<double>::quiet_NaN();
                data._atmCorrFluxErr = std::numeric_limits<float>::quiet_NaN();
                data._refFlux = std::numeric_limits<double>::quiet_NaN();

                #if 0
                //Not defined in DC3a. Keep for DC3b.
                if (db->columnIsNull(PROC_HISTORY_ID)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"procHistoryId\""); 
                }
                if (db->columnIsNull(SC_ID)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"scId\""); 
                }                
                if (db->columnIsNull(SSM_ID)) { data.setNull(det::SSM_ID); }
                if (db->columnIsNull(X_FLUX)) { data.setNull(det::X_FLUX); }
                if (db->columnIsNull(X_FLUX_ERR)) { data.setNull(det::X_FLUX_ERR); }
                if (db->columnIsNull(Y_FLUX)) { data.setNull(det::Y_FLUX); }
                if (db->columnIsNull(Y_FLUX_ERR)) { data.setNull(det::Y_FLUX_ERR); }
                if (db->columnIsNull(RA_FLUX)) { data.setNull(det::RA_FLUX); }
                if (db->columnIsNull(RA_FLUX_ERR)) { data.setNull(det::RA_FLUX_ERR); }
                if (db->columnIsNull(DEC_FLUX)) { data.setNull(det::DEC_FLUX); }
                if (db->columnIsNull(DEC_FLUX_ERR)) { data.setNull(det::DEC_FLUX_ERR); }
                if (db->columnIsNull(X_PEAK)) { data.setNull(det::X_PEAK); }
                if (db->columnIsNull(Y_PEAK)) { data.setNull(det::Y_PEAK); }
                if (db->columnIsNull(RA_PEAK)) { data.setNull(det::RA_PEAK); }
                if (db->columnIsNull(DEC_PEAK)) { data.setNull(det::DEC_PEAK); }
                if (db->columnIsNull(RA_ASTROM)) { data.setNull(det::RA_ASTROM); }
                if (db->columnIsNull(RA_ASTROM_ERR)) { data.setNull(det::RA_ASTROM_ERR); }
                if (db->columnIsNull(DEC_ASTROM)) { data.setNull(det::DEC_ASTROM); }
                if (db->columnIsNull(DEC_ASTROM_ERR)) { data.setNull(det::DEC_ASTROM_ERR); }
                if (db->columnIsNull(LENGTH_DEG)) {
                    throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"lengthDeg\"");        
                }     
                if (db->columnIsNull(NON_GRAY_CORR_FLUX)) { data.setNull(det::NON_GRAY_CORR_FLUX); }
                if (db->columnIsNull(NON_GRAY_CORR_FLUX_ERR)) { data.setNull(det::NON_GRAY_CORR_FLUX_ERR);}
                if (db->columnIsNull(ATM_CORR_FLUX)) { data.setNull(det::ATM_CORR_FLUX); }
                if (db->columnIsNull(ATM_CORR_FLUX_ERR)) { data.setNull(det::ATM_CORR_FLUX_ERR); }
                if (db->columnIsNull(REF_FLUX)) { data.setNull(det::REF_FLUX);}  
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
                if (db->columnIsNull(OBS_CODE)) { data.setNull(det::OBS_CODE); }
                if (db->columnIsNull(IS_SYNTHETIC)) { data.setNull(det::IS_SYNTHETIC); }
                if (db->columnIsNull(MOPS_STATUS)) { data.setNull(det::MOPS_STATUS); }
                if (db->columnIsNull(FLAG_FOR_ASSOCIATION)) { data.setNull(det::FLAG_FOR_ASSOCIATION);}
                if (db->columnIsNull(FLAG_FOR_WCS)) { data.setNull(det::FLAG_FOR_WCS); }       
                #endif 

                //add source to vector
                DiaSource::Ptr sourcePtr(new DiaSource(data));
                sourceVector.push_back(sourcePtr);

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
    Storage::Ptr, lsst::daf::base::PropertySet::Ptr
) {
    throw LSST_EXCEPT(ex::RuntimeErrorException, "DiaSourceVectorFormatter: updates not supported");
}
