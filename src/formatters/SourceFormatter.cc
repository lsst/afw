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
namespace det = lsst::afw::detection;

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::Source;
using lsst::afw::detection::SourceSet;
using lsst::afw::detection::PersistableSourceVector;

namespace form = lsst::afw::formatters;

// -- SourceVectorFormatter ----------------

form::SourceVectorFormatter::SourceVectorFormatter(Policy::Ptr const & policy) :
    lsst::daf::persistence::Formatter(typeid(this)),
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
    \internal   Generates a unique identifier for a Source given the ampExposureId of the
                originating amplifier ad the sequence number of the Source within the amplifier.
 */
inline static int64_t generateSourceId(unsigned short seqNum, boost::int64_t ampExposureId) {
    return (ampExposureId << 16) + seqNum;
}

template <typename T, typename F>
inline static void insertFp(T & db, F const & val, char const * const col, bool isNull=false) {
    if (isNull || isnan(val)) {
        db.setColumnToNull(col);
    } else {
        db.template setColumn<F>(col, val);
    }
}

/*!
    Inserts a single Source into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::SourceVectorFormatter::insertRow(T & db, Source const & d)
{
    db.template setColumn<boost::int64_t>("sourceId", d._id);

    if (!d.isNull(det::AMP_EXPOSURE_ID)) {
        db.template setColumn<boost::int64_t>("ampExposureId", d._ampExposureId);    
    } else {
        db.setColumnToNull("ampExposureId");
    }
   
    db.template setColumn<char>("filterId", static_cast<char>(d._filterId));
   
    if (!d.isNull(det::OBJECT_ID)) {
        db.template setColumn<boost::int64_t>("objectId", d._objectId);
    } else {
        db.setColumnToNull("objectId");
    }

    if (!d.isNull(det::MOVING_OBJECT_ID)) {
        db.template setColumn<boost::int64_t>("movingObjectId", d._movingObjectId);
    } else {
        db.setColumnToNull("movingObjectId");
    }

    db.template setColumn<boost::int32_t>("procHistoryID", d._procHistoryId);

    insertFp(db, d._ra, "ra");
    insertFp(db, d._raErrForDetection, "raErrForDetection", d.isNull(det::RA_ERR_FOR_DETECTION));
    insertFp(db, d._raErrForWcs, "raErrForWcs");
    insertFp(db, d._dec, "decl");
    insertFp(db, d._decErrForDetection, "declErrForDetection", d.isNull(det::DEC_ERR_FOR_DETECTION));
    insertFp(db, d._decErrForWcs, "declErrForWcs");

    insertFp(db, d._xFlux, "xFlux", d.isNull(det::X_FLUX));
    insertFp(db, d._xFluxErr, "xFluxErr", d.isNull(det::X_FLUX_ERR));
    insertFp(db, d._yFlux, "yFlux", d.isNull(det::Y_FLUX));
    insertFp(db, d._yFluxErr, "yFluxErr", d.isNull(det::Y_FLUX_ERR));
    insertFp(db, d._raFlux, "raFlux", d.isNull(det::RA_FLUX));
    insertFp(db, d._raFluxErr, "raFluxErr", d.isNull(det::RA_FLUX_ERR));
    insertFp(db, d._decFlux, "declFlux", d.isNull(det::DEC_FLUX));
    insertFp(db, d._decFluxErr, "declFluxErr", d.isNull(det::DEC_FLUX_ERR));

    insertFp(db, d._xPeak, "xPeak", d.isNull(det::X_PEAK));
    insertFp(db, d._yPeak, "yPeak", d.isNull(det::Y_PEAK));
    insertFp(db, d._raPeak, "raPeak", d.isNull(det::RA_PEAK));
    insertFp(db, d._decPeak, "declPeak", d.isNull(det::DEC_PEAK));

    insertFp(db, d._xAstrom, "xAstrom", d.isNull(det::X_ASTROM));
    insertFp(db, d._xAstromErr, "xAstromErr", d.isNull(det::X_ASTROM_ERR));
    insertFp(db, d._yAstrom, "yAstrom", d.isNull(det::Y_ASTROM));
    insertFp(db, d._yAstromErr, "yAstromErr", d.isNull(det::Y_ASTROM_ERR));
    insertFp(db, d._raAstrom, "raAstrom", d.isNull(det::RA_ASTROM));
    insertFp(db, d._raAstromErr, "raAstromErr", d.isNull(det::RA_ASTROM_ERR));
    insertFp(db, d._decAstrom, "declAstrom", d.isNull(det::DEC_ASTROM));
    insertFp(db, d._decAstromErr, "declAstromErr", d.isNull(det::DEC_ASTROM_ERR));

    insertFp(db, d._raObject, "raObject", d.isNull(det::RA_OBJECT));
    insertFp(db, d._decObject, "declObject", d.isNull(det::DEC_OBJECT));

    insertFp(db, d._taiMidPoint, "taiMidPoint"); 
    insertFp(db, d._taiRange, "taiRange", d.isNull(det::TAI_RANGE));
 
    insertFp(db, d._psfFlux, "psfFlux");
    insertFp(db, d._psfFluxErr, "psfFluxErr");
    insertFp(db, d._apFlux, "apFlux");
    insertFp(db, d._apFluxErr, "apFluxErr");
    insertFp(db, d._modelFlux, "modelFlux");
    insertFp(db, d._modelFluxErr, "modelFluxErr");
    insertFp(db, d._petroFlux, "petroFlux", d.isNull(det::PETRO_FLUX));
    insertFp(db, d._petroFluxErr, "petroFluxErr", d.isNull(det::PETRO_FLUX_ERR));
    insertFp(db, d._instFlux, "instFlux");
    insertFp(db, d._instFluxErr, "instFluxErr");
    insertFp(db, d._nonGrayCorrFlux, "nonGrayCorrFlux", d.isNull(det::NON_GRAY_CORR_FLUX));
    insertFp(db, d._nonGrayCorrFluxErr, "nonGrayCorrFluxErr", d.isNull(det::NON_GRAY_CORR_FLUX_ERR));
    insertFp(db, d._atmCorrFlux, "atmCorrFlux", d.isNull(det::ATM_CORR_FLUX));
    insertFp(db, d._atmCorrFluxErr, "atmCorrFluxErr", d.isNull(det::ATM_CORR_FLUX_ERR));

    insertFp(db, d._apDia, "apDia", d.isNull(det::AP_DIA));

    insertFp(db, d._ixx, "Ixx", d.isNull(det::IXX));
    insertFp(db, d._ixxErr, "IxxErr", d.isNull(det::IXX_ERR));
    insertFp(db, d._iyy, "Iyy", d.isNull(det::IYY));
    insertFp(db, d._iyyErr, "IyyErr", d.isNull(det::IYY_ERR));
    insertFp(db, d._ixy, "Ixy", d.isNull(det::IXY));
    insertFp(db, d._ixyErr, "IxyErr", d.isNull(det::IXY_ERR));

    insertFp(db, d._snr, "snr");
    insertFp(db, d._chi2, "chi2");

    insertFp(db, d._sky, "sky", d.isNull(det::SKY));
    insertFp(db, d._skyErr, "skyErr", d.isNull(det::SKY_ERR));
 
    if (!d.isNull(det::FLAG_FOR_ASSOCIATION)) {
        db.template setColumn<boost::int16_t>("flagForAssociation", d._flagForAssociation);
    } else {
        db.setColumnToNull("flagForAssociation");
    } 
    if (!d.isNull(det::FLAG_FOR_DETECTION)) {
        db.template setColumn<boost::int64_t>("flagForDetection", d._flagForDetection);
    } else {
        db.setColumnToNull("flagForDetection");
    } 
    if (!d.isNull(det::FLAG_FOR_WCS)) {
        db.template setColumn<boost::int16_t>("flagForWcs", d._flagForWcs);
    } else {
        db.setColumnToNull("flagForWcs");
    }

    db.insertRow();
}

//! \cond
template void form::SourceVectorFormatter::insertRow<DbStorage>   (DbStorage & db,    Source const &d);
template void form::SourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage & db, Source const &d);
//! \endcond


/*! Prepares for reading Source instances from a database table. */
void form::SourceVectorFormatter::setupFetch(DbStorage & db, Source & d) {
    db.outParam("sourceId",           &(d._id));
    db.outParam("ampExposureId",      &(d._ampExposureId));
    db.outParam("filterId",           reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",           &(d._objectId));
    db.outParam("movingObjectId",     &(d._movingObjectId));
    db.outParam("procHistoryId",      &(d._procHistoryId));
    db.outParam("ra",                 &(d._ra));
    db.outParam("raErrForDetection",  &(d._raErrForDetection));
    db.outParam("raErrForWcs",        &(d._raErrForWcs));
    db.outParam("decl",               &(d._dec));
    db.outParam("declErrForDetection",&(d._decErrForDetection));
    db.outParam("declErrForWcs",      &(d._decErrForWcs));
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
    db.outParam("xAstrom",            &(d._xAstrom));
    db.outParam("xAstromErr",         &(d._xAstromErr));    
    db.outParam("yAstrom",            &(d._yAstrom));
    db.outParam("yAstromErr",         &(d._yAstromErr));  
    db.outParam("raAstrom",           &(d._raAstrom));
    db.outParam("raAstromErr",        &(d._raAstromErr));    
    db.outParam("declAstrom",         &(d._decAstrom));
    db.outParam("declAstromErr",      &(d._decAstromErr));
    db.outParam("raObject",           &(d._raObject));
    db.outParam("declObject",         &(d._decObject));
    db.outParam("taiMidPoint",        &(d._taiMidPoint));
    db.outParam("taiRange",           &(d._taiRange));
    db.outParam("psfFlux",            &(d._psfFlux));
    db.outParam("psfFluxErr",         &(d._psfFluxErr));
    db.outParam("apFlux",             &(d._apFlux));
    db.outParam("apFluxErr",          &(d._apFluxErr));
    db.outParam("modelFlux",          &(d._modelFlux));
    db.outParam("modelFluxErr",       &(d._modelFluxErr));
    db.outParam("petroFlux",          &(d._petroFlux));
    db.outParam("petroFluxErr",       &(d._petroFluxErr));
    db.outParam("instFlux",           &(d._instFlux));
    db.outParam("instFluxErr",        &(d._instFluxErr));
    db.outParam("nonGrayCorrFlux",    &(d._nonGrayCorrFlux));
    db.outParam("nonGrayCorrFluxErr", &(d._nonGrayCorrFluxErr));    
    db.outParam("atmCorrFlux",        &(d._atmCorrFlux));
    db.outParam("atmCorrFluxErr",     &(d._atmCorrFluxErr));     
    db.outParam("apDia",              &(d._apDia));    
    db.outParam("Ixx",                &(d._ixx));
    db.outParam("IxxErr",             &(d._ixxErr));
    db.outParam("Iyy",                &(d._iyy));
    db.outParam("IyyErr",             &(d._iyyErr));
    db.outParam("Ixy",                &(d._ixy));
    db.outParam("IxyErr",             &(d._ixyErr)); 
    db.outParam("snr",                &(d._snr));
    db.outParam("chi2",               &(d._chi2));
    db.outParam("sky",                &(d._sky));
    db.outParam("skyErr",             &(d._skyErr));    
    db.outParam("flagForAssociation", &(d._flagForAssociation));
    db.outParam("flagForDetection",   &(d._flagForDetection));
    db.outParam("flagForWcs",         &(d._flagForWcs));
}


template <class Archive>
void form::SourceVectorFormatter::delegateSerialize(
    Archive & archive,
    unsigned int const,
    Persistable * persistable
) {  
    PersistableSourceVector * p = dynamic_cast<PersistableSourceVector*>(persistable);

    archive & boost::serialization::base_object<Persistable>(*p);
    
    SourceSet::size_type sz;

    if (Archive::is_loading::value) {        
        Source data;        
        archive & sz;
        p->_sources.clear();
        p->_sources.reserve(sz);
        for (; sz > 0; --sz) {
            archive & data;
            Source::Ptr sourcePtr(new Source(data));
            p->_sources.push_back(sourcePtr);
        }
    } else {
        sz = p->_sources.size();
        archive & sz;
        SourceSet::iterator i = p->_sources.begin();
        SourceSet::iterator const end(p->_sources.end());
        for ( ; i != end; ++i) {
            archive &  **i;
        }
    }
}


template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_oarchive>(
    boost::archive::text_oarchive &, unsigned int const, Persistable *
);
template void form::SourceVectorFormatter::delegateSerialize<boost::archive::text_iarchive>(
    boost::archive::text_iarchive &, unsigned int const, Persistable *
);

/** 
 * Persist a collection of Source to BoostStorage, DbStorage or DbTsvStorage
 */
void form::SourceVectorFormatter::write(
    Persistable const * persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {
    if (persistable == 0) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Persistable provided");
    }
    if (!storage) {
        throw LSST_EXCEPT(ex::InvalidParameterException, "No Storage provided");
    }

    PersistableSourceVector const * p = dynamic_cast<PersistableSourceVector const *>(persistable);
    if (p == 0) {
        throw LSST_EXCEPT(ex::RuntimeErrorException, 
                "Persistable was not of concrete type SourceVector");
    }
    SourceSet sourceVector = p->getSources();   

    if (typeid(*storage) == typeid(BoostStorage)) {
        //persist to BoostStorage    
        BoostStorage * bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) {
            throw LSST_EXCEPT(ex::RuntimeErrorException, 
                    "Didn't get BoostStorage");
        }

        //call serializeDelegate
        bs->getOArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) 
            || typeid(*storage) == typeid(DbTsvStorage)) {

        // Assume all have ids or none do.
        // If none do, assign them ids.
        if ((*sourceVector.begin())->getId() == 0 && 
            (!_policy || !_policy->exists("GenerateIds") 
            || _policy->getBool("GenerateIds"))
        ) {
     
            unsigned short seq = 1;
            boost::int64_t ampExposureId = extractAmpExposureId(additionalData);
            if (sourceVector.size() > 65536) {
                throw LSST_EXCEPT(ex::RangeErrorException, "too many Sources per-amp: "
                    "sequence number overflows 16 bits, potentially causing unique-id conflicts");
            }
            
            SourceSet::iterator i = sourceVector.begin();
            for ( ; i != sourceVector.end(); ++i) {
                (*i)->setId(generateSourceId(seq, ampExposureId));
                (*i)->setAmpExposureId(ampExposureId);
                ++seq;
                if (seq == 0) { // Overflowed
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "Too many Sources");
                }
            }        
        }

        std::string itemName(getItemName(additionalData));
        std::string name(getTableName(_policy, additionalData));
        std::string model = _policy->getString(itemName + ".templateTableName");

        if (typeid(*storage) == typeid(DbStorage)) {
            //handle persisting to DbStorage
            DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
            if (db == 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException, 
                        "Didn't get DbStorage");
            }

            db->createTableFromTemplate(name, model, true);
            db->setTableForInsert(name);
            
            SourceSet::const_iterator i(sourceVector.begin());
            SourceSet::const_iterator const end(sourceVector.end());
            for ( ; i != end; ++i) {
                insertRow<DbStorage>(*db, **i);
            }
        } else {
            //handle persisting to DbTsvStorage
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException, 
                        "Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, true);
            db->setTableForInsert(name);

            SourceSet::const_iterator i(sourceVector.begin());
            SourceSet::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, **i);
            }
        }
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                "Storage type is not supported"); 
    }
}

template <typename T>
static inline void handleNullFp(DbStorage * db, int col, T & val) {
    if (db->columnIsNull(col)) {
        val = std::numeric_limits<T>::quiet_NaN();
    }
}
template <typename T>
static inline void handleNullFp(DbStorage * db, Source & src, int col, T & val, int flag) {
    if (db->columnIsNull(col)) {
        src.setNull(flag);
        val = std::numeric_limits<T>::quiet_NaN();
    }           
}


/** 
 * Retrieve a collection of Source from BoostStorage, DbStorage or DbTsvStorage
 */
Persistable* form::SourceVectorFormatter::read(
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {   
    std::auto_ptr<PersistableSourceVector> p(new PersistableSourceVector);
    
    if (typeid(*storage) == typeid(BoostStorage)) {
        //handle retrieval from BoostStorage
        BoostStorage* bs = dynamic_cast<BoostStorage *>(storage.get());
        if (bs == 0) { 
            throw LSST_EXCEPT(ex::RuntimeErrorException, 
                    "Didn't get BoostStorage");
        }
        //calls serializeDelegate
        bs->getIArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) 
            || typeid(*storage) == typeid(DbTsvStorage)) {
        //handle retrieval from DbStorage, DbTsvStorage    
        DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
        if (db == 0) {  
            throw LSST_EXCEPT(ex::RuntimeErrorException, 
                    "Didn't get DbStorage");
        }
        //get a list of tables from policy and additionalData
        std::vector<std::string> tables = getAllSliceTableNames(_policy, additionalData);

        SourceSet sourceVector;
        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator i;
        std::vector<std::string>::const_iterator const end = tables.end();
        for (i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            Source data;
            
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
                if (db->columnIsNull(SOURCE_ID)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"sourceId\""); 
                }
                if (db->columnIsNull(AMP_EXPOSURE_ID)) { 
                    data.setNull(det::AMP_EXPOSURE_ID); 
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
                if (db->columnIsNull(PROC_HISTORY_ID)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"procHistoryId\""); 
                }
                handleNullFp(db, RA, data._ra);
                handleNullFp(db, DECL, data._dec);
                handleNullFp(db, RA_ERR_FOR_WCS, data._raErrForWcs);
                handleNullFp(db, DEC_ERR_FOR_WCS, data._decErrForWcs);
                handleNullFp(db, data, RA_ERR_FOR_DETECTION,
                             data._raErrForDetection, det::RA_ERR_FOR_DETECTION);
                handleNullFp(db, data, DEC_ERR_FOR_DETECTION,
                             data._decErrForDetection, det::DEC_ERR_FOR_DETECTION);

                handleNullFp(db, data, X_FLUX, data._xFlux, det::X_FLUX);
                handleNullFp(db, data, X_FLUX_ERR, data._xFluxErr, det::X_FLUX_ERR);
                handleNullFp(db, data, Y_FLUX, data._yFlux, det::Y_FLUX);
                handleNullFp(db, data, Y_FLUX_ERR, data._yFluxErr, det::Y_FLUX_ERR);
                handleNullFp(db, data, RA_FLUX, data._raFlux, det::RA_FLUX);
                handleNullFp(db, data, RA_FLUX_ERR, data._raFluxErr, det::RA_FLUX_ERR);
                handleNullFp(db, data, DEC_FLUX, data._decFlux, det::DEC_FLUX);
                handleNullFp(db, data, DEC_FLUX_ERR, data._decFluxErr, det::DEC_FLUX_ERR);

                handleNullFp(db, data, X_PEAK, data._xPeak, det::X_PEAK);
                handleNullFp(db, data, Y_PEAK, data._yPeak, det::Y_PEAK);
                handleNullFp(db, data, RA_PEAK, data._raPeak, det::RA_PEAK);
                handleNullFp(db, data, DEC_PEAK, data._decPeak, det::DEC_PEAK);

                handleNullFp(db, data, X_ASTROM, data._xAstrom, det::X_ASTROM);
                handleNullFp(db, data, X_ASTROM_ERR, data._xAstromErr, det::X_ASTROM_ERR);
                handleNullFp(db, data, Y_ASTROM, data._yAstrom, det::Y_ASTROM);
                handleNullFp(db, data, Y_ASTROM_ERR, data._yAstromErr, det::Y_ASTROM_ERR);
                handleNullFp(db, data, RA_ASTROM, data._raAstrom, det::RA_ASTROM);
                handleNullFp(db, data, RA_ASTROM_ERR, data._raAstromErr, det::RA_ASTROM_ERR);
                handleNullFp(db, data, DEC_ASTROM, data._decAstrom, det::DEC_ASTROM);
                handleNullFp(db, data, DEC_ASTROM_ERR, data._decAstromErr, det::DEC_ASTROM_ERR);

                handleNullFp(db, data, RA_OBJECT, data._raObject, det::RA_OBJECT);
                handleNullFp(db, data, DEC_OBJECT, data._decObject, det::DEC_OBJECT);

                handleNullFp(db, TAI_MID_POINT, data._taiMidPoint);
                handleNullFp(db, data, TAI_RANGE, data._taiRange, det::TAI_RANGE);

                handleNullFp(db, PSF_FLUX, data._psfFlux);
                handleNullFp(db, PSF_FLUX_ERR, data._psfFluxErr);
                handleNullFp(db, AP_FLUX, data._apFlux);
                handleNullFp(db, AP_FLUX_ERR, data._apFluxErr);
                handleNullFp(db, MODEL_FLUX, data._modelFlux);
                handleNullFp(db, MODEL_FLUX_ERR, data._modelFluxErr);
                handleNullFp(db, data, PETRO_FLUX, data._petroFlux, det::PETRO_FLUX);
                handleNullFp(db, data, PETRO_FLUX_ERR, data._petroFluxErr, det::PETRO_FLUX_ERR);
                handleNullFp(db, INST_FLUX, data._instFlux);
                handleNullFp(db, INST_FLUX_ERR, data._instFluxErr);
                handleNullFp(db, data, NON_GRAY_CORR_FLUX,
                             data._nonGrayCorrFlux, det::NON_GRAY_CORR_FLUX);
                handleNullFp(db, data, NON_GRAY_CORR_FLUX_ERR,
                             data._nonGrayCorrFluxErr, det::NON_GRAY_CORR_FLUX_ERR);
                handleNullFp(db, data, ATM_CORR_FLUX,
                             data._atmCorrFlux, det::ATM_CORR_FLUX);
                handleNullFp(db, data, ATM_CORR_FLUX_ERR,
                             data._atmCorrFluxErr, det::ATM_CORR_FLUX_ERR);

                handleNullFp(db, data, AP_DIA, data._apDia, det::AP_DIA);

                handleNullFp(db, data, IXX, data._ixx, det::IXX);
                handleNullFp(db, data, IXX_ERR, data._ixxErr, det::IXX_ERR);
                handleNullFp(db, data, IYY, data._iyy, det::IYY);
                handleNullFp(db, data, IYY_ERR, data._iyyErr, det::IYY_ERR);
                handleNullFp(db, data, IXY, data._ixy, det::IXY);
                handleNullFp(db, data, IXY_ERR, data._ixyErr, det::IXY_ERR);

                handleNullFp(db, SNR, data._snr);
                handleNullFp(db, CHI2, data._chi2);
                handleNullFp(db, data, SKY, data._sky, det::SKY);
                handleNullFp(db, data, SKY_ERR, data._skyErr, det::SKY_ERR);

                if (db->columnIsNull(FLAG_FOR_ASSOCIATION)) { 
                    data.setNull(det::FLAG_FOR_ASSOCIATION);
                }
                if (db->columnIsNull(FLAG_FOR_DETECTION)) { 
                    data.setNull(det::FLAG_FOR_DETECTION); 
                }
                if (db->columnIsNull(FLAG_FOR_WCS)) { 
                    data.setNull(det::FLAG_FOR_WCS); 
                }
                                                
                //add source to vector
                Source::Ptr sourcePtr(new Source(data));
                sourceVector.push_back(sourcePtr);
                
                //reset nulls for next source
                data.setNotNull();                
            }
            db->finishQuery();
        }
        p->setSources(sourceVector);
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, 
                "Storage type is not supported");
    }
    return p.release();
}


void form::SourceVectorFormatter::update(Persistable*, 
    Storage::Ptr, lsst::daf::base::PropertySet::Ptr
) {
    throw LSST_EXCEPT(ex::RuntimeErrorException, 
            "SourceVectorFormatter: updates not supported");
}
