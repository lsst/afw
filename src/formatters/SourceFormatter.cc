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
    \internal   Generates a unique identifier for a Source given the ampExposureId of the
                originating amplifier ad the sequence number of the Source within the amplifier.
 */
inline static int64_t generateSourceId(unsigned short seqNum, boost::int64_t ampExposureId) {
    return (ampExposureId << 16) + seqNum;
}


/*!
    Inserts a single Source into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::SourceVectorFormatter::insertRow(T & db, Source const & d) {

    db.template setColumn<boost::int64_t>("sourceId", d._id);

    if (!d.isNull(det::AMP_EXPOSURE_ID))
        db.template setColumn<boost::int64_t>("ampExposureId", d._ampExposureId);    
    else db.setColumnToNull("ampExposureId");
   
    db.template setColumn<char>("filterId", static_cast<char>(d._filterId));
   
    if (!d.isNull(det::OBJECT_ID))
        db. template setColumn<boost::int64_t>("objectId", d._objectId);
    else db.setColumnToNull("objectId");
    
    if (!d.isNull(det::MOVING_OBJECT_ID))
        db. template setColumn<boost::int64_t>("movingObjectId", d._movingObjectId);
    else db.setColumnToNull("movingObjectId");
   
    db.template setColumn<boost::int32_t>("procHistoryID", d._procHistoryId);

    db.template setColumn<double>("ra", d._ra);
    
    if (!d.isNull(det::RA_ERR_FOR_DETECTION))
        db.template setColumn<float>("raErrForDetection", d._raErrForDetection);
    else db.setColumnToNull("raErrForDetection");
    
    db. template setColumn<float>("raErrForWcs", d._raErrForWcs);    
    db.template setColumn<double>("decl", d._dec);
    
    if (!d.isNull(det::DEC_ERR_FOR_DETECTION)) {
        db.template setColumn<float>("declErrForDetection", 
                d._decErrForDetection);
    }
    else db.setColumnToNull("declErrForDetection");
    
    db. template setColumn<float>("declErrForWcs", d._decErrForWcs);    

    if (!d.isNull(det::X_FLUX))
        db. template setColumn<double>("xFlux", d._xFlux);
    else db.setColumnToNull("xFlux");

    if (!d.isNull(det::X_FLUX_ERR))    
        db. template setColumn<float>("xFluxErr", d._xFluxErr);
    else db.setColumnToNull("xFluxErr");
    
    if (!d.isNull(det::Y_FLUX))
        db. template setColumn<double>("yFlux", d._yFlux);
    else db.setColumnToNull("yFlux");
    
    if (!d.isNull(det::Y_FLUX_ERR))
        db. template setColumn<float>("yFluxErr", d._yFluxErr);
    else db.setColumnToNull("yFluxErr");
    
    if (!d.isNull(det::RA_FLUX))
        db. template setColumn<double>("raFlux", d._raFlux);
    else db.setColumnToNull("raFlux");
    
    if (!d.isNull(det::RA_FLUX_ERR))
        db. template setColumn<float>("raFluxErr", d._raFluxErr);
    else db.setColumnToNull("raFluxErr");
    
    if (!d.isNull(det::DEC_FLUX))
        db. template setColumn<double>("declFlux", d._decFlux);
    else db.setColumnToNull("declFlux");
    
    if (!d.isNull(det::DEC_FLUX_ERR))
        db. template setColumn<float>("declFluxErr", d._decFluxErr);
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
        db. template setColumn("declPeak", d._decPeak);
    else db.setColumnToNull("declPeak");
    
    if (!d.isNull(det::X_ASTROM))
        db. template setColumn<double>("xAstrom", d._xAstrom);
    else db.setColumnToNull("xAstrom");
    
    if (!d.isNull(det::X_ASTROM_ERR))
        db. template setColumn<float>("xAstromErr", d._xAstromErr);
    else db.setColumnToNull("xAstromErr");
    
    if (!d.isNull(det::Y_ASTROM))
        db. template setColumn<double>("yAstrom", d._yAstrom);
    else db.setColumnToNull("yAstrom");
    
    if (!d.isNull(det::Y_ASTROM_ERR))
        db. template setColumn<float>("yAstromErr", d._yAstromErr);
    else db.setColumnToNull("yAstromErr");
    
    if (!d.isNull(det::RA_ASTROM))
        db. template setColumn<double>("raAstrom", d._raAstrom);
    else db.setColumnToNull("raAstrom");
    
    if (!d.isNull(det::RA_ASTROM_ERR))
        db. template setColumn<float>("raAstromErr", d._raAstromErr);
    else db.setColumnToNull("raAstromErr");
    
    if (!d.isNull(det::DEC_ASTROM))
        db. template setColumn<double>("declAstrom", d._decAstrom);
    else db.setColumnToNull("declAstrom");
    
    if (!d.isNull(det::DEC_ASTROM_ERR))
        db. template setColumn<float>("declAstromErr", d._decAstromErr);        
    else db.setColumnToNull("declAstromErr");
  
    db.template setColumn<double>("taiMidPoint", d._taiMidPoint);
    
    if (!d.isNull(det::TAI_RANGE))
        db.template setColumn<double>("taiRange", d._taiRange);
    else db.setColumnToNull("taiRange");
    
    db.template setColumn<double>("psfFlux", d._psfFlux);
    db.template setColumn<float>("psfFluxErr", d._psfFluxErr);
    db.template setColumn<double>("apFlux", d._apFlux);
    db.template setColumn<float>("apFluxErr", d._apFluxErr);
    db.template setColumn<double>("modelFlux", d._modelFlux);            
    db.template setColumn<float>("modelFluxErr", d._modelFluxErr);   

    if (!d.isNull(det::PETRO_FLUX))
        db. template setColumn<double>("petroFlux", d._petroFlux);
    else db.setColumnToNull("petroFlux");
    
    if (!d.isNull(det::PETRO_FLUX_ERR))
        db. template setColumn<float>("petroFluxErr", d._petroFluxErr);  
    else db.setColumnToNull("petroFluxErr");    
    
    db.template setColumn<double>("instFlux", d._instFlux);
    db.template setColumn<float>("instFluxErr", d._instFluxErr);
    
    if (!d.isNull(det::NON_GRAY_CORR_FLUX))
        db.template setColumn<double>("nonGrayCorrFlux", d._nonGrayCorrFlux);
    else db.setColumnToNull("nonGrayCorrFlux");
    
    if (!d.isNull(det::NON_GRAY_CORR_FLUX_ERR)) {
        db.template setColumn<float>("nonGrayCorrFluxErr", 
                d._nonGrayCorrFluxErr);
    }
    else db.setColumnToNull("nonGrayCorrFluxErr");
        
    if (!d.isNull(det::ATM_CORR_FLUX))
        db.template setColumn<double>("atmCorrFlux", d._atmCorrFlux);
    else db.setColumnToNull("atmCorrFlux");
    
    if (!d.isNull(det::ATM_CORR_FLUX_ERR))
        db.template setColumn<float>("atmCorrFluxErr", d._atmCorrFluxErr);
    else db.setColumnToNull("atmCorrFluxErr");
    
    if (!d.isNull(det::AP_DIA))
        db.template setColumn<float>("apDia", d._apDia);
    else db.setColumnToNull("apDia");
   
    if (!d.isNull(det::IXX))
        db.template setColumn<float>("Ixx", d._ixx);
    else db.setColumnToNull("Ixx");
    
    if (!d.isNull(det::IXX_ERR))
        db.template setColumn<float>("IxxErr", d._ixxErr);
    else db.setColumnToNull("IxxErr");
    
    if (!d.isNull(det::IYY))    
        db.template setColumn<float>("Iyy", d._iyy);
    else db.setColumnToNull("Iyy");
    
    if (!d.isNull(det::IYY_ERR))
        db.template setColumn<float>("IyyErr", d._iyyErr);
    else db.setColumnToNull("IyyErr");
    
    if (!d.isNull(det::IXY))
        db.template setColumn<float>("Ixy", d._ixy);
    else db.setColumnToNull("Ixy");
    
    if (!d.isNull(det::IXY_ERR))
        db.template setColumn<float>("IxyErr", d._ixyErr);        
    else db.setColumnToNull("IxyErr");

    db.template setColumn<float>("snr", d._snr);
    db.template setColumn<float>("chi2", d._chi2);
    
    if (!d.isNull(det::SKY))
        db.template setColumn<float>("sky", d._sky);
    else db.setColumnToNull("sky");
        
    if (!d.isNull(det::SKY_ERR))
        db.template setColumn<float>("skyErr", d._skyErr);
    else db.setColumnToNull("skyErr");      
        
    if (!d.isNull(det::FLAG_FOR_ASSOCIATION))
        db.template setColumn<boost::int16_t>("flagForAssociation", d._flagForAssociation);
    else db.setColumnToNull("flagForAssociation");
    
    if (!d.isNull(det::FLAG_FOR_DETECTION))
        db.template setColumn<boost::int64_t>("flagForDetection", d._flagForDetection);
    else db.setColumnToNull("flagForDetection");
    
    if (!d.isNull(det::FLAG_FOR_WCS))
        db.template setColumn<boost::int16_t>("flagForWcs", d._flagForWcs);
    else db.setColumnToNull("flagForWcs");

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
    unsigned int const version,
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

    // Assume all have ids or none do.
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
        std::string itemName(getItemName(additionalData));
        std::string name(getTableName(_policy, additionalData));
        std::string model = _policy->getString(itemName + ".templateTableName");

        bool mayExist = !extractOptionalFlag(
            additionalData, 
            itemName + ".isPerSliceTable");

        if (typeid(*storage) == typeid(DbStorage)) {
        	//handle persisting to DbStorag
            DbStorage * db = dynamic_cast<DbStorage *>(storage.get());
            if (db == 0) {
                throw LSST_EXCEPT(ex::RuntimeErrorException, 
                        "Didn't get DbStorage");
            }

            db->createTableFromTemplate(name, model, mayExist);
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
            db->createTableFromTemplate(name, model, mayExist);
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
                if (db->columnIsNull(RA)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"ra\""); 
                }
                if (db->columnIsNull(DECL)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"decl\""); 
                }
                if (db->columnIsNull(RA_ERR_FOR_WCS)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"raErrForWcs\""); 
                }
                if (db->columnIsNull(DEC_ERR_FOR_WCS)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"declErrForWcs\""); 
                }
                if (db->columnIsNull(RA_ERR_FOR_DETECTION)) { 
                    data.setNull(det::RA_ERR_FOR_DETECTION); 
                }
                if (db->columnIsNull(DEC_ERR_FOR_DETECTION)) { 
                    data.setNull(det::DEC_ERR_FOR_DETECTION); 
                }
                if (db->columnIsNull(X_FLUX)) { data.setNull(det::X_FLUX); }
                if (db->columnIsNull(X_FLUX_ERR)) { 
                    data.setNull(det::X_FLUX_ERR); 
                }
                if (db->columnIsNull(Y_FLUX)) { data.setNull(det::Y_FLUX); }
                if (db->columnIsNull(Y_FLUX_ERR)) { 
                    data.setNull(det::Y_FLUX_ERR); 
                }
                if (db->columnIsNull(RA_FLUX)) { data.setNull(det::RA_FLUX); }
                if (db->columnIsNull(RA_FLUX_ERR)) { 
                    data.setNull(det::RA_FLUX_ERR); 
                }
                if (db->columnIsNull(DEC_FLUX)) { data.setNull(det::DEC_FLUX); }
                if (db->columnIsNull(DEC_FLUX_ERR)) { 
                    data.setNull(det::DEC_FLUX_ERR); 
                }
                if (db->columnIsNull(X_PEAK)) { data.setNull(det::X_PEAK); }
                if (db->columnIsNull(Y_PEAK)) { data.setNull(det::Y_PEAK); }
                if (db->columnIsNull(RA_PEAK)) { data.setNull(det::RA_PEAK); }
                if (db->columnIsNull(DEC_PEAK)) { data.setNull(det::DEC_PEAK); }
                if (db->columnIsNull(X_ASTROM)) { data.setNull(det::X_ASTROM); }
                if (db->columnIsNull(X_ASTROM_ERR)) { 
                    data.setNull(det::X_ASTROM_ERR); 
                }
                if (db->columnIsNull(Y_ASTROM)) { data.setNull(det::Y_ASTROM); }
                if (db->columnIsNull(Y_ASTROM_ERR)) { 
                    data.setNull(det::Y_ASTROM_ERR); 
                }
                if (db->columnIsNull(RA_ASTROM)) { 
                    data.setNull(det::RA_ASTROM); 
                }
                if (db->columnIsNull(RA_ASTROM_ERR)) { 
                    data.setNull(det::RA_ASTROM_ERR); 
                }
                if (db->columnIsNull(DEC_ASTROM)) { 
                    data.setNull(det::DEC_ASTROM); 
                }
                if (db->columnIsNull(DEC_ASTROM_ERR)) { 
                    data.setNull(det::DEC_ASTROM_ERR); 
                }
                if (db->columnIsNull(TAI_MID_POINT)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"taiMidPoint\""); 
                }
                if (db->columnIsNull(TAI_RANGE)) { 
                    data.setNull(det::TAI_RANGE); 
                }
                if (db->columnIsNull(PSF_FLUX)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"psfFlux\"");           
                }
                if (db->columnIsNull(PSF_FLUX_ERR)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"psfFluxErr\"");        
                }
                if (db->columnIsNull(AP_FLUX)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"apFlux\"");  
                }
                if (db->columnIsNull(AP_FLUX_ERR)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"apFluxErr\"");  
                }
                if (db->columnIsNull(MODEL_FLUX)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"modelFlux\""); 
                }
                if (db->columnIsNull(MODEL_FLUX_ERR)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"modelFluxErr\""); 
                }
                if (db->columnIsNull(PETRO_FLUX)) { 
                    data.setNull(det::PETRO_FLUX); 
                }
                if (db->columnIsNull(PETRO_FLUX_ERR)) { 
                    data.setNull(det::PETRO_FLUX_ERR); 
                }    
                if (db->columnIsNull(INST_FLUX)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"instFlux\""); 
                }
                if (db->columnIsNull(INST_FLUX_ERR)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"instFluxErr\"");
                }
                if (db->columnIsNull(NON_GRAY_CORR_FLUX)) { 
                    data.setNull(det::NON_GRAY_CORR_FLUX); 
                }
                if (db->columnIsNull(NON_GRAY_CORR_FLUX_ERR)) { 
                    data.setNull(det::NON_GRAY_CORR_FLUX_ERR); 
                }
                if (db->columnIsNull(ATM_CORR_FLUX)) { 
                    data.setNull(det::ATM_CORR_FLUX); 
                }
                if (db->columnIsNull(ATM_CORR_FLUX_ERR)) { 
                    data.setNull(det::ATM_CORR_FLUX_ERR); 
                }
                if (db->columnIsNull(AP_DIA)) { data.setNull(det::AP_DIA); }
                if (db->columnIsNull(IXX)) { data.setNull(det::IXX);}
                if (db->columnIsNull(IXX_ERR)) { data.setNull(det::IXX_ERR);}
                if (db->columnIsNull(IYY)) { data.setNull(det::IYY); }
                if (db->columnIsNull(IYY_ERR)) { data.setNull(det::IYY_ERR);}
                if (db->columnIsNull(IXY)) { data.setNull(det::IXY);}
                if (db->columnIsNull(IXY_ERR)) { data.setNull(det::IXY_ERR); }

                if (db->columnIsNull(SNR)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"snr\""); 
                }
                if (db->columnIsNull(CHI2)) {  
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"chi2\"");
                }
                if (db->columnIsNull(SKY)) { 
                    data.setNull(det::SKY); }
                if (db->columnIsNull(SKY_ERR)) { 
                    data.setNull(det::SKY_ERR); 
                } 
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
