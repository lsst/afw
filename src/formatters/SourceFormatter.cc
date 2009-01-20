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
namespace detail = lsst::afw::detection::source_detail;

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::Source;
using lsst::afw::detection::SourceVector;
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
    db.template setColumn<int64_t>("sourceId", d._id);
    if(!d.isNull(detail::AMP_EXPOSURE_ID))
	    db.template setColumn<int64_t>("ampExposureId", d._ampExposureId);    
   	else db.setColumnToNull("ampExposureId");
   	
    db.template setColumn<char>("filterId", static_cast<char>(d._filterId));
    
    if(!d.isNull(detail::OBJECT_ID))
    	db. template setColumn<int64_t>("objectId", d._objectId);
   	else db.setColumnToNull("objectId");
   	
    if(!d.isNull(detail::MOVING_OBJECT_ID))
    	db. template setColumn<int64_t>("movingObjectId", d._movingObjectId);
   	else db.setColumnToNull("movingObjectId");
   	
    db.template setColumn<int32_t>("procHistoryID", d._procHistoryId);
   	
    db.template setColumn<double>("ra", d._ra);
    
    if(!d.isNull(detail::RA_ERR_4_DETECTION))
	    db.template setColumn<float>("raErr4detection", d._raErr4detection);
	else db.setColumnToNull("raErr4detection");
	
    db. template setColumn<float>("raErr4wcs", d._raErr4wcs);    
    db.template setColumn<double>("decl", d._dec);
	
	if(!d.isNull(detail::DEC_ERR_4_DETECTION))
	    db.template setColumn<float>("decErr4detection", d._decErr4detection);
    else db.setColumnToNull("decErr4detection");
    
    db. template setColumn<float>("decErr4wcs", d._decErr4wcs);    

    if(!d.isNull(detail::X_FLUX))
    	db. template setColumn<double>("xFlux", d._xFlux);
	else db.setColumnToNull("xFlux");

    if(!d.isNull(detail::X_FLUX_ERR))	
    	db. template setColumn<double>("xFluxErr", d._xFluxErr);
    else db.setColumnToNull("xFluxErr");
    
    if(!d.isNull(detail::Y_FLUX))
    	db. template setColumn<double>("yFlux", d._yFlux);
	else db.setColumnToNull("yFlux");
	
    if(!d.isNull(detail::Y_FLUX_ERR))
     	db. template setColumn<double>("yFluxErr", d._yFluxErr);
	else db.setColumnToNull("yFluxErr");
	
    if(!d.isNull(detail::RA_FLUX))
        db. template setColumn<double>("raFlux", d._raFlux);
    else db.setColumnToNull("raFlux");
    
    if(!d.isNull(detail::RA_FLUX_ERR))
        db. template setColumn<double>("raFluxErr", d._raFluxErr);
    else db.setColumnToNull("raFluxErr");
    
    if(!d.isNull(detail::DEC_FLUX))
        db. template setColumn<double>("decFlux", d._decFlux);
    else db.setColumnToNull("decFlux");
    
    if(!d.isNull(detail::DEC_FLUX_ERR))
        db. template setColumn<double>("decFluxErr", d._decFluxErr);
    else db.setColumnToNull("decFluxErr");
    
    if(!d.isNull(detail::X_PEAK))
        db. template setColumn<double>("xPeak", d._xPeak);
    else db.setColumnToNull("xPeak");
    
    if(!d.isNull(detail::Y_PEAK))
        db. template setColumn<double>("yPeak", d._yPeak);
    else db.setColumnToNull("yPeak");
    
    if(!d.isNull(detail::RA_PEAK))
        db. template setColumn<double>("raPeak", d._raPeak);
    else db.setColumnToNull("raPeak");
        
    if(!d.isNull(detail::DEC_PEAK))    
    	db. template setColumn("decPeak", d._decPeak);
    else db.setColumnToNull("decPeak");
    
    if(!d.isNull(detail::X_ASTROM))
        db. template setColumn<double>("xAstrom", d._xAstrom);
    else db.setColumnToNull("xAstrom");
    
    if(!d.isNull(detail::X_ASTROM_ERR))
        db. template setColumn<double>("xAstromErr", d._xAstromErr);
    else db.setColumnToNull("xAstromErr");
    
    if(!d.isNull(detail::Y_ASTROM))
        db. template setColumn<double>("yAstrom", d._yAstrom);
    else db.setColumnToNull("yAstrom");
    
    if(!d.isNull(detail::Y_ASTROM_ERR))
        db. template setColumn("yAstromErr", d._yAstromErr);
    else db.setColumnToNull("yAstromErr");
    
    if(!d.isNull(detail::RA_ASTROM))
        db. template setColumn<double>("raAstrom", d._raAstrom);
    else db.setColumnToNull("raAstrom");
    
    if(!d.isNull(detail::RA_ASTROM_ERR))
        db. template setColumn<double>("raAstromErr", d._raAstromErr);
    else db.setColumnToNull("raAstromErr");
    
    if(!d.isNull(detail::DEC_ASTROM))
        db. template setColumn<double>("decAstrom", d._decAstrom);
    else db.setColumnToNull("decAstrom");
    
    if(!d.isNull(detail::DEC_ASTROM_ERR))
        db. template setColumn<double>("decAstromErr", d._decAstromErr);        
    else db.setColumnToNull("decAstromErr");
    
    db.template setColumn<double>("taiMidPoint", d._taiMidPoint);
    
    if(!d.isNull(detail::TAI_RANGE))
	    db.template setColumn<float>("taiRange", d._taiRange);
	else db.setColumnToNull("taiRange");
	
    db.template setColumn<float>("fwhmA", d._fwhmA);
    db.template setColumn<float>("fwhmB", d._fwhmB);
    db.template setColumn<float>("fwhmTheta", d._fwhmTheta);     
    db.template setColumn<double>("psfMag", d._psfMag);
    db.template setColumn<float>("psfMagErr", d._psfMagErr);
    db.template setColumn<double>("apMag", d._apMag);
    db.template setColumn<float>("apMagErr", d._apMagErr);
    db.template setColumn<double>("modelMag", d._modelMag);            
	db.template setColumn<float>("modelMagErr", d._modelMagErr);   

	if(!d.isNull(detail::PETRO_MAG))
	    db. template setColumn<double>("petroMag", d._petroMag);
	else db.setColumnToNull("petroMag");
	
	if(!d.isNull(detail::PETRO_MAG_ERR))
	    db. template setColumn<float>("petroMagErr", d._petroMagErr);  
	else db.setColumnToNull("petroMagErr");    
	
    db.template setColumn<double>("instMag", d._instMag);
    db.template setColumn("instMagErr", d._instMagErr);
    
    if(!d.isNull(detail::NON_GRAY_CORR_MAG))
    	db.template setColumn<double>("nonGrayCorrMag", d._nonGrayCorrMag);
	else db.setColumnToNull("nonGrayCorrMag");
	
	if(!d.isNull(detail::NON_GRAY_CORR_MAG_ERR))
	    db.template setColumn<double>("nonGrayCorrMagErr", d._nonGrayCorrMagErr);
    else db.setColumnToNull("nonGrayCorrMagErr");
        
    if(!d.isNull(detail::ATM_CORR_MAG))
    	db.template setColumn<double>("atmCorrMag", d._atmCorrMag);
    else db.setColumnToNull("atmCorrMag");
    
    if(!d.isNull(detail::ATM_CORR_MAG_ERR))
        db.template setColumn<double>("atmCorrMagErr", d._atmCorrMagErr);
    else db.setColumnToNull("atmCorrMagErr");
    
    if(!d.isNull(detail::AP_DIA))
        db.template setColumn<float>("apDia", d._apDia);
    else db.setColumnToNull("apDia");

	
    db.template setColumn<float>("snr", d._snr);
    db.template setColumn("chi2", d._chi2);

	if(!d.isNull(detail::SKY))
    	db.template setColumn<float>("sky", d._sky);
    else db.setColumnToNull("sky");
       	
   	if(!d.isNull(detail::SKY_ERR))
    	db.template setColumn<float>("skyErr", d._skyErr);
    else db.setColumnToNull("skyErr");
        
    if(!d.isNull(detail::FLAG_4_ASSOCIATION))
        db.template setColumn<int16_t>("flag4association", d._flag4association);
    else db.setColumnToNull("flag4association");
    
    if(!d.isNull(detail::FLAG_4_DETECTION))
        db.template setColumn<int16_t>("flag4detection", d._flag4detection);
    else db.setColumnToNull("flag4detection");
    
    if(!d.isNull(detail::FLAG_4_WCS))
        db.template setColumn<int16_t>("flag4wcs", d._flag4wcs);
    else db.setColumnToNull("flag4wcs");

    db.insertRow();
}

//! \cond
template void form::SourceVectorFormatter::insertRow<DbStorage>   (DbStorage & db,    Source const &d);
template void form::SourceVectorFormatter::insertRow<DbTsvStorage>(DbTsvStorage & db, Source const &d);
//! \endcond


/*! Prepares for reading Source instances from a database table. */
void form::SourceVectorFormatter::setupFetch(DbStorage & db, Source & d) {
    db.outParam("sourceId",         &(d._id));
    db.outParam("ampExposureId",    &(d._ampExposureId));
    db.outParam("filterId",         reinterpret_cast<char *>(&(d._filterId)));
    db.outParam("objectId",         &(d._objectId));
    db.outParam("movingObjectId",   &(d._movingObjectId));
    db.outParam("procHistoryId",    &(d._procHistoryId));
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
    db.outParam("psfMag",           &(d._psfMag));
    db.outParam("psfMagErr",        &(d._psfMagErr));
    db.outParam("apMag",            &(d._apMag));
    db.outParam("apMagErr",         &(d._apMagErr));
    db.outParam("modelMag",         &(d._modelMag));
    db.outParam("modelMagErr",      &(d._modelMagErr));
    db.outParam("petroMag",         &(d._petroMag));
    db.outParam("petroMagErr",      &(d._petroMagErr));
    db.outParam("instMag",          &(d._instMag));
    db.outParam("instMagErr",       &(d._instMagErr));
    db.outParam("nonGrayCorrMag",   &(d._nonGrayCorrMag));
    db.outParam("nonGrayCorrMagErr",&(d._nonGrayCorrMagErr));    
    db.outParam("atmCorrMag",       &(d._atmCorrMag));
    db.outParam("atmCorrMagErr",    &(d._atmCorrMagErr));     
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
    PersistableSourceVector * p = dynamic_cast<PersistableSourceVector *>(persistable);
    
    archive & boost::serialization::base_object<Persistable>(*p);
    
    SourceVector::size_type sz;

    if (Archive::is_loading::value) {        
        Source data;
        Source::Ptr sourcePtr;
        archive & sz;
        SourceVector sourceVector(sz);
        for (; sz > 0; --sz) {
            archive & data;
			sourcePtr.reset(new Source(data));
            sourceVector.push_back(sourcePtr);
        }
        p->setSources(sourceVector);
    } else {
        SourceVector sourceVector = p->getSources();
        sz = sourceVector.size();
        archive & sz;
        SourceVector::iterator i = sourceVector.begin();
        SourceVector::iterator const end(sourceVector.end());
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


void form::SourceVectorFormatter::write(
    Persistable const *   persistable,
    Storage::Ptr          storage,
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
		throw LSST_EXCEPT(ex::RuntimeErrorException, "Persistable was not of concrete type SourceVector");
    }
    SourceVector sourceVector = p->getSources();   

    // Assume all have ids or none do.
    if ((*(sourceVector.begin()))->getId() == 0 && 
        (!_policy || !_policy->exists("GenerateIds") || _policy->getBool("GenerateIds"))
    ) {
     
        unsigned short seq    = 1;
        int64_t visitId       = extractVisitId(additionalData);
        int64_t ampExposureId = extractCcdExposureId(additionalData);
        int     ccdId         = extractCcdId(additionalData);
        if (ccdId < 0 || ccdId >= 256) {
			throw LSST_EXCEPT(ex::InvalidParameterException, "ampExposureId out of range");
        }
        
        SourceVector::iterator i = sourceVector.begin();
        for ( ; i != sourceVector.end(); ++i) {
            (*i)->setId(generateSourceId(seq, ccdId, visitId));
            (*i)->setAmpExposureId(ampExposureId);
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
            
            SourceVector::const_iterator i(sourceVector.begin());
            SourceVector::const_iterator const end(sourceVector.end());
            for ( ; i != end; ++i) {
                insertRow<DbStorage>(*db, **i);
            }
        } else {
            DbTsvStorage * db = dynamic_cast<DbTsvStorage *>(storage.get());
            if (db == 0) {
    			throw LSST_EXCEPT(ex::RuntimeErrorException, "Didn't get DbTsvStorage");
            }
            db->createTableFromTemplate(name, model, mayExist);
            db->setTableForInsert(name);

            SourceVector::const_iterator i(sourceVector.begin());
            SourceVector::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, **i);
            }
        }
    } else {
		throw LSST_EXCEPT(ex::InvalidParameterException, "Storage type is not supported"); 
    }
}


Persistable* form::SourceVectorFormatter::read(
    Storage::Ptr          storage,
    lsst::daf::base::PropertySet::Ptr additionalData
) {   
    std::auto_ptr<PersistableSourceVector> p(new PersistableSourceVector);

    
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

        Source::Ptr sourcePtr;
        SourceVector sourceVector;
        // loop over all retrieve tables, reading in everything
        std::vector<std::string>::const_iterator const end = tables.end();
        for (std::vector<std::string>::const_iterator i = tables.begin(); i != end; ++i) {
            db->setTableForQuery(*i);
            Source data;
            setupFetch(*db, data);
            db->query();
            data.setNotNull();
            while (db->next()) {
            	if (db->columnIsNull(SOURCE_ID)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"sourceId\""); 
        		}
                if (db->columnIsNull(AMP_EXPOSURE_ID)) { data.setNull(detail::AMP_EXPOSURE_ID); }
                if (db->columnIsNull(FILTER_ID)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"filterId\""); 
        		}
                if (db->columnIsNull(OBJECT_ID)) { data.setNull(detail::OBJECT_ID); }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { data.setNull(detail::MOVING_OBJECT_ID); }
                if (db->columnIsNull(PROC_HISTORY_ID)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"procHistoryId\""); 
            	}
                if (db->columnIsNull(RA)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"ra\""); 
            	}
                if (db->columnIsNull(DECL)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decl\""); 
            	}
                if (db->columnIsNull(RA_ERR_4_WCS)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"raErr4wcs\""); 
            	}
                if (db->columnIsNull(DEC_ERR_4_WCS)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decErr4wcs\""); 
            	}
                if (db->columnIsNull(RA_ERR_4_DETECTION)) { data.setNull(detail::RA_ERR_4_DETECTION); }
                if (db->columnIsNull(DEC_ERR_4_DETECTION)) { data.setNull(detail::DEC_ERR_4_DETECTION); }
                if (db->columnIsNull(X_FLUX)) { data.setNull(detail::X_FLUX); }
                if (db->columnIsNull(X_FLUX_ERR)) { data.setNull(detail::X_FLUX_ERR); }
                if (db->columnIsNull(Y_FLUX)) { data.setNull(detail::Y_FLUX); }
                if (db->columnIsNull(Y_FLUX_ERR)) { data.setNull(detail::Y_FLUX_ERR); }
                if (db->columnIsNull(RA_FLUX)) { data.setNull(detail::RA_FLUX); }
                if (db->columnIsNull(RA_FLUX_ERR)) { data.setNull(detail::RA_FLUX_ERR); }
                if (db->columnIsNull(DEC_FLUX)) { data.setNull(detail::DEC_FLUX); }
                if (db->columnIsNull(DEC_FLUX_ERR)) { data.setNull(detail::DEC_FLUX_ERR); }
                if (db->columnIsNull(X_PEAK)) { data.setNull(detail::X_PEAK); }
                if (db->columnIsNull(Y_PEAK)) { data.setNull(detail::Y_PEAK); }
                if (db->columnIsNull(RA_PEAK)) { data.setNull(detail::RA_PEAK); }
                if (db->columnIsNull(DEC_PEAK)) { data.setNull(detail::DEC_PEAK); }
                if (db->columnIsNull(X_ASTROM)) { data.setNull(detail::X_ASTROM); }
                if (db->columnIsNull(X_ASTROM_ERR)) { data.setNull(detail::X_ASTROM_ERR); }
                if (db->columnIsNull(Y_ASTROM)) { data.setNull(detail::Y_ASTROM); }
                if (db->columnIsNull(Y_ASTROM_ERR)) { data.setNull(detail::Y_ASTROM_ERR); }
                if (db->columnIsNull(RA_ASTROM)) { data.setNull(detail::RA_ASTROM); }
                if (db->columnIsNull(RA_ASTROM_ERR)) { data.setNull(detail::RA_ASTROM_ERR); }
                if (db->columnIsNull(DEC_ASTROM)) { data.setNull(detail::DEC_ASTROM); }
                if (db->columnIsNull(DEC_ASTROM_ERR)) { data.setNull(detail::DEC_ASTROM_ERR); }
                if (db->columnIsNull(TAI_MID_POINT)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"taiMidPoint\""); 
            	}
                if (db->columnIsNull(TAI_RANGE)) { data.setNull(detail::TAI_RANGE); }
                if (db->columnIsNull(FWHM_A)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmA\"");            
            	}
                if (db->columnIsNull(FWHM_B)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmB\"");            
            	}
                if (db->columnIsNull(FWHM_THETA)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"fwhmTheta\"");        
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
                if (db->columnIsNull(MODEL_MAG_ERR)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"modelMagErr\""); 
            	}
                if (db->columnIsNull(PETRO_MAG)) { data.setNull(detail::PETRO_MAG); }
                if (db->columnIsNull(PETRO_MAG_ERR)) { data.setNull(detail::PETRO_MAG_ERR); }    
                if (db->columnIsNull(INST_MAG)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMag\""); 
            	}
                if (db->columnIsNull(INST_MAG_ERR)) { 
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMagErr\"");
            	}
                if (db->columnIsNull(NON_GRAY_CORR_MAG)) { data.setNull(detail::NON_GRAY_CORR_MAG); }
            	if (db->columnIsNull(NON_GRAY_CORR_MAG_ERR)) { data.setNull(detail::NON_GRAY_CORR_MAG_ERR); }
                if (db->columnIsNull(ATM_CORR_MAG)) { data.setNull(detail::ATM_CORR_MAG); }
                if (db->columnIsNull(ATM_CORR_MAG_ERR)) { data.setNull(detail::ATM_CORR_MAG_ERR); }
                if (db->columnIsNull(AP_DIA)) { data.setNull(detail::AP_DIA); }                
                if (db->columnIsNull(SNR)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"snr\""); 
            	}
                if (db->columnIsNull(CHI2)) {  
            		throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"chi2\"");
            	}
                if (db->columnIsNull(SKY)) { data.setNull(detail::SKY); }
                if (db->columnIsNull(SKY_ERR)) { data.setNull(detail::SKY_ERR); } 
                if (db->columnIsNull(FLAG_4_ASSOCIATION)) { data.setNull(detail::FLAG_4_ASSOCIATION);}
                if (db->columnIsNull(FLAG_4_DETECTION)) { data.setNull(detail::FLAG_4_DETECTION); }
                if (db->columnIsNull(FLAG_4_WCS)) { data.setNull(detail::FLAG_4_WCS); }
                
                sourcePtr.reset(new Source(data));
                
                //add source to vector
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


void form::SourceVectorFormatter::update(Persistable*, 
    Storage::Ptr, lsst::daf::base::PropertySet::Ptr
) {
	throw LSST_EXCEPT(ex::RuntimeErrorException, "SourceVectorFormatter: updates not supported");
}
