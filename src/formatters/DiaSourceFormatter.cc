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
namespace detail = lsst::afw::detection::source_detail;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::DbStorage;
using lsst::daf::persistence::DbTsvStorage;
using lsst::daf::persistence::Storage;
using lsst::pex::policy::Policy;
using lsst::afw::detection::DiaSource;
using lsst::afw::detection::DiaSourceVector;
using lsst::afw::detection::PersistableDiaSourceVector;

namespace form = lsst::afw::formatters;

// -- DiaSourceVectorFormatter ----------------

form::DiaSourceVectorFormatter::DiaSourceVectorFormatter(Policy::Ptr const & policy) 
    : lsst::daf::persistence::Formatter(typeid(*this)), _policy(policy) {}
    
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
    db.template setColumn<int64_t>("diaSourceId", d._id);
    db.template setColumn<int64_t>("ampExposureId", d._ampExposureId);
    
    if(!d.isNull(detail::DIA_SOURCE_2_ID))
	    db.template setColumn<int64_t>("diaSource2Id", d._diaSource2Id);
   	else db.setColumnToNull("diaSource2Id");
   	
    db.template setColumn<char>("filterId", static_cast<char>(d._filterId));
    
    if(!d.isNull(detail::OBJECT_ID))
    	db. template setColumn<int64_t>("objectId", d._objectId);
   	else db.setColumnToNull("objectId");
   	
    if(!d.isNull(detail::MOVING_OBJECT_ID))
    	db. template setColumn<int64_t>("movingObjectId", d._movingObjectId);
   	else db.setColumnToNull("movingObjectId");
   	
    db.template setColumn<int32_t>("procHistoryID", d._procHistoryId);
    db.template setColumn<int32_t>("scId", d._scId);
    
    if(!d.isNull(detail::SSM_ID))
    	db. template setColumn<int64_t>("ssmId", d._ssmId);        
   	else db.setColumnToNull("ssmId");
   	
    db.template setColumn<double>("ra", d._ra);
    db.template setColumn<float>("raErr4detection", d._raErr4detection);
    if(!d.isNull(detail::RA_ERR_4_WCS))
  	    db. template setColumn<float>("raErr4wcs", d._raErr4wcs);    
   	else db.setColumnToNull("raErr4wcs");

    db.template setColumn<double>("decl", d._dec);
    db.template setColumn<float>("decErr4detection", d._decErr4detection);
    if(!d.isNull(detail::DEC_ERR_4_WCS))
        db. template setColumn<float>("decErr4wcs", d._decErr4wcs);    
    else db.setColumnToNull("decErr4wcs");

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
    	db. template setColumn<double>("decPeak", d._decPeak);
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
        db. template setColumn<double>("yAstromErr", d._yAstromErr);
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
    db.template setColumn<float>("taiRange", d._taiRange);
    db.template setColumn<float>("fwhmA", d._fwhmA);
    db.template setColumn<float>("fwhmB", d._fwhmB);
    db.template setColumn<float>("fwhmTheta", d._fwhmTheta);
    db.template setColumn<double>("lengthDeg", d._lengthDeg);
    db.template setColumn<float>("flux", d._flux);
    db.template setColumn<float>("fluxErr", d._fluxErr);         
    db.template setColumn<double>("psfMag", d._psfMag);
    db.template setColumn<float>("psfMagErr", d._psfMagErr);
    db.template setColumn<double>("apMag", d._apMag);
    db.template setColumn<float>("apMagErr", d._apMagErr);
    db.template setColumn<double>("modelMag", d._modelMag);            
    
	if(!d.isNull(detail::MODEL_MAG_ERR))
    	db.template setColumn<float>("modelMagErr", d._modelMagErr);   
	else db.setColumnToNull("modelMagErr");
	
    db.template setColumn<double>("instMag", d._instMag);
    db.template setColumn<double>("instMagErr", d._instMagErr);
    
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
    
    if(!d.isNull(detail::REF_MAG))
        db.template setColumn<float>("refMag", d._refMag);
    else db.setColumnToNull("refMag");
    
    if(!d.isNull(detail::IXX))
        db.template setColumn<float>("Ixx", d._ixx);
    else db.setColumnToNull("Ixx");
    
    if(!d.isNull(detail::IXX_ERR))
        db.template setColumn<float>("IxxErr", d._ixxErr);
    else db.setColumnToNull("IxxErr");
    
    if(!d.isNull(detail::IYY))    
    	db.template setColumn<float>("Iyy", d._iyy);
    else db.setColumnToNull("Iyy");
    
    if(!d.isNull(detail::IYY_ERR))
        db.template setColumn<float>("IyyErr", d._iyyErr);
    else db.setColumnToNull("IyyErr");
    
    if(!d.isNull(detail::IXY))
        db.template setColumn<float>("Ixy", d._ixy);
    else db.setColumnToNull("Ixy");
    
    if(!d.isNull(detail::IXY_ERR))
        db.template setColumn<float>("IxyErr", d._ixyErr);        
	else db.setColumnToNull("IxyErr");
	
    db.template setColumn<float>("snr", d._snr);
    db.template setColumn<float>("chi2", d._chi2);
    db.template setColumn<double>("valx1", d._valX1);
    db.template setColumn<double>("valx2", d._valX2);
    db.template setColumn<double>("valy1", d._valY1);
    db.template setColumn<double>("valy2", d._valY2);
    db.template setColumn<double>("valxy", d._valXY);        

    if(!d.isNull(detail::OBS_CODE))
        db.template setColumn<char>("obsCode", d._obsCode);
    else db.setColumnToNull("obsCode");
    
    if(!d.isNull(detail::IS_SYNTHETIC))
        db.template setColumn<char>("isSynthetic", d._isSynthetic);
    else db.setColumnToNull("isSynthetic");
    
    if(!d.isNull(detail::MOPS_STATUS))
        db.template setColumn<char>("mopsStatus", d._mopsStatus);      
    else db.setColumnToNull("mopsStatus");
    
    if(!d.isNull(detail::FLAG_4_ASSOCIATION))
        db.template setColumn<int16_t>("flag4association", d._flag4association);
    else db.setColumnToNull("flag4association");
    
    if(!d.isNull(detail::FLAG_4_DETECTION))
        db.template setColumn<int16_t>("flag4detection", d._flag4detection);
    else db.setColumnToNull("flag4detection");
    
    if(!d.isNull(detail::FLAG_4_WCS))
        db.template setColumn<int16_t>("flag4wcs", d._flag4wcs);
    else db.setColumnToNull("flag4wcs");
    
    if(!d.isNull(detail::FLAG_CLASSIFICATION))
    	db.template setColumn<int64_t>("flagClassification", d._flagClassification);  
    else db.setColumnToNull("flagClassification");  

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
    db.outParam("ssmId",            &(d._ssmId));      
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
    db.outParam("lengthDeg",        &(d._lengthDeg));
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
        archive & sz;
        DiaSource::Ptr sourcePtr;
        DiaSourceVector sourceVector(sz);
        for (; sz > 0; --sz) {
            archive & data;
            sourcePtr.reset(new DiaSource(data));
            sourceVector.push_back(sourcePtr);
        }
        p->setSources(sourceVector);
    } else {
        DiaSourceVector sourceVector = p->getSources();
        sz = sourceVector.size();
        archive & sz;
        DiaSourceVector::iterator i = sourceVector.begin();
        DiaSourceVector::iterator const end(sourceVector.end());
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
    if ( (*(sourceVector.begin()))->getId() == 0 && 
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
            (*i)->setId(generateDiaSourceId(seq, ccdId, visitId));
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
            
            DiaSourceVector::const_iterator i(sourceVector.begin());
            DiaSourceVector::const_iterator const end(sourceVector.end());
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

            DiaSourceVector::const_iterator i(sourceVector.begin());
            DiaSourceVector::const_iterator const end(sourceVector.end());
            for (; i != end; ++i) {
                insertRow<DbTsvStorage>(*db, **i);
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
    std::auto_ptr<PersistableDiaSourceVector> p(new PersistableDiaSourceVector());

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

		DiaSource::Ptr sourcePtr;
        DiaSourceVector sourceVector;
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
                if (db->columnIsNull(DIA_SOURCE_2_ID)) { data.setNull(detail::DIA_SOURCE_2_ID); }
                if (db->columnIsNull(FILTER_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"filterId\""); 
                }
                if (db->columnIsNull(OBJECT_ID)) { data.setNull(detail::OBJECT_ID); }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { data.setNull(detail::MOVING_OBJECT_ID); }
                if (db->columnIsNull(PROC_HISTORY_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"procHistoryId\""); 
                }
                if (db->columnIsNull(SC_ID)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"scId\""); 
                }                
                if (db->columnIsNull(SSM_ID)) { data.setNull(detail::SSM_ID); }
                if (db->columnIsNull(RA)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"ra\""); 
                }                
                if (db->columnIsNull(RA_ERR_4_DETECTION)) { 
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"raErr4detection\"");
                }
                if (db->columnIsNull(RA_ERR_4_WCS)) { data.setNull(detail::RA_ERR_4_WCS); }
                if (db->columnIsNull(DECL)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decl\""); 
                }
                if (db->columnIsNull(DEC_ERR_4_DETECTION)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"decErr4detection\""); 
                }                
                if (db->columnIsNull(DEC_ERR_4_WCS)) { data.setNull(detail::DEC_ERR_4_WCS);                }
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
                if (db->columnIsNull(MODEL_MAG_ERR)) { data.setNull(detail::MODEL_MAG_ERR);  } 
                if (db->columnIsNull(INST_MAG)) {
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMag\""); 
                }
                if (db->columnIsNull(INST_MAG_ERR)) { 
                	throw LSST_EXCEPT(ex::RuntimeErrorException, "null column \"instMagErr\""); 
                }                            
                if (db->columnIsNull(NON_GRAY_CORR_MAG)) { data.setNull(detail::NON_GRAY_CORR_MAG); }
                if (db->columnIsNull(NON_GRAY_CORR_MAG_ERR)) { data.setNull(detail::NON_GRAY_CORR_MAG_ERR);}
                if (db->columnIsNull(ATM_CORR_MAG)) { data.setNull(detail::ATM_CORR_MAG); }
                if (db->columnIsNull(ATM_CORR_MAG_ERR)) { data.setNull(detail::ATM_CORR_MAG_ERR); }
                if (db->columnIsNull(AP_DIA)) { data.setNull(detail::AP_DIA); }
                if (db->columnIsNull(REF_MAG)) { data.setNull(detail::REF_MAG);}                
                if (db->columnIsNull(IXX)) { data.setNull(detail::IXX);}
                if (db->columnIsNull(IXX_ERR)) { data.setNull(detail::IXX_ERR);}
                if (db->columnIsNull(IYY)) { data.setNull(detail::IYY); }
                if (db->columnIsNull(IYY_ERR)) { data.setNull(detail::IYY_ERR);}
                if (db->columnIsNull(IXY)) { data.setNull(detail::IXY);}
                if (db->columnIsNull(IXY_ERR)) { data.setNull(detail::IXY_ERR); }
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
                if (db->columnIsNull(OBS_CODE)) { data.setNull(detail::OBS_CODE); }
                if (db->columnIsNull(IS_SYNTHETIC)) { data.setNull(detail::IS_SYNTHETIC); }
                if (db->columnIsNull(MOPS_STATUS)) { data.setNull(detail::MOPS_STATUS); }
                if (db->columnIsNull(FLAG_4_ASSOCIATION)) { data.setNull(detail::FLAG_4_ASSOCIATION);}
                if (db->columnIsNull(FLAG_4_DETECTION)) { data.setNull(detail::FLAG_4_DETECTION); }
                if (db->columnIsNull(FLAG_4_WCS)) { data.setNull(detail::FLAG_4_WCS); }       
                if (db->columnIsNull(FLAG_CLASSIFICATION)) {data.setNull(detail::FLAG_CLASSIFICATION); }     

				sourcePtr.reset(new DiaSource(data));
				
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


void form::DiaSourceVectorFormatter::update(Persistable*, 
    Storage::Ptr, lsst::daf::base::PropertySet::Ptr
) {
	throw LSST_EXCEPT(ex::RuntimeErrorException, "DiaSourceVectorFormatter: updates not supported");
}
