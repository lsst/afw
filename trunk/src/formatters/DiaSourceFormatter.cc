// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Implementation of persistence for DiaSource instances
//
//##====----------------                                ----------------====##/

#include <memory>

#include "boost/format.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/policy/Policy.h"
#include "lsst/afw/formatters/DiaSourceFormatter.h"
#include "lsst/afw/formatters/Utils.h"
#include "lsst/afw/detection/DiaSource.h"
#include "lsst/utils/ieee.h"

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


class Binder {
public:
    Binder(DbStorage* db) : _db(db), _fieldCount(0) { };
    template <typename T> void bind(char const* fieldName, T* ptr,
              int nullLocation = -1) {
        _db->outParam(fieldName, ptr);
        if (nullLocation >= 0) {
            _nullMap.insert(std::pair<int, int>(_fieldCount, nullLocation));
        }
        ++_fieldCount;
    };
    void setNulls(DiaSource& s) {
        for (std::map<int, int>::const_iterator it = _nullMap.begin();
             it != _nullMap.end(); ++it) {
            if (_db->columnIsNull((*it).first)) {
                s.setNull((*it).second);
            }
        }
    };
private:
    DbStorage* _db;
    int _fieldCount;
    std::map<int, int> _nullMap;
};


/*!
    \internal   Generates a unique identifier for a DiaSource given the ampExposureId of the
                originating amplifier and the sequence number of the DiaSource within the amplifier.
 */
inline static int64_t generateDiaSourceId(unsigned short seqNum, int64_t ampExposureId) {
    return (ampExposureId << 16) + seqNum;
}

template <typename T, typename F>
inline static void insertFp(T & db, F const & val, char const * const col, bool isNull=false) {
    if (isNull || lsst::utils::isnan(val)) {
        db.setColumnToNull(col);
    } else if (lsst::utils::isinf(val)) {
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
    db.template setColumn<int64_t>("scienceCcdExposureId", d._ampExposureId);

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
    insertFp(db, d._raErrForDetection, "raSigmaForDetection", d.isNull(det::RA_ERR_FOR_DETECTION));
    insertFp(db, d._raErrForWcs, "raSigmaForWcs", d.isNull(det::RA_ERR_FOR_WCS));
    insertFp(db, d._dec, "decl");
    insertFp(db, d._decErrForDetection, "declSigmaForDetection", d.isNull(det::DEC_ERR_FOR_DETECTION));
    insertFp(db, d._decErrForWcs, "declSigmaForWcs", d.isNull(det::DEC_ERR_FOR_WCS));

    insertFp(db, d._xAstrom, "xAstrom");
    insertFp(db, d._xAstromErr, "xAstromSigma", d.isNull(det::X_ASTROM_ERR));
    insertFp(db, d._yAstrom, "yAstrom");
    insertFp(db, d._yAstromErr, "yAstromSigma", d.isNull(det::Y_ASTROM_ERR));

    insertFp(db, d._taiMidPoint, "taiMidPoint");
    insertFp(db, d._taiRange, "taiRange");

    insertFp(db, d._psfFlux, "psfFlux");
    insertFp(db, d._psfFluxErr, "psfFluxSigma", d.isNull(det::PSF_FLUX_ERR));
    insertFp(db, d._apFlux, "apFlux");
    insertFp(db, d._apFluxErr, "apFluxSigma", d.isNull(det::AP_FLUX_ERR));
    insertFp(db, d._modelFlux, "modelFlux");
    insertFp(db, d._modelFluxErr, "modelFluxSigma", d.isNull(det::MODEL_FLUX_ERR));
    insertFp(db, d._instFlux, "instFlux");
    insertFp(db, d._instFluxErr, "instFluxSigma", d.isNull(det::INST_FLUX_ERR));

    insertFp(db, d._apDia, "apDia", d.isNull(det::AP_DIA));
   
    insertFp(db, d._ixx, "Ixx", d.isNull(det::IXX));
    insertFp(db, d._ixxErr, "IxxSigma", d.isNull(det::IXX_ERR));
    insertFp(db, d._iyy, "Iyy", d.isNull(det::IYY));
    insertFp(db, d._iyyErr, "IyySigma", d.isNull(det::IYY_ERR));
    insertFp(db, d._ixy, "Ixy", d.isNull(det::IXY));
    insertFp(db, d._ixyErr, "IxySigma", d.isNull(det::IXY_ERR));

    insertFp(db, d._psfIxx, "psfIxx", d.isNull(det::PSF_IXX));
    insertFp(db, d._psfIxxErr, "psfIxxSigma", d.isNull(det::PSF_IXX_ERR));
    insertFp(db, d._psfIyy, "psfIyy", d.isNull(det::PSF_IYY));
    insertFp(db, d._psfIyyErr, "psfIyySigma", d.isNull(det::PSF_IYY_ERR));
    insertFp(db, d._psfIxy, "psfIxy", d.isNull(det::PSF_IXY));
    insertFp(db, d._psfIxyErr, "psfIxySigma", d.isNull(det::PSF_IXY_ERR));

    insertFp(db, d._e1, "e1_SG", d.isNull(det::E1));
    insertFp(db, d._e1Err, "e1_SG_Sigma", d.isNull(det::E1_ERR));
    insertFp(db, d._e2, "e2_SG", d.isNull(det::E2));
    insertFp(db, d._e2Err, "e2_SG_Sigma", d.isNull(det::E2_ERR));
    insertFp(db, d._shear1, "shear1_SG", d.isNull(det::SHEAR1));
    insertFp(db, d._shear1Err, "shear1_SG_Sigma", d.isNull(det::SHEAR1_ERR));
    insertFp(db, d._shear2, "shear2_SG", d.isNull(det::SHEAR2));
    insertFp(db, d._shear2Err, "shear2_SG_Sigma", d.isNull(det::SHEAR2_ERR));

    insertFp(db, d._resolution, "resolution_SG", d.isNull(det::RESOLUTION));
    insertFp(db, d._sigma, "sourceWidth_SG", d.isNull(det::SIGMA));
    insertFp(db, d._sigmaErr, "sourceWidth_SG_Sigma", d.isNull(det::SIGMA_ERR));
    insertFp(db, d._shapeStatus, "shapeFlag_SG", d.isNull(det::SHAPE_STATUS));
    
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
template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive &, unsigned int const, Persistable *
);
template void form::DiaSourceVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
    boost::archive::binary_iarchive &, unsigned int const, Persistable *
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
            Binder b(db);
            
            //set target for query output
            b.bind("diaSourceId",          &(data._id));
            b.bind("scienceCcdExposureId", &(data._ampExposureId));
            b.bind("diaSourceToId",        &(data._diaSourceToId), det::DIA_SOURCE_TO_ID);
            b.bind("filterId",             reinterpret_cast<char *>(&(data._filterId)));
            b.bind("objectId",             &(data._objectId), det::OBJECT_ID);
            b.bind("movingObjectId",       &(data._movingObjectId), det::MOVING_OBJECT_ID);
            b.bind("xAstrom",              &(data._xAstrom));
            b.bind("xAstromSigma",         &(data._xAstromErr), det::X_ASTROM_ERR);    
            b.bind("yAstrom",              &(data._yAstrom));
            b.bind("yAstromSigma",         &(data._yAstromErr), det::Y_ASTROM_ERR);  
            b.bind("ra",                   &(data._ra));
            b.bind("raSigmaForDetection",  &(data._raErrForDetection), det::RA_ERR_FOR_DETECTION);
            b.bind("raSigmaForWcs",        &(data._raErrForWcs), det::RA_ERR_FOR_WCS);
            b.bind("decl",                 &(data._dec));
            b.bind("declSigmaForDetection",&(data._decErrForDetection), det::DEC_ERR_FOR_DETECTION);
            b.bind("declSigmaForWcs",      &(data._decErrForWcs), det::DEC_ERR_FOR_WCS);
            b.bind("taiMidPoint",          &(data._taiMidPoint));
            b.bind("taiRange",             &(data._taiRange)); 
            b.bind("Ixx",                  &(data._ixx), det::IXX);
            b.bind("IxxSigma",             &(data._ixxErr), det::IXX_ERR);
            b.bind("Iyy",                  &(data._iyy), det::IYY);
            b.bind("IyySigma",             &(data._iyyErr), det::IYY_ERR);
            b.bind("Ixy",                  &(data._ixy), det::IXY);
            b.bind("IxySigma",             &(data._ixyErr), det::IXY_ERR); 
            b.bind("psfIxx",               &(data._psfIxx), det::PSF_IXX);
            b.bind("psfIxxSigma",          &(data._psfIxxErr), det::PSF_IXX_ERR);
            b.bind("psfIyy",               &(data._psfIyy), det::PSF_IYY);
            b.bind("psfIyySigma",            &(data._psfIyyErr), det::PSF_IYY_ERR);
            b.bind("psfIxy",               &(data._psfIxy), det::PSF_IXY);
            b.bind("psfIxySigma",          &(data._psfIxyErr), det::PSF_IXY_ERR);
            b.bind("psfFlux",              &(data._psfFlux));
            b.bind("psfFluxSigma",           &(data._psfFluxErr), det::PSF_FLUX_ERR);
            b.bind("e1_SG",                   &(data._e1), det::E1);
            b.bind("e1_SG_Sigma",                &(data._e1Err), det::E1_ERR);
            b.bind("e2_SG",                   &(data._e2), det::E2);
            b.bind("e2_SG_Sigma",                &(data._e2Err), det::E2_ERR);
            b.bind("shear1_SG",               &(data._shear1), det::SHEAR1);
            b.bind("shear1_SG_Sigma",            &(data._shear1Err), det::SHEAR1_ERR);
            b.bind("shear2_SG",               &(data._shear2), det::SHEAR2);
            b.bind("shear2_SG_Sigma",            &(data._shear2Err), det::SHEAR2_ERR);
            b.bind("resolution_SG",           &(data._resolution), det::RESOLUTION);
            b.bind("sourceWidth_SG",                &(data._sigma), det::SIGMA);
            b.bind("sourceWidth_SG_Sigma",             &(data._sigmaErr), det::SIGMA_ERR);
            b.bind("shapeFlag_SG",          &(data._shapeStatus), det::SHAPE_STATUS);
            b.bind("apFlux",               &(data._apFlux));
            b.bind("apFluxSigma",            &(data._apFluxErr), det::AP_FLUX_ERR);
            b.bind("modelFlux",            &(data._modelFlux));
            b.bind("modelFluxSigma",         &(data._modelFluxErr), det::MODEL_FLUX_ERR);
            b.bind("instFlux",             &(data._instFlux));
            b.bind("instFluxSigma",          &(data._instFluxErr), det::INST_FLUX_ERR);
            b.bind("apDia",                &(data._apDia), det::AP_DIA);      
            b.bind("flagForClassification",&(data._flagClassification), det::FLAG_CLASSIFICATION);
            b.bind("flagForDetection",     &(data._flagForDetection), det::FLAG_FOR_DETECTION);
            b.bind("snr",                  &(data._snr));
            b.bind("chi2",                 &(data._chi2));

#if 0
            //not defined in DC3a. Keep for DC3b
            b.bind("procHistoryId",      &(data._procHistoryId));
            b.bind("scId",               &(data._scId));
            b.bind("ssmId",              &(data._ssmId));  
            b.bind("xFlux",              &(data._xFlux));
            b.bind("xFluxSigma",           &(data._xFluxErr));
            b.bind("yFlux",              &(data._yFlux));    
            b.bind("yFluxSigma",           &(data._yFluxErr));
            b.bind("raFlux",             &(data._raFlux));
            b.bind("raFluxSigma",          &(data._raFluxErr));
            b.bind("declFlux",           &(data._decFlux));    
            b.bind("declFluxSigma",        &(data._decFluxErr));
            b.bind("xPeak",              &(data._xPeak));
            b.bind("yPeak",              &(data._yPeak));
            b.bind("raPeak",             &(data._raPeak));
            b.bind("declPeak",           &(data._decPeak));            
            b.bind("raAstrom",           &(data._raAstrom));
            b.bind("raAstromSigma",        &(data._raAstromErr));    
            b.bind("declAstrom",         &(data._decAstrom));
            b.bind("declAstromSigma",      &(data._decAstromErr));
            b.bind("lengthDeg",          &(data._lengthDeg));
            b.bind("nonGrayCorrFlux",    &(data._nonGrayCorrFlux));
            b.bind("nonGrayCorrFluxSigma", &(data._nonGrayCorrFluxErr));    
            b.bind("atmCorrFlux",        &(data._atmCorrFlux));
            b.bind("atmCorrFluxSigma",     &(data._atmCorrFluxErr)); 
            b.bind("refFlux",            &(data._refFlux));
            b.bind("flag4wcs",           &(data._flagForWcs));
            b.bind("flag4association",   &(data._flagForAssociation));
            b.bind("valx1",              &(data._valX1));
            b.bind("valx2",              &(data._valX2));
            b.bind("valy1",              &(data._valY1));
            b.bind("valy2",              &(data._valY2));
            b.bind("valxy",              &(data._valXY));  
            b.bind("obsCode",            &(data._obsCode));
            b.bind("isSynthetic",        &(data._isSynthetic));
            b.bind("mopsStatus",         &(data._mopsStatus));  
#endif
            
            //perform query
            db->query();
            
            //Loop over every value in the returned query
            //add a DiaSource to sourceVector
            data.setNotNull();
            while (db->next()) {
                //Now validate each column. 
                b.setNulls(data);

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
