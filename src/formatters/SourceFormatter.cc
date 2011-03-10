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
#include "lsst/afw/detection/Measurement.h"
#include "lsst/afw/detection/Astrometry.h"
#include "lsst/afw/detection/Photometry.h"
#include "lsst/afw/detection/Shape.h"

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
using lsst::afw::image::Filter;

namespace form = lsst::afw::formatters;


namespace lsst { namespace afw { namespace formatters { namespace {

template <typename FloatT> inline FloatT _radians(FloatT degrees) {
    return static_cast<FloatT>(degrees * (M_PI/180.0));
}
template <typename FloatT> inline FloatT _degrees(FloatT radians) {
    return static_cast<FloatT>(radians * (180.0/M_PI));
}

template <typename FloatT>
class DegreeValue {
public:
    DegreeValue() : _value(0.0) { }
    DegreeValue(FloatT const & radians) : _value(_degrees(radians)) { }
    DegreeValue & operator=(FloatT const & radians) {
        _value = _degrees(radians);
        return *this;
    }
    FloatT * operator&() { return &_value; }
    FloatT const * operator&() const { return &_value; }
    operator FloatT() const {
        return _radians(_value);
    }
    void rangeReduce() {
        double d = std::fmod(_value, 360.0);
        _value = (d < 0.0) ? d + 360.0 : d;
    }
private:
    FloatT _value;
};

struct FieldsToConvert {
    DegreeValue<double> ra;
    DegreeValue<double> raFlux;
    DegreeValue<double> raPeak;
    DegreeValue<double> raAstrom;
    DegreeValue<double> raObject;
    DegreeValue<double> dec;
    DegreeValue<double> decFlux;
    DegreeValue<double> decPeak;
    DegreeValue<double> decAstrom;
    DegreeValue<double> decObject;
    DegreeValue<float> raErrForDetection;
    DegreeValue<float> raErrForWcs;
    DegreeValue<float> raFluxErr;
    DegreeValue<float> raAstromErr;
    DegreeValue<float> decErrForDetection;
    DegreeValue<float> decErrForWcs;
    DegreeValue<float> decFluxErr;
    DegreeValue<float> decAstromErr;

    FieldsToConvert() { }
    FieldsToConvert(Source const & s) :
        ra(s.getRa()),
        raFlux(s.getRaFlux()),
        raPeak(s.getRaPeak()),
        raAstrom(s.getRaAstrom()),
        raObject(s.getRaObject()),
        dec(s.getDec()),
        decFlux(s.getDecFlux()),
        decPeak(s.getDecPeak()),
        decAstrom(s.getDecAstrom()),
        decObject(s.getDecObject()),
        raErrForDetection(s.getRaErrForDetection()),
        raErrForWcs(s.getRaErrForWcs()),
        raFluxErr(s.getRaFluxErr()),
        raAstromErr(s.getRaAstromErr()),
        decErrForDetection(s.getDecErrForDetection()),
        decErrForWcs(s.getDecErrForWcs()),
        decFluxErr(s.getDecFluxErr()),
        decAstromErr(s.getDecAstromErr())
    {
        ra.rangeReduce();
        raFlux.rangeReduce();
        raPeak.rangeReduce();
        raAstrom.rangeReduce();
        raObject.rangeReduce();
    }

    void fill(Source & s) const {
        s.setRa(ra);
        s.setRaFlux(raFlux);
        s.setRaPeak(raPeak);
        s.setRaAstrom(raAstrom);
        s.setRaObject(raObject);
        s.setDec(dec);
        s.setDecFlux(decFlux);
        s.setDecPeak(decPeak);
        s.setDecAstrom(decAstrom);
        s.setDecObject(decObject);
        s.setRaErrForDetection(raErrForDetection);
        s.setRaErrForWcs(raErrForWcs);
        s.setRaFluxErr(raFluxErr);
        s.setRaAstromErr(raAstromErr);
        s.setDecErrForDetection(decErrForDetection);
        s.setDecErrForWcs(decErrForWcs);
        s.setDecFluxErr(decFluxErr);
        s.setDecAstromErr(decAstromErr);
    }
};

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
    void setNulls(Source& s) {
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


}}}} // namespace lsst::afw::formatters::<anonymous>


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
    Inserts a single Source into a database table using \a db
    (an instance of lsst::daf::persistence::DbStorage or subclass thereof).
 */
template <typename T>
void form::SourceVectorFormatter::insertRow(T & db, Source const & d)
{
    // convert from radians to degrees
    FieldsToConvert cnv(d);

    db.template setColumn<boost::int64_t>("sourceId", d._id);

    if (!d.isNull(det::AMP_EXPOSURE_ID)) {
        db.template setColumn<boost::int64_t>("scienceCcdExposureId", d._ampExposureId);    
    } else {
        db.setColumnToNull("scienceCcdExposureId");
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

    insertFp(db, *&cnv.ra, "ra");
    insertFp(db, *&cnv.raErrForDetection, "raSigmaForDetection", d.isNull(det::RA_ERR_FOR_DETECTION));
    insertFp(db, *&cnv.raErrForWcs, "raSigmaForWcs");
    insertFp(db, *&cnv.dec, "decl");
    insertFp(db, *&cnv.decErrForDetection, "declSigmaForDetection", d.isNull(det::DEC_ERR_FOR_DETECTION));
    insertFp(db, *&cnv.decErrForWcs, "declSigmaForWcs");

    insertFp(db, d._xFlux, "xFlux", d.isNull(det::X_FLUX));
    insertFp(db, d._xFluxErr, "xFluxSigma", d.isNull(det::X_FLUX_ERR));
    insertFp(db, d._yFlux, "yFlux", d.isNull(det::Y_FLUX));
    insertFp(db, d._yFluxErr, "yFluxSigma", d.isNull(det::Y_FLUX_ERR));
    insertFp(db, *&cnv.raFlux, "raFlux", d.isNull(det::RA_FLUX));
    insertFp(db, *&cnv.raFluxErr, "raFluxSigma", d.isNull(det::RA_FLUX_ERR));
    insertFp(db, *&cnv.decFlux, "declFlux", d.isNull(det::DEC_FLUX));
    insertFp(db, *&cnv.decFluxErr, "declFluxSigma", d.isNull(det::DEC_FLUX_ERR));

    insertFp(db, d._xPeak, "xPeak", d.isNull(det::X_PEAK));
    insertFp(db, d._yPeak, "yPeak", d.isNull(det::Y_PEAK));
    insertFp(db, *&cnv.raPeak, "raPeak", d.isNull(det::RA_PEAK));
    insertFp(db, *&cnv.decPeak, "declPeak", d.isNull(det::DEC_PEAK));

    insertFp(db, d._xAstrom, "xAstrom", d.isNull(det::X_ASTROM));
    insertFp(db, d._xAstromErr, "xAstromSigma", d.isNull(det::X_ASTROM_ERR));
    insertFp(db, d._yAstrom, "yAstrom", d.isNull(det::Y_ASTROM));
    insertFp(db, d._yAstromErr, "yAstromSigma", d.isNull(det::Y_ASTROM_ERR));
    insertFp(db, *&cnv.raAstrom, "raAstrom", d.isNull(det::RA_ASTROM));
    insertFp(db, *&cnv.raAstromErr, "raAstromSigma", d.isNull(det::RA_ASTROM_ERR));
    insertFp(db, *&cnv.decAstrom, "declAstrom", d.isNull(det::DEC_ASTROM));
    insertFp(db, *&cnv.decAstromErr, "declAstromSigma", d.isNull(det::DEC_ASTROM_ERR));

    insertFp(db, *&cnv.raObject, "raObject", d.isNull(det::RA_OBJECT));
    insertFp(db, *&cnv.decObject, "declObject", d.isNull(det::DEC_OBJECT));

    insertFp(db, d._taiMidPoint, "taiMidPoint"); 
    insertFp(db, d._taiRange, "taiRange", d.isNull(det::TAI_RANGE));
 
    insertFp(db, d._psfFlux, "psfFlux");
    insertFp(db, d._psfFluxErr, "psfFluxSigma");
    insertFp(db, d._apFlux, "apFlux");
    insertFp(db, d._apFluxErr, "apFluxSigma");
    insertFp(db, d._modelFlux, "modelFlux");
    insertFp(db, d._modelFluxErr, "modelFluxSigma");
    insertFp(db, d._petroFlux, "petroFlux", d.isNull(det::PETRO_FLUX));
    insertFp(db, d._petroFluxErr, "petroFluxSigma", d.isNull(det::PETRO_FLUX_ERR));
    insertFp(db, d._instFlux, "instFlux");
    insertFp(db, d._instFluxErr, "instFluxSigma");
    insertFp(db, d._nonGrayCorrFlux, "nonGrayCorrFlux", d.isNull(det::NON_GRAY_CORR_FLUX));
    insertFp(db, d._nonGrayCorrFluxErr, "nonGrayCorrFluxSigma", d.isNull(det::NON_GRAY_CORR_FLUX_ERR));
    insertFp(db, d._atmCorrFlux, "atmCorrFlux", d.isNull(det::ATM_CORR_FLUX));
    insertFp(db, d._atmCorrFluxErr, "atmCorrFluxSigma", d.isNull(det::ATM_CORR_FLUX_ERR));

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
    insertFp(db, d._resolution, "resolution_SG", d.isNull(det::RESOLUTION));
    insertFp(db, d._shear1, "shear1_SG", d.isNull(det::SHEAR1));
    insertFp(db, d._shear1Err, "shear1_SG_Sigma", d.isNull(det::SHEAR1_ERR));
    insertFp(db, d._shear2, "shear2_SG", d.isNull(det::SHEAR2));
    insertFp(db, d._shear2Err, "shear2_SG_Sigma", d.isNull(det::SHEAR2_ERR));
    insertFp(db, d._sigma, "sourceWidth_SG", d.isNull(det::SIGMA));
    insertFp(db, d._sigmaErr, "sourceWidth_SG_Sigma", d.isNull(det::SIGMA_ERR));

    if (!d.isNull(det::SHAPE_STATUS)) {
        db.template setColumn<boost::int16_t>("shapeFlag_SG", d._shapeStatus);
    } else {
        db.setColumnToNull("shapeFlag_SG");
    } 
    
    insertFp(db, d._snr, "snr");
    insertFp(db, d._chi2, "chi2");
    insertFp(db, d._sky, "sky", d.isNull(det::SKY));
    insertFp(db, d._skyErr, "skySigma", d.isNull(det::SKY_ERR));



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
    // Set filter id for sources with an unknown filter
    if (additionalData && additionalData->exists("filterId") && !additionalData->isArray("filterId")) {
        int filterId = additionalData->getAsInt("filterId");
        for (SourceSet::iterator i = sourceVector.begin(), e = sourceVector.end(); i != e; ++i) {
           if ((*i)->getFilterId() == Filter::UNKNOWN) {
               (*i)->setFilterId(filterId);
           }
        }
    }
    // Assume all have ids or none do.  If none do, assign them ids.
    if (sourceVector.front()->getId() == 0 && additionalData && additionalData->exists("ampExposureId") &&
        (!_policy || !_policy->exists("generateIds") || _policy->getBool("generateIds"))) {
        unsigned short seq = 1;
        boost::int64_t ampExposureId = extractAmpExposureId(additionalData);
        if (sourceVector.size() >= 65536) {
            throw LSST_EXCEPT(ex::RangeErrorException, "too many Sources per-amp: "
                "sequence number overflows 16 bits, potentially causing unique-id conflicts");
        }
        for (SourceSet::iterator i = sourceVector.begin(); i != sourceVector.end(); ++i) {
            (*i)->setId(generateSourceId(seq, ampExposureId));
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
            throw LSST_EXCEPT(ex::RuntimeErrorException, 
                    "Didn't get BoostStorage");
        }
        bs->getOArchive() & *p;
    } else if (typeid(*storage) == typeid(DbStorage) 
            || typeid(*storage) == typeid(DbTsvStorage)) {

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
            Source s;
            FieldsToConvert cnv;
            Binder b(db);

            //set target for query output
            b.bind("sourceId",             &(s._id));
            b.bind("scienceCcdExposureId", &(s._ampExposureId), det::AMP_EXPOSURE_ID);
            b.bind("filterId",             reinterpret_cast<char *>(&(s._filterId)));
            b.bind("objectId",             &(s._objectId), det::OBJECT_ID);
            b.bind("movingObjectId",       &(s._movingObjectId), det::MOVING_OBJECT_ID);
            b.bind("procHistoryId",        &(s._procHistoryId));
            b.bind("ra",                   &(cnv.ra));
            b.bind("raSigmaForDetection",  &(cnv.raErrForDetection), det::RA_ERR_FOR_DETECTION);
            b.bind("raSigmaForWcs",        &(cnv.raErrForWcs));
            b.bind("decl",                 &(cnv.dec));
            b.bind("declSigmaForDetection",&(cnv.decErrForDetection), det::DEC_ERR_FOR_DETECTION);
            b.bind("declSigmaForWcs",      &(cnv.decErrForWcs));
            b.bind("xFlux",                &(s._xFlux), det::X_FLUX);
            b.bind("xFluxSigma",           &(s._xFluxErr), det::X_FLUX_ERR);
            b.bind("yFlux",                &(s._yFlux), det::Y_FLUX);
            b.bind("yFluxSigma",           &(s._yFluxErr), det::Y_FLUX_ERR);
            b.bind("raFlux",               &(cnv.raFlux), det::RA_FLUX);
            b.bind("raFluxSigma",          &(cnv.raFluxErr), det::RA_FLUX_ERR);
            b.bind("declFlux",             &(cnv.decFlux), det::DEC_FLUX);
            b.bind("declFluxSigma",        &(cnv.decFluxErr), det::DEC_FLUX_ERR);
            b.bind("xPeak",                &(s._xPeak), det::X_PEAK);
            b.bind("yPeak",                &(s._yPeak), det::Y_PEAK);
            b.bind("raPeak",               &(cnv.raPeak), det::RA_PEAK);
            b.bind("declPeak",             &(cnv.decPeak), det::DEC_PEAK);
            b.bind("xAstrom",              &(s._xAstrom), det::X_ASTROM);
            b.bind("xAstromSigma",         &(s._xAstromErr), det::X_ASTROM_ERR);
            b.bind("yAstrom",              &(s._yAstrom), det::Y_ASTROM);
            b.bind("yAstromSigma",         &(s._yAstromErr), det::Y_ASTROM_ERR);
            b.bind("raAstrom",             &(cnv.raAstrom), det::RA_ASTROM);
            b.bind("raAstromSigma",        &(cnv.raAstromErr), det::RA_ASTROM_ERR);
            b.bind("declAstrom",           &(cnv.decAstrom), det::DEC_ASTROM);
            b.bind("declAstromSigma",      &(cnv.decAstromErr), det::DEC_ASTROM_ERR);
            b.bind("raObject",             &(cnv.raObject), det::RA_OBJECT);
            b.bind("declObject",           &(cnv.decObject), det::DEC_OBJECT);
            b.bind("taiMidPoint",          &(s._taiMidPoint));
            b.bind("taiRange",             &(s._taiRange), det::TAI_RANGE);
            b.bind("psfFlux",              &(s._psfFlux));
            b.bind("psfFluxSigma",         &(s._psfFluxErr));
            b.bind("apFlux",               &(s._apFlux));
            b.bind("apFluxSigma",          &(s._apFluxErr));
            b.bind("modelFlux",            &(s._modelFlux));
            b.bind("modelFluxSigma",       &(s._modelFluxErr));
            b.bind("petroFlux",            &(s._petroFlux), det::PETRO_FLUX);
            b.bind("petroFluxSigma",       &(s._petroFluxErr), det::PETRO_FLUX_ERR);
            b.bind("instFlux",             &(s._instFlux));
            b.bind("instFluxSigma",        &(s._instFluxErr));
            b.bind("nonGrayCorrFlux",      &(s._nonGrayCorrFlux), det::NON_GRAY_CORR_FLUX);
            b.bind("nonGrayCorrFluxSigma", &(s._nonGrayCorrFluxErr), det::NON_GRAY_CORR_FLUX_ERR);
            b.bind("atmCorrFlux",          &(s._atmCorrFlux), det::ATM_CORR_FLUX);
            b.bind("atmCorrFluxSigma",     &(s._atmCorrFluxErr), det::ATM_CORR_FLUX_ERR);
            b.bind("apDia",                &(s._apDia), det::AP_DIA);
            b.bind("Ixx",                  &(s._ixx), det::IXX);
            b.bind("IxxSigma",             &(s._ixxErr), det::IXX_ERR);
            b.bind("Iyy",                  &(s._iyy), det::IYY);
            b.bind("IyySigma",             &(s._iyyErr), det::IYY_ERR);
            b.bind("Ixy",                  &(s._ixy), det::IXY);
            b.bind("IxySigma",             &(s._ixyErr), det::IXY_ERR);
            b.bind("psfIxx",               &(s._psfIxx), det::PSF_IXX);
            b.bind("psfIxxSigma",          &(s._psfIxxErr), det::PSF_IXX_ERR);
            b.bind("psfIyy",               &(s._psfIyy), det::PSF_IYY);
            b.bind("psfIyySigma",          &(s._psfIyyErr), det::PSF_IYY_ERR);
            b.bind("psfIxy",               &(s._psfIxy), det::PSF_IXY);
            b.bind("psfIxySigma",          &(s._psfIxyErr), det::PSF_IXY_ERR);
            b.bind("e1_SG",                &(s._e1), det::E1);
            b.bind("e1_SG_Sigma",          &(s._e1Err), det::E1_ERR);
            b.bind("e2_SG",                &(s._e2), det::E2);
            b.bind("e2_SG_Sigma",          &(s._e2Err), det::E2_ERR);
            b.bind("shear1_SG",            &(s._shear1), det::SHEAR1);
            b.bind("shear1_SG_Sigma",      &(s._shear1Err), det::SHEAR1_ERR);
            b.bind("shear2_SG",            &(s._shear2), det::SHEAR2);
            b.bind("shear2_SG_Sigma",      &(s._shear2Err), det::SHEAR2_ERR);
            b.bind("resolution_SG",        &(s._resolution), det::RESOLUTION);
            b.bind("sourceWidth_SG",       &(s._sigma), det::SIGMA);
            b.bind("sourceWidth_SG_Sigma", &(s._sigmaErr), det::SIGMA_ERR);
            b.bind("shapeFlag_SG",         &(s._shapeStatus), det::SHAPE_STATUS);
            b.bind("snr",                  &(s._snr));
            b.bind("chi2",                 &(s._chi2));
            b.bind("sky",                  &(s._sky), det::SKY);
            b.bind("skySigma",             &(s._skyErr), det::SKY_ERR);
            b.bind("flagForAssociation",   &(s._flagForAssociation), det::FLAG_FOR_ASSOCIATION);
            b.bind("flagForDetection",     &(s._flagForDetection), det::FLAG_FOR_DETECTION);
            b.bind("flagForWcs",           &(s._flagForWcs), det::FLAG_FOR_WCS);

            //perform query
            db->query();

            //Loop over every value in the returned query
            //add a Source to sourceVector
            s.setNotNull();
            while (db->next()) {
                // convert from degrees to radians for sky coords
                cnv.fill(s);
                //Handle/validate NULL values from the db. 
                b.setNulls(s);

                //add source to vector
                Source::Ptr sourcePtr(new Source(s));
                sourceVector.push_back(sourcePtr);

                //reset nulls for next source
                s.setNotNull();
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
