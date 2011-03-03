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
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

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

    insertFp(db, *&cnv.ra, "ra");
    insertFp(db, *&cnv.raErrForDetection, "raErrForDetection", d.isNull(det::RA_ERR_FOR_DETECTION));
    insertFp(db, *&cnv.raErrForWcs, "raErrForWcs");
    insertFp(db, *&cnv.dec, "decl");
    insertFp(db, *&cnv.decErrForDetection, "declErrForDetection", d.isNull(det::DEC_ERR_FOR_DETECTION));
    insertFp(db, *&cnv.decErrForWcs, "declErrForWcs");

    insertFp(db, d._xFlux, "xFlux", d.isNull(det::X_FLUX));
    insertFp(db, d._xFluxErr, "xFluxErr", d.isNull(det::X_FLUX_ERR));
    insertFp(db, d._yFlux, "yFlux", d.isNull(det::Y_FLUX));
    insertFp(db, d._yFluxErr, "yFluxErr", d.isNull(det::Y_FLUX_ERR));
    insertFp(db, *&cnv.raFlux, "raFlux", d.isNull(det::RA_FLUX));
    insertFp(db, *&cnv.raFluxErr, "raFluxErr", d.isNull(det::RA_FLUX_ERR));
    insertFp(db, *&cnv.decFlux, "declFlux", d.isNull(det::DEC_FLUX));
    insertFp(db, *&cnv.decFluxErr, "declFluxErr", d.isNull(det::DEC_FLUX_ERR));

    insertFp(db, d._xPeak, "xPeak", d.isNull(det::X_PEAK));
    insertFp(db, d._yPeak, "yPeak", d.isNull(det::Y_PEAK));
    insertFp(db, *&cnv.raPeak, "raPeak", d.isNull(det::RA_PEAK));
    insertFp(db, *&cnv.decPeak, "declPeak", d.isNull(det::DEC_PEAK));

    insertFp(db, d._xAstrom, "xAstrom", d.isNull(det::X_ASTROM));
    insertFp(db, d._xAstromErr, "xAstromErr", d.isNull(det::X_ASTROM_ERR));
    insertFp(db, d._yAstrom, "yAstrom", d.isNull(det::Y_ASTROM));
    insertFp(db, d._yAstromErr, "yAstromErr", d.isNull(det::Y_ASTROM_ERR));
    insertFp(db, *&cnv.raAstrom, "raAstrom", d.isNull(det::RA_ASTROM));
    insertFp(db, *&cnv.raAstromErr, "raAstromErr", d.isNull(det::RA_ASTROM_ERR));
    insertFp(db, *&cnv.decAstrom, "declAstrom", d.isNull(det::DEC_ASTROM));
    insertFp(db, *&cnv.decAstromErr, "declAstromErr", d.isNull(det::DEC_ASTROM_ERR));

    insertFp(db, *&cnv.raObject, "raObject", d.isNull(det::RA_OBJECT));
    insertFp(db, *&cnv.decObject, "declObject", d.isNull(det::DEC_OBJECT));

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


    insertFp(db, d._psfIxx, "PsfIxx", d.isNull(det::PSF_IXX));
    insertFp(db, d._psfIxxErr, "PsfIxxErr", d.isNull(det::PSF_IXX_ERR));
    insertFp(db, d._psfIyy, "PsfIyy", d.isNull(det::PSF_IYY));
    insertFp(db, d._psfIyyErr, "PsfIyyErr", d.isNull(det::PSF_IYY_ERR));
    insertFp(db, d._psfIxy, "PsfIxy", d.isNull(det::PSF_IXY));
    insertFp(db, d._psfIxyErr, "PsfIxyErr", d.isNull(det::PSF_IXY_ERR));
    
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
template void form::SourceVectorFormatter::delegateSerialize<boost::archive::binary_oarchive>(
    boost::archive::binary_oarchive &, unsigned int const, Persistable *
);
template void form::SourceVectorFormatter::delegateSerialize<boost::archive::binary_iarchive>(
    boost::archive::binary_iarchive &, unsigned int const, Persistable *
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

            //set target for query output
            db->outParam("sourceId",           &(s._id));
            db->outParam("ampExposureId",      &(s._ampExposureId));
            db->outParam("filterId",           reinterpret_cast<char *>(&(s._filterId)));
            db->outParam("objectId",           &(s._objectId));
            db->outParam("movingObjectId",     &(s._movingObjectId));
            db->outParam("procHistoryId",      &(s._procHistoryId));
            db->outParam("ra",                 &(cnv.ra));
            db->outParam("raErrForDetection",  &(cnv.raErrForDetection));
            db->outParam("raErrForWcs",        &(cnv.raErrForWcs));
            db->outParam("decl",               &(cnv.dec));
            db->outParam("declErrForDetection",&(cnv.decErrForDetection));
            db->outParam("declErrForWcs",      &(cnv.decErrForWcs));
            db->outParam("xFlux",              &(s._xFlux));
            db->outParam("xFluxErr",           &(s._xFluxErr));
            db->outParam("yFlux",              &(s._yFlux));
            db->outParam("yFluxErr",           &(s._yFluxErr));
            db->outParam("raFlux",             &(cnv.raFlux));
            db->outParam("raFluxErr",          &(cnv.raFluxErr));
            db->outParam("declFlux",           &(cnv.decFlux));
            db->outParam("declFluxErr",        &(cnv.decFluxErr));
            db->outParam("xPeak",              &(s._xPeak));
            db->outParam("yPeak",              &(s._yPeak));
            db->outParam("raPeak",             &(cnv.raPeak));
            db->outParam("declPeak",           &(cnv.decPeak));
            db->outParam("xAstrom",            &(s._xAstrom));
            db->outParam("xAstromErr",         &(s._xAstromErr));
            db->outParam("yAstrom",            &(s._yAstrom));
            db->outParam("yAstromErr",         &(s._yAstromErr));
            db->outParam("raAstrom",           &(cnv.raAstrom));
            db->outParam("raAstromErr",        &(cnv.raAstromErr));
            db->outParam("declAstrom",         &(cnv.decAstrom));
            db->outParam("declAstromErr",      &(cnv.decAstromErr));
            db->outParam("raObject",           &(cnv.raObject));
            db->outParam("declObject",         &(cnv.decObject));
            db->outParam("taiMidPoint",        &(s._taiMidPoint));
            db->outParam("taiRange",           &(s._taiRange));
            db->outParam("psfFlux",            &(s._psfFlux));
            db->outParam("psfFluxErr",         &(s._psfFluxErr));
            db->outParam("apFlux",             &(s._apFlux));
            db->outParam("apFluxErr",          &(s._apFluxErr));
            db->outParam("modelFlux",          &(s._modelFlux));
            db->outParam("modelFluxErr",       &(s._modelFluxErr));
            db->outParam("petroFlux",          &(s._petroFlux));
            db->outParam("petroFluxErr",       &(s._petroFluxErr));
            db->outParam("instFlux",           &(s._instFlux));
            db->outParam("instFluxErr",        &(s._instFluxErr));
            db->outParam("nonGrayCorrFlux",    &(s._nonGrayCorrFlux));
            db->outParam("nonGrayCorrFluxErr", &(s._nonGrayCorrFluxErr));
            db->outParam("atmCorrFlux",        &(s._atmCorrFlux));
            db->outParam("atmCorrFluxErr",     &(s._atmCorrFluxErr));
            db->outParam("apDia",              &(s._apDia));
            db->outParam("Ixx",                &(s._ixx));
            db->outParam("IxxErr",             &(s._ixxErr));
            db->outParam("Iyy",                &(s._iyy));
            db->outParam("IyyErr",             &(s._iyyErr));
            db->outParam("Ixy",                &(s._ixy));
            db->outParam("IxyErr",             &(s._ixyErr));
            db->outParam("PsfIxx",             &(s._psfIxx));
            db->outParam("PsfIxxErr",          &(s._psfIxxErr));
            db->outParam("PsfIyy",             &(s._psfIyy));
            db->outParam("PsfIyyErr",          &(s._psfIyyErr));
            db->outParam("PsfIxy",             &(s._psfIxy));
            db->outParam("PsfIxyErr",          &(s._psfIxyErr));
            db->outParam("e1",                   &(s._e1));
            db->outParam("e1Err",                &(s._e1Err));
            db->outParam("e2",                   &(s._e2));
            db->outParam("e2Err",                &(s._e2Err));
            db->outParam("shear1",               &(s._shear1));
            db->outParam("shear1Err",            &(s._shear1Err));
            db->outParam("shear2",               &(s._shear2));
            db->outParam("shear2Err",            &(s._shear2Err));
            db->outParam("resolution",           &(s._resolution));
            db->outParam("sigma",                &(s._sigma));
            db->outParam("sigmaErr",             &(s._sigmaErr));
            db->outParam("shapeStatus",          &(s._shapeStatus));
            db->outParam("snr",                &(s._snr));
            db->outParam("chi2",               &(s._chi2));
            db->outParam("sky",                &(s._sky));
            db->outParam("skyErr",             &(s._skyErr));
            db->outParam("flagForAssociation", &(s._flagForAssociation));
            db->outParam("flagForDetection",   &(s._flagForDetection));
            db->outParam("flagForWcs",         &(s._flagForWcs));

            //perform query
            db->query();

            //Loop over every value in the returned query
            //add a Source to sourceVector
            s.setNotNull();
            while (db->next()) {
                // convert from degrees to radians for sky coords
                cnv.fill(s);
                //Handle/validate NULL values from the db. 
                if (db->columnIsNull(SOURCE_ID)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"sourceId\""); 
                }
                if (db->columnIsNull(AMP_EXPOSURE_ID)) { 
                    s.setNull(det::AMP_EXPOSURE_ID); 
                }
                if (db->columnIsNull(FILTER_ID)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"filterId\""); 
                }
                if (db->columnIsNull(OBJECT_ID)) { 
                    s.setNull(det::OBJECT_ID); 
                }
                if (db->columnIsNull(MOVING_OBJECT_ID)) { 
                    s.setNull(det::MOVING_OBJECT_ID); 
                }
                if (db->columnIsNull(PROC_HISTORY_ID)) { 
                    throw LSST_EXCEPT(ex::RuntimeErrorException, 
                            "null column \"procHistoryId\""); 
                }
                handleNullFp(db, RA, s._ra);
                handleNullFp(db, DECL, s._dec);
                handleNullFp(db, RA_ERR_FOR_WCS, s._raErrForWcs);
                handleNullFp(db, DEC_ERR_FOR_WCS, s._decErrForWcs);
                handleNullFp(db, s, RA_ERR_FOR_DETECTION,
                             s._raErrForDetection, det::RA_ERR_FOR_DETECTION);
                handleNullFp(db, s, DEC_ERR_FOR_DETECTION,
                             s._decErrForDetection, det::DEC_ERR_FOR_DETECTION);

                handleNullFp(db, s, X_FLUX, s._xFlux, det::X_FLUX);
                handleNullFp(db, s, X_FLUX_ERR, s._xFluxErr, det::X_FLUX_ERR);
                handleNullFp(db, s, Y_FLUX, s._yFlux, det::Y_FLUX);
                handleNullFp(db, s, Y_FLUX_ERR, s._yFluxErr, det::Y_FLUX_ERR);
                handleNullFp(db, s, RA_FLUX, s._raFlux, det::RA_FLUX);
                handleNullFp(db, s, RA_FLUX_ERR, s._raFluxErr, det::RA_FLUX_ERR);
                handleNullFp(db, s, DEC_FLUX, s._decFlux, det::DEC_FLUX);
                handleNullFp(db, s, DEC_FLUX_ERR, s._decFluxErr, det::DEC_FLUX_ERR);

                handleNullFp(db, s, X_PEAK, s._xPeak, det::X_PEAK);
                handleNullFp(db, s, Y_PEAK, s._yPeak, det::Y_PEAK);
                handleNullFp(db, s, RA_PEAK, s._raPeak, det::RA_PEAK);
                handleNullFp(db, s, DEC_PEAK, s._decPeak, det::DEC_PEAK);

                handleNullFp(db, s, X_ASTROM, s._xAstrom, det::X_ASTROM);
                handleNullFp(db, s, X_ASTROM_ERR, s._xAstromErr, det::X_ASTROM_ERR);
                handleNullFp(db, s, Y_ASTROM, s._yAstrom, det::Y_ASTROM);
                handleNullFp(db, s, Y_ASTROM_ERR, s._yAstromErr, det::Y_ASTROM_ERR);
                handleNullFp(db, s, RA_ASTROM, s._raAstrom, det::RA_ASTROM);
                handleNullFp(db, s, RA_ASTROM_ERR, s._raAstromErr, det::RA_ASTROM_ERR);
                handleNullFp(db, s, DEC_ASTROM, s._decAstrom, det::DEC_ASTROM);
                handleNullFp(db, s, DEC_ASTROM_ERR, s._decAstromErr, det::DEC_ASTROM_ERR);

                handleNullFp(db, s, RA_OBJECT, s._raObject, det::RA_OBJECT);
                handleNullFp(db, s, DEC_OBJECT, s._decObject, det::DEC_OBJECT);

                handleNullFp(db, TAI_MID_POINT, s._taiMidPoint);
                handleNullFp(db, s, TAI_RANGE, s._taiRange, det::TAI_RANGE);

                handleNullFp(db, PSF_FLUX, s._psfFlux);
                handleNullFp(db, PSF_FLUX_ERR, s._psfFluxErr);
                handleNullFp(db, AP_FLUX, s._apFlux);
                handleNullFp(db, AP_FLUX_ERR, s._apFluxErr);
                handleNullFp(db, MODEL_FLUX, s._modelFlux);
                handleNullFp(db, MODEL_FLUX_ERR, s._modelFluxErr);
                handleNullFp(db, s, PETRO_FLUX, s._petroFlux, det::PETRO_FLUX);
                handleNullFp(db, s, PETRO_FLUX_ERR, s._petroFluxErr, det::PETRO_FLUX_ERR);
                handleNullFp(db, INST_FLUX, s._instFlux);
                handleNullFp(db, INST_FLUX_ERR, s._instFluxErr);
                handleNullFp(db, s, NON_GRAY_CORR_FLUX,
                             s._nonGrayCorrFlux, det::NON_GRAY_CORR_FLUX);
                handleNullFp(db, s, NON_GRAY_CORR_FLUX_ERR,
                             s._nonGrayCorrFluxErr, det::NON_GRAY_CORR_FLUX_ERR);
                handleNullFp(db, s, ATM_CORR_FLUX,
                             s._atmCorrFlux, det::ATM_CORR_FLUX);
                handleNullFp(db, s, ATM_CORR_FLUX_ERR,
                             s._atmCorrFluxErr, det::ATM_CORR_FLUX_ERR);

                handleNullFp(db, s, AP_DIA, s._apDia, det::AP_DIA);

                handleNullFp(db, s, IXX, s._ixx, det::IXX);
                handleNullFp(db, s, IXX_ERR, s._ixxErr, det::IXX_ERR);
                handleNullFp(db, s, IYY, s._iyy, det::IYY);
                handleNullFp(db, s, IYY_ERR, s._iyyErr, det::IYY_ERR);
                handleNullFp(db, s, IXY, s._ixy, det::IXY);
                handleNullFp(db, s, IXY_ERR, s._ixyErr, det::IXY_ERR);

                handleNullFp(db, s, PSF_IXX, s._psfIxx, det::PSF_IXX);
                handleNullFp(db, s, PSF_IXX_ERR, s._psfIxxErr, det::PSF_IXX_ERR);
                handleNullFp(db, s, PSF_IYY, s._psfIyy, det::PSF_IYY);
                handleNullFp(db, s, PSF_IYY_ERR, s._psfIyyErr, det::PSF_IYY_ERR);
                handleNullFp(db, s, PSF_IXY, s._psfIxy, det::PSF_IXY);
                handleNullFp(db, s, PSF_IXY_ERR, s._psfIxyErr, det::PSF_IXY_ERR);
                handleNullFp(db, s, E1, s._e1, det::E1);
                handleNullFp(db, s, E1_ERR, s._e1Err, det::E1_ERR);
                handleNullFp(db, s, E2, s._e2, det::E2);
                handleNullFp(db, s, E2_ERR, s._e2Err, det::E2_ERR);
                handleNullFp(db, s, SHEAR1, s._shear1, det::SHEAR1);
                handleNullFp(db, s, SHEAR1_ERR, s._shear1Err, det::SHEAR1_ERR);
                handleNullFp(db, s, SHEAR2, s._shear2, det::SHEAR2);
                handleNullFp(db, s, SHEAR2_ERR, s._shear2Err, det::SHEAR2_ERR);
                handleNullFp(db, s, RESOLUTION, s._resolution, det::RESOLUTION);
                handleNullFp(db, s, SIGMA, s._sigma, det::SIGMA);
                handleNullFp(db, s, SIGMA_ERR, s._sigmaErr, det::SIGMA_ERR);
                handleNullFp(db, s, SHAPE_STATUS, s._shapeStatus, det::SHAPE_STATUS);

                
                handleNullFp(db, SNR, s._snr);
                handleNullFp(db, CHI2, s._chi2);
                handleNullFp(db, s, SKY, s._sky, det::SKY);
                handleNullFp(db, s, SKY_ERR, s._skyErr, det::SKY_ERR);

                if (db->columnIsNull(FLAG_FOR_ASSOCIATION)) { 
                    s.setNull(det::FLAG_FOR_ASSOCIATION);
                }
                if (db->columnIsNull(FLAG_FOR_DETECTION)) { 
                    s.setNull(det::FLAG_FOR_DETECTION); 
                }
                if (db->columnIsNull(FLAG_FOR_WCS)) { 
                    s.setNull(det::FLAG_FOR_WCS); 
                }

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
